# Copyright 2020 Google LLC
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Fold batchnormalization with previous QDense layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import warnings

import numpy as np
import six


from qkeras.qlayers import QDense
from qkeras.quantizers import *

import tensorflow.compat.v2 as tf
from tensorflow.keras import layers
from tensorflow.python.framework import smart_cond as tf_utils
from tensorflow.python.ops import math_ops

tf.compat.v2.enable_v2_behavior()


class QDenseBatchnorm(QDense):
  """Implements a quantized Dense layer fused with Batchnorm."""

  def __init__(
    self,
    units,
    activation=None,
    use_bias=True,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    kernel_quantizer=None,
    bias_quantizer=None,
    kernel_range=None,
    bias_range=None,

    # batchnorm params
    axis=-1,
    momentum=0.99,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer="zeros",
    gamma_initializer="ones",
    moving_mean_initializer="zeros",
    moving_variance_initializer="ones",
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    renorm=False,
    renorm_clipping=None,
    renorm_momentum=0.99,
    fused=None,
    trainable=True,
    virtual_batch_size=None,
    adjustment=None,

    # other params
    ema_freeze_delay=None,
    folding_mode="ema_stats_folding",
    **kwargs):

    super(QDenseBatchnorm, self).__init__(
      units=units,
      activation=activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      bias_regularizer=bias_regularizer,
      activity_regularizer=activity_regularizer,
      kernel_constraint=kernel_constraint,
      bias_constraint=bias_constraint,
      kernel_quantizer=kernel_quantizer,
      bias_quantizer=bias_quantizer,
      kernel_range=kernel_range,
      bias_range=bias_range,
      **kwargs)
    
    # initialization of batchnorm part of the composite layer
    self.batchnorm = layers.BatchNormalization(
      axis=axis, momentum=momentum, epsilon=epsilon, center=center,
      scale=scale, beta_initializer=beta_initializer,
      gamma_initializer=gamma_initializer,
      moving_mean_initializer=moving_mean_initializer,
      moving_variance_initializer=moving_variance_initializer,
      beta_regularizer=beta_regularizer,
      gamma_regularizer=gamma_regularizer,
      beta_constraint=beta_constraint, gamma_constraint=gamma_constraint,
      renorm=renorm, renorm_clipping=renorm_clipping,
      renorm_momentum=renorm_momentum, fused=fused, trainable=trainable,
      virtual_batch_size=virtual_batch_size, adjustment=adjustment
      )

    self.ema_freeze_delay = ema_freeze_delay
    assert folding_mode in ["ema_stats_folding", "batch_stats_folding"]
    self.folding_mode = folding_mode          

  def build(self, input_shape):
    super(QDenseBatchnorm, self).build(input_shape)

    # self._iteration (i.e., training_steps) is initialized with -1. When
    # loading ckpt, it can load the number of training steps that have been
    # previously trainied. If start training from scratch.
    # TODO(lishanok): develop a way to count iterations outside layer
    self._iteration = tf.Variable(-1, trainable=False, name="iteration",
                                  dtype=tf.int64)

  def call(self, inputs, training=None):

    # numpy value, mark the layer is in training
    training = self.batchnorm._get_training_value(training)  # pylint: disable=protected-access

    # checking if to update batchnorm params
    if (self.ema_freeze_delay is None) or (self.ema_freeze_delay < 0):
      # if ema_freeze_delay is None or a negative value, do not freeze bn stats
      bn_training = tf.cast(training, dtype=bool)
    else:
      bn_training = tf.math.logical_and(training, tf.math.less_equal(
        self._iteration, self.ema_freeze_delay))

    kernel = self.kernel
    
    #execute qdense output
    qdense_outputs = tf.keras.backend.dot(
      inputs, 
      kernel
      )

    if self.use_bias:
      bias = self.bias
      qdense_outputs = tf.keras.backend.bias_add(
        qdense_outputs, 
        bias,
        data_format="channels_last")
    else:
      bias = 0
    # If loaded from a ckpt, bias_quantizer is the ckpt value
    # Else if the layer is called for the first time, in this case bias
    #   quantizer is None and we need to calculate bias quantizer
    #   type according to accumulator type
    if self.bias_quantizer_internal is not None:
        q_bias = self.bias_quantizer_internal(bias)
    else:
        q_bias = bias

    # begin batchnorm
    _ = self.batchnorm(qdense_outputs, training=bn_training)

    self._iteration.assign_add(tf_utils.smart_cond(
        training, lambda: tf.constant(1, tf.int64),
        lambda: tf.constant(0, tf.int64)))
    
    # calculate mean and variance from current batch
    bn_shape = qdense_outputs.shape
    ndims = len(bn_shape)
    reduction_axes = [i for i in range(ndims) if i not in self.batchnorm.axis]
    keep_dims = len(self.batchnorm.axis) > 1
    mean, variance = self.batchnorm._moments(  # pylint: disable=protected-access
        math_ops.cast(qdense_outputs, self.batchnorm._param_dtype),  # pylint: disable=protected-access
        reduction_axes,
        keep_dims=keep_dims)

    # get batchnorm weights
    gamma = self.batchnorm.gamma
    beta = self.batchnorm.beta
    moving_mean = self.batchnorm.moving_mean
    moving_variance = self.batchnorm.moving_variance

    if self.folding_mode == "batch_stats_folding":
      # using batch mean and variance in the initial training stage
      # after sufficient training, switch to moving mean and variance
      new_mean = tf_utils.smart_cond(
        bn_training, lambda: mean, lambda: moving_mean)
      new_variance = tf_utils.smart_cond(
        bn_training, lambda: variance, lambda: moving_variance)

      # get the inversion factor so that we replace division by multiplication
      inv = math_ops.rsqrt(new_variance + self.batchnorm.epsilon)
      if gamma is not None:
        inv *= gamma

      # fold bias with bn stats
      folded_bias = inv * (q_bias - new_mean) + beta

    elif self.folding_mode == "ema_stats_folding":
        # We always scale the weights with a correction factor to the long term
        # statistics prior to quantization. This ensures that there is no jitter
        # in the quantized weights due to batch to batch variation. During the
        # initial phase of training, we undo the scaling of the weights so that
        # outputs are identical to regular batch normalization. We also modify
        # the bias terms correspondingly. After sufficient training, switch from
        # using batch statistics to long term moving averages for batch
        # normalization.

        # use batch stats for calcuating bias before bn freeze, and use moving
        # stats after bn freeze
        mv_inv = math_ops.rsqrt(moving_variance + self.batchnorm.epsilon)
        batch_inv = math_ops.rsqrt(variance + self.batchnorm.epsilon)

        if gamma is not None:
            mv_inv *= gamma
            batch_inv *= gamma
        folded_bias = tf_utils.smart_cond(
          bn_training,
          lambda: batch_inv * (q_bias - mean) + beta,
          lambda: mv_inv * (q_bias - moving_mean) + beta)
        # moving stats is always used to fold kernel in tflite; before bn freeze
        # an additional correction factor will be applied to the conv2d output
        # end batchnorm 
        inv = mv_inv
    else:
        assert ValueError

    # quantize the folded kernel
    if self.kernel_quantizer is not None:
        q_kernel = self.kernel_quantizer_internal(kernel)
    else:
        q_kernel = kernel
    # wrap qdense kernel with bn parameters
    folded_kernel = inv * q_kernel

    applied_kernel = folded_kernel
    applied_bias = folded_bias
    
    #calculate qdense output using the quantized folded kernel
    folded_outputs = tf.keras.backend.dot(inputs, applied_kernel)

    if training is True and self.folding_mode == "ema_stats_folding":
      batch_inv = math_ops.rsqrt(variance + self.batchnorm.epsilon)
      y_corr = tf_utils.smart_cond(
          bn_training,
          lambda: (math_ops.sqrt(moving_variance + self.batchnorm.epsilon) *
                   math_ops.rsqrt(variance + self.batchnorm.epsilon)),
          lambda: tf.constant(1.0, shape=moving_variance.shape))
      folded_outputs = math_ops.mul(folded_outputs, y_corr)

    folded_outputs = tf.keras.backend.bias_add(
      folded_outputs, 
      applied_bias,
      data_format="channels_last"
    )
    
    if self.activation is not None:
      return self.activation(folded_outputs)
    
    return folded_outputs

  def get_config(self):
    base_config = super().get_config()
    bn_config = self.batchnorm.get_config()
    config = {"ema_freeze_delay": self.ema_freeze_delay,
              "folding_mode": self.folding_mode}
    name = base_config["name"]
    out_config = dict(
        list(base_config.items())
        + list(bn_config.items()) + list(config.items()))

    # names from different config override each other; use the base layer name
    # as the this layer's config name
    out_config["name"] = name
    return out_config

  def get_quantization_config(self):
    return {
        "kernel_quantizer": str(self.kernel_quantizer_internal),
        "bias_quantizer": str(self.bias_quantizer_internal),
    }
    def get_quantizers(self):
      return self.quantizers

  # def get_prunable_weights(self):
  #   return [self.kernel]

  def get_folded_weights(self):
    """Function to get the batchnorm folded weights.
    This function converts the weights by folding batchnorm parameters into
    the weight of QDense. The high-level equation:
    W_fold = gamma * W / sqrt(variance + epsilon)
    bias_fold = gamma * (bias - moving_mean) / sqrt(variance + epsilon) + beta
    """

    kernel = self.kernel
    if self.use_bias:
      bias = self.bias
    else:
      bias = 0

    # get batchnorm weights and moving stats
    gamma = self.batchnorm.gamma
    beta = self.batchnorm.beta
    moving_mean = self.batchnorm.moving_mean
    moving_variance = self.batchnorm.moving_variance

    # get the inversion factor so that we replace division by multiplication
    inv = math_ops.rsqrt(moving_variance + self.batchnorm.epsilon)
    if gamma is not None:
      inv *= gamma

    # wrap conv kernel and bias with bn parameters
    folded_kernel = inv * kernel
    folded_bias = inv * (bias - moving_mean) + beta

    return [folded_kernel, folded_bias]