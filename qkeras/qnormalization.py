# Copyright 2019 Google LLC
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
#
# ==============================================================================
"""Definition of normalization quantization package."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import six
import warnings

import tensorflow.compat.v2 as tf

from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond as tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from .qlayers import Clip
from .qlayers import get_auto_range_constraint_initializer
from .qlayers import get_quantizer
from .quantizers import quantized_relu_po2
from .quantizers import quantized_po2
from .safe_eval import safe_eval
from tensorflow_model_optimization.python.core.sparsity.keras.prunable_layer import PrunableLayer


class QBatchNormalization(BatchNormalization, PrunableLayer):
  """Quantized Batch Normalization layer.
  For training, mean and variance are not quantized.
  For inference, the quantized moving mean and moving variance are used.

  output = (x - mean) / sqrt(var + epsilon) * quantized_gamma + quantized_beta

  """

  def __init__(
      self,
      axis=-1,
      momentum=0.99,
      epsilon=1e-3,
      center=True,
      scale=True,
      activation=None,
      beta_initializer='zeros',
      gamma_initializer='ones',
      moving_mean_initializer='zeros',
      moving_variance_initializer='ones',
      beta_regularizer=None,
      gamma_regularizer=None,
      beta_quantizer='quantized_po2(5)',
      gamma_quantizer='quantized_relu_po2(6, 2048)',
      mean_quantizer='quantized_po2(5)',
      variance_quantizer='quantized_relu_po2(6, quadratic_approximation=True)',
      gamma_constraint=None,
      beta_constraint=None,
      # use quantized_po2 and enforce quadratic approximation
      # to get an even exponent for sqrt
      beta_range=None,
      gamma_range=None,
      **kwargs):

    if gamma_range is not None:
      warnings.warn('gamma_range is deprecated in QBatchNormalization layer.')

    if beta_range is not None:
      warnings.warn('beta_range is deprecated in QBatchNormalization layer.')

    self.gamma_range = gamma_range
    self.beta_range = beta_range
    self.activation = activation


    self.beta_quantizer = beta_quantizer
    self.gamma_quantizer = gamma_quantizer
    self.mean_quantizer = mean_quantizer
    self.variance_quantizer = variance_quantizer

    self.beta_quantizer_internal = get_quantizer(self.beta_quantizer)
    self.gamma_quantizer_internal = get_quantizer(self.gamma_quantizer)
    self.mean_quantizer_internal = get_quantizer(self.mean_quantizer)
    self.variance_quantizer_internal = get_quantizer(self.variance_quantizer)

    if hasattr(self.gamma_quantizer_internal, '_set_trainable_parameter'):
      self.gamma_quantizer_internal._set_trainable_parameter()
    if hasattr(self.variance_quantizer_internal, '_set_trainable_parameter'):
      self.variance_quantizer_internal._set_trainable_parameter()

    self.quantizers = [
        self.gamma_quantizer_internal,
        self.beta_quantizer_internal,
        self.mean_quantizer_internal,
        self.variance_quantizer_internal
    ]

    if scale and self.gamma_quantizer:
      gamma_constraint, gamma_initializer = (
          get_auto_range_constraint_initializer(
              self.gamma_quantizer_internal,
              gamma_constraint,
              gamma_initializer)
      )

    if center and self.beta_quantizer:
      beta_constraint, beta_initializer = (
          get_auto_range_constraint_initializer(
              self.beta_quantizer_internal,
              beta_constraint,
              beta_initializer)
      )

    if kwargs.get('fused', None):
      warnings.warn('batch normalization fused is disabled '
                    'in qkeras qnormalization.py.')
      del kwargs['fused']

    if kwargs.get('renorm', None):
      warnings.warn('batch normalization renorm is disabled '
                    'in qkeras qnormalization.py.')
      del kwargs['renorm']

    if kwargs.get('virtual_batch_size', None):
      warnings.warn('batch normalization virtual_batch_size is disabled '
                    'in qkeras qnormalization.py.')
      del kwargs['virtual_batch_size']

    if kwargs.get('adjustment', None):
      warnings.warn('batch normalization adjustment is disabled '
                    'in qkeras qnormalization.py.')
      del kwargs['adjustment']

    super(QBatchNormalization, self).__init__(
        axis=axis,
        momentum=momentum,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer=beta_initializer,
        gamma_initializer=gamma_initializer,
        moving_mean_initializer=moving_mean_initializer,
        moving_variance_initializer=moving_variance_initializer,
        beta_regularizer=beta_regularizer,
        gamma_regularizer=gamma_regularizer,
        beta_constraint=beta_constraint,
        gamma_constraint=gamma_constraint,
        fused=False,
        renorm=False,
        virtual_batch_size=None,
        adjustment=None,
        **kwargs)

  def call(self, inputs, training=None):
    if self.scale and self.gamma_quantizer:
      quantized_gamma = self.gamma_quantizer_internal(self.gamma)
    else:
      quantized_gamma = self.gamma

    if self.center and self.beta_quantizer:
      quantized_beta = self.beta_quantizer_internal(self.beta)
    else:
      quantized_beta = self.beta

    if self.mean_quantizer:
      quantized_moving_mean = self.mean_quantizer_internal(self.moving_mean)
    else:
      quantized_moving_mean = self.moving_mean

    if self.variance_quantizer:
      quantized_moving_variance = self.variance_quantizer_internal(
          self.moving_variance)
    else:
      quantized_moving_variance = self.moving_variance

    training = self._get_training_value(training)

    # Compute the axes along which to reduce the mean / variance
    input_shape = inputs.shape
    ndims = len(input_shape)
    reduction_axes = [i for i in range(ndims) if i not in self.axis]

    # Broadcasting only necessary for single-axis batch norm where the axis is
    # not the last dimension
    broadcast_shape = [1] * ndims
    broadcast_shape[self.axis[0]] = input_shape.dims[self.axis[0]].value
    def _broadcast(v):
      if (v is not None and len(v.shape) != ndims and
          reduction_axes != list(range(ndims - 1))):
        return array_ops.reshape(v, broadcast_shape)
      return v

    scale, offset = _broadcast(quantized_gamma), _broadcast(quantized_beta)

    # Determine a boolean value for `training`: could be True, False, or None.
    training_value = tf_utils.smart_constant_value(training)
    if training_value == False:  # pylint: disable=singleton-comparison,g-explicit-bool-comparison
      quantized_mean, quantized_variance = (quantized_moving_mean,
                                            quantized_moving_variance)
    else:
      # Some of the computations here are not necessary when training==False
      # but not a constant. However, this makes the code simpler.
      keep_dims = len(self.axis) > 1
      mean, variance = self._moments(
          math_ops.cast(inputs, self._param_dtype),
          reduction_axes,
          keep_dims=keep_dims)

      moving_mean = self.moving_mean
      moving_variance = self.moving_variance

      mean = tf_utils.smart_cond(
          training, lambda: mean, lambda: ops.convert_to_tensor(moving_mean))
      variance = tf_utils.smart_cond(
          training,
          lambda: variance,
          lambda: ops.convert_to_tensor(moving_variance))

      new_mean, new_variance = mean, variance

      if self.mean_quantizer:
        quantized_mean = self.mean_quantizer_internal(mean)
      else:
        quantized_mean = mean

      if self.variance_quantizer:
        quantized_variance = self.variance_quantizer_internal(variance)
      else:
        quantized_variance = variance

      if self._support_zero_size_input():
        inputs_size = array_ops.size(inputs)
      else:
        inputs_size = None

      def _do_update(var, value):
        """Compute the updates for mean and variance."""
        return self._assign_moving_average(var, value, self.momentum,
                                           inputs_size)

      def mean_update():
        true_branch = lambda: _do_update(self.moving_mean, new_mean)
        false_branch = lambda: self.moving_mean
        return tf_utils.smart_cond(training, true_branch, false_branch)

      def variance_update():
        """Update the moving variance."""
        true_branch = lambda: _do_update(self.moving_variance, new_variance)
        false_branch = lambda: self.moving_variance
        return tf_utils.smart_cond(training, true_branch, false_branch)

      self.add_update(mean_update)
      self.add_update(variance_update)

    quantized_mean = math_ops.cast(quantized_mean, inputs.dtype)
    quantized_variance = math_ops.cast(quantized_variance, inputs.dtype)
    if offset is not None:
      offset = math_ops.cast(offset, inputs.dtype)
    if scale is not None:
      scale = math_ops.cast(scale, inputs.dtype)
    # TODO(reedwm): Maybe do math in float32 if given float16 inputs, if doing
    # math in float16 hurts validation accuracy of popular models like resnet.
    outputs = nn.batch_normalization(inputs,
                                     _broadcast(quantized_mean),
                                     _broadcast(quantized_variance),
                                     offset,
                                     scale,
                                     self.epsilon)
    # If some components of the shape got lost due to adjustments, fix that.
    outputs.set_shape(input_shape)

    return outputs

  def get_config(self):
    config = {
        'axis': self.axis,
        'momentum': self.momentum,
        'epsilon': self.epsilon,
        'center': self.center,
        'scale': self.scale,
        'beta_quantizer':
            constraints.serialize(self.beta_quantizer_internal),
        'gamma_quantizer':
            constraints.serialize(self.gamma_quantizer_internal),
        'mean_quantizer':
            constraints.serialize(self.mean_quantizer_internal),
        'variance_quantizer':
            constraints.serialize(self.variance_quantizer_internal),
        'beta_initializer': initializers.serialize(self.beta_initializer),
        'gamma_initializer': initializers.serialize(self.gamma_initializer),
        'moving_mean_initializer':
            initializers.serialize(self.moving_mean_initializer),
        'moving_variance_initializer':
            initializers.serialize(self.moving_variance_initializer),
        'beta_regularizer': regularizers.serialize(self.beta_regularizer),
        'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
        'beta_constraint': constraints.serialize(self.beta_constraint),
        'gamma_constraint': constraints.serialize(self.gamma_constraint),
        'beta_range': self.beta_range,
        'gamma_range': self.gamma_range,
    }
    base_config = super(QBatchNormalization, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_quantizers(self):
    return self.quantizers

  def get_prunable_weights(self):
    return []

