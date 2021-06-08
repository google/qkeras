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
"""Definition of quantization package."""

# Some parts of the code were adapted from
#
# https://github.com/BertMoons/QuantizedNeuralNetworks-Keras-Tensorflow
#
# "Copyright (c) 2017, Bert Moons" where it applies
#
# and were implemented following several papers.
#
#    https://arxiv.org/pdf/1609.07061.pdf
#    https://arxiv.org/abs/1602.02830
#    https://arxiv.org/abs/1603.05279
#    https://arxiv.org/abs/1605.04711
#    https://ieeexplore.ieee.org/abstract/document/6986082
#    https://ieeexplore.ieee.org/iel4/78/5934/00229903.pdf
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import warnings

import numpy as np
import six
import tensorflow.compat.v2 as tf
from tensorflow.keras import activations
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Layer
from tensorflow.python.framework import smart_cond as tf_utils

from .quantizers import *
from .quantizers import _get_integer_bits
from .quantizers import get_quantizer
from tensorflow_model_optimization.python.core.sparsity.keras.prunable_layer import PrunableLayer


def get_auto_range_constraint_initializer(quantizer, constraint, initializer):
  """Get value range automatically for quantizer.

  Arguments:
   quantizer: A quantizer class in quantizers.py.
   constraint: A tf.keras constraint.
   initializer: A tf.keras initializer.

  Returns:
    a tuple (constraint, initializer), where
      constraint is clipped by Clip class in this file, based on the
      value range of quantizer.
      initializer is initializer contraint by value range of quantizer.
  """
  if quantizer is not None:
    constraint = get_constraint(constraint, quantizer)
    initializer = get_initializer(initializer)

    if initializer and initializer.__class__.__name__ not in ["Ones", "Zeros", 'QInitializer']:
      # we want to get the max value of the quantizer that depends
      # on the distribution and scale
      if not (hasattr(quantizer, "alpha") and
              isinstance(quantizer.alpha, six.string_types)):
        initializer = QInitializer(
            initializer, use_scale=True, quantizer=quantizer)
  return constraint, initializer


class QInitializer(Initializer):
  """Wraps around Keras initializer to provide a fanin scaling factor."""

  def __init__(self, initializer, use_scale, quantizer):
    self.initializer = initializer
    self.use_scale = use_scale
    self.quantizer = quantizer

    try:
      self.is_po2 = "po2" in quantizer.__class__.__name__
    except:
      self.is_po2 = False

  def __call__(self, shape, dtype=None):
    x = self.initializer(shape, dtype)

    max_x = np.max(abs(x))
    std_x = np.std(x)
    delta = self.quantizer.max() * 2**-self.quantizer.bits

    # delta is the minimum resolution of the number system.
    # we want to make sure we have enough values.
    if delta > std_x and hasattr(self.initializer, "scale"):
      q = self.quantizer(x)
      max_q = np.max(abs(q))
      scale = 1.0
      if max_q == 0.0:
        xx = np.mean(x * x)
        scale = self.quantizer.max() / np.sqrt(xx)
      else:
        qx = np.sum(q * x)
        qq = np.sum(q * q)

        scale = qq / qx

      self.initializer.scale *= max(scale, 1)
      x = self.initializer(shape, dtype)

    return np.clip(x, -self.quantizer.max(), self.quantizer.max())

  def get_config(self):
    return {
        "initializer": self.initializer,
        "use_scale": self.use_scale,
        "quantizer": self.quantizer,
    }

  @classmethod
  def from_config(cls, config):
    config = {
      'initializer' : get_initializer(config['initializer']),
      'use_scale'   : config['use_scale'],
      'quantizer'   : get_quantizer(config['quantizer'])}
    return cls(**config)

#
# Because it may be hard to get serialization from activation functions,
# we may be replacing their instantiation by QActivation in the future.
#


class QActivation(Layer, PrunableLayer):
  """Implements quantized activation layers."""

  def __init__(self, activation, **kwargs):

    super(QActivation, self).__init__(**kwargs)

    self.activation = activation

    if not isinstance(activation, six.string_types):
      self.quantizer = activation
      if hasattr(self.quantizer, "__name__"):
        self.__name__ = self.quantizer.__name__
      elif hasattr(self.quantizer, "name"):
        self.__name__ = self.quantizer.name
      elif hasattr(self.quantizer, "__class__"):
        self.__name__ = self.quantizer.__class__.__name__
      return

    self.__name__ = activation

    try:
      self.quantizer = get_quantizer(activation)
    except KeyError:
      raise ValueError("invalid activation '{}'".format(activation))

  def call(self, inputs):
    return self.quantizer(inputs)

  def get_config(self):
    config = {"activation": self.activation}
    base_config = super(QActivation, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_quantization_config(self):
    return str(self.activation)

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_prunable_weights(self):
    return []


class QAdaptiveActivation(Layer, PrunableLayer):
  """[EXPERIMENTAL] Implements an adaptive quantized activation layer using EMA.

  This layer calculates an exponential moving average of min and max of the
  activation values to automatically determine the scale (integer bits) of
  the quantizer used in this layer.
  """

  def __init__(self,
               activation,
               total_bits,
               current_step=None,
               symmetric=True,
               quantization_delay=0,
               ema_freeze_delay=None,
               ema_decay=0.9999,
               per_channel=False,
               po2_rounding=False,
               relu_neg_slope=0.0,
               relu_upper_bound=None,
               **kwargs):
    """Initializes this QAdaptiveActivation layer.

    Args:
      activation: Str. The activation quantizer type to use for this activation
        layer, such as 'quantized_relu'. Should be a string with no params.
      total_bits: Int. The total bits that can be used by the quantizer
      current_step: tf.Variable specifying the current step in training.
        You can find this by passing model.optimizer.iterations
        (see tf.keras.optimizers.Optimizer.iterations). If set to None, the
        layer will attempt to estimate the current step itself, but please note
        that this number may not always match the optimizer step.
      symmetric: Bool. If to enforce symmetry about the origin in the quantized
        bit representation of the value. When using linear activation, this
        should be True for best results.
      quantization_delay: Int. How many training steps to wait until quantizing
        the activation values.
      ema_freeze_delay: Int. Steps to wait until stopping the update of the
        exponential moving average values. Set to None for an infinite delay.
      ema_decay: Float. The decay value used for exponential moving average (see
        tf.keras.backend.moving_average_update)
      per_channel: Bool. If to quantize the activation values on a
        per-channel basis.
      po2_rounding: Bool. If true, the EMA max value is rounded to the nearest
        power-of-2. If false, the EMA max value is rounded up (with ceil) to a
        power-of-two. These power-of-two operations are necessary to calculate
        the number of integer bits used in the quantizer, and the difference
        between using round and ceil trade off the quantizer's range and
        precision.
      relu_neg_slope: Float. Slope of the negative values in relu to enable the
        use of leaky relu. This parameter will only be used with the quantizer
        type quantized_relu. Set to 0.0 to use normal relu.
      relu_upper_bound: Float. The upper bound to use if the activation is set
        to relu. Set to None to not artificially set an upper bound. Pease note
        that this param is ignored if the activation is not quantized_relu
      **kwargs: Args passed to the Layer class.
    """
    super(QAdaptiveActivation, self).__init__(**kwargs)

    self.total_bits = total_bits
    self.symmetric = symmetric
    self.is_estimating_step_count = False  # If the layer should estimate its
                                           # own step count by incrementing it
                                           # every call.
    if isinstance(current_step, tf.Variable):
      self.step = current_step
    elif current_step is None:
      self.step = tf.Variable(-1, dtype=tf.int64)
      self.is_estimating_step_count = True
      print("[WARNING] QAdaptiveActivation is estimating it's own training "
            "step count, which may not always be the same as the true optimizer"
            " training step. To mitigate this, please set the current_step "
            "parameter when initializing QAdaptiveActivation", file=sys.stderr)
    else:
      self.step = tf.Variable(current_step, dtype=tf.int64)
      print("[WARNING] QAdaptiveActivation is disconnected from the optimizer "
            "current step, which may lead to incorrect training. If you wish to"
            " resume training, set this layer's self.step to the optimizer's "
            "tf.Variable current step", file=sys.stderr)
    self.quantization_delay = quantization_delay
    self.ema_freeze_delay = ema_freeze_delay
    self.will_ema_freeze = True if ema_freeze_delay else False
    self.ema_decay = ema_decay
    self.per_channel = per_channel
    self.po2_rounding = po2_rounding
    self.ema_min = None
    self.ema_max = None
    self.relu_neg_slope = relu_neg_slope
    self.relu_upper_bound = relu_upper_bound

    # Verify quantizer type is correct
    self.supported_quantizers = ["quantized_bits", "quantized_relu"]
    if activation not in self.supported_quantizers:
      raise ValueError(("Invalid activation {}. Activation quantizer may NOT "
                        "contain any parameters (they will be set automatically"
                        " by this layer), and only the quantizer types {} are "
                        "supported.").format(activation,
                                             self.supported_quantizers))

    # Get the quantizer associated with the activation
    try:
      self.quantizer = get_quantizer(activation)
    except KeyError:
      raise ValueError("Invalid activation '{}'".format(activation))

    # Check that the quantizer is supported
    if self.quantizer.__class__.__name__ not in self.supported_quantizers:
      raise ValueError("Unsupported activation quantizer '{}'".format(
          self.quantizer.__class__.__name__))

    # Set keep_negative
    if self.quantizer.__class__.__name__ == "quantized_relu":
      self.quantizer.is_quantized_clip = False  # Use relu_upper_bound instead
      if self.relu_upper_bound:
        self.quantizer.relu_upper_bound = self.relu_upper_bound
      self.quantizer.negative_slope = relu_neg_slope
      self.keep_negative = relu_neg_slope != 0.0
      self.quantizer.is_quantized_clip = False  # Use normal relu when qnoise=0
    elif self.quantizer.__class__.__name__ == "quantized_bits":
      self.keep_negative = True
      self.quantizer.keep_negative = True

    # If not using quantization delay, then print warning
    if self.quantization_delay < 1:
      print("[WARNING] If QAdaptiveActivation has the quantization_delay set "
            "to 0, then the moving averages will be heavily biased towards the "
            "initial quantizer configuration, which will likely prevent the "
            "model from converging. Consider a larger quantization_delay.",
            file=sys.stderr)

    self.activation = self.quantizer  # self.activation is used by QTools

  def build(self, input_shape):
    if self.will_ema_freeze:
      self.ema_freeze_delay = tf.constant(self.ema_freeze_delay, dtype=tf.int64)

    self.ema_decay = tf.constant(self.ema_decay, dtype=tf.float32)
    self.is_estimating_step_count = tf.constant(self.is_estimating_step_count,
                                                dtype=tf.bool)

    # Calculate the number of channels
    channel_index = -1 if K.image_data_format() == "channels_last" else 1
    if self.per_channel:
      input_shape_list = list(input_shape) if isinstance(
          input_shape, tuple) else input_shape.as_list()
      num_channels = tf.constant(input_shape_list[channel_index],
                                 shape=(1), dtype=tf.int64)
    else:
      num_channels = tf.constant(1, shape=(1), dtype=tf.int64)

    # Initialize the moving mins and max
    if self.ema_min is None or self.ema_max is None:
      self.ema_min = tf.Variable(tf.zeros(num_channels), name="ema_min",
                                 trainable=False)
      self.ema_max = tf.Variable(tf.zeros(num_channels), name="ema_max",
                                 trainable=False)

    # Determine the parameters for the quantizer
    self.quantizer.bits = self.total_bits

    # Set up the initial integer bits and quantizer params
    self.quantizer.integer = tf.Variable(tf.zeros(num_channels,
                                                  dtype=tf.int32),
                                         name="quantizer_integer_bits",
                                         trainable=False)
    integer_bits = _get_integer_bits(min_value=self.ema_min,
                                     max_value=self.ema_max,
                                     bits=self.total_bits,
                                     symmetric=self.symmetric,
                                     keep_negative=self.keep_negative,
                                     is_clipping=self.po2_rounding)
    self.quantizer.integer.assign(integer_bits)
    self.quantizer.alpha = 1.0  # Setting alpha to 1.0 allows the integer bits
                                # to serve as the scale
    self.quantizer.symmetric = self.symmetric
    self.quantization_delay = tf.constant(self.quantization_delay,
                                          dtype=tf.int64)

  def call(self, inputs, training=False):
    x = inputs
    training = training and self.trainable
    self.will_ema_freeze = self.will_ema_freeze and self.trainable

    # Update the step count if the optimizer step count is unknown
    self.step.assign_add(K.switch(
        tf.math.logical_and(self.is_estimating_step_count, training),
        tf.constant(1, tf.int64), tf.constant(0, tf.int64)))

    # Perform the quantization
    if training:
      # Calculate the qnoise, a scalar from 0 to 1 that represents the level of
      # quantization noise to use. At training start, we want no quantization,
      # so qnoise_factor = 0.0. After quantization_delay steps, we want normal
      # quantization, so qnoise_factor = 1.0.
      qnoise_factor = K.switch(
          tf.greater_equal(self.step, self.quantization_delay),
          lambda: tf.constant(1.0), lambda: tf.constant(0.0))
      self.quantizer.update_qnoise_factor(qnoise_factor)
      qx = self.quantizer(x)

    else:  # If not training, we always want to use full quantization
      self.quantizer.update_qnoise_factor(tf.constant(1.0))
      qx = self.quantizer(x)

    # Calculate the axis along where to find the min and max EMAs
    len_axis = len(x.shape)
    if len_axis > 1:
      if self.per_channel:
        if K.image_data_format() == "channels_last":
          axis = list(range(len_axis - 1))
        else:
          axis = list(range(1, len_axis))
      else:
        axis = list(range(len_axis))
    else:
      axis = [0]

    # Determine if freezing the EMA
    is_ema_training = tf.constant(training, dtype=tf.bool)
    if self.will_ema_freeze:
      is_ema_training = tf.cond(
          tf.greater(self.step, self.ema_freeze_delay),
          lambda: tf.constant(False), lambda: tf.constant(True))

    def update_branch():
      """ Update the moving average when is_ema_training is True."""

      # Set the qnoise factor to 0 to update the EMA using the unquantized input
      prev_qnoise_factor = tf.identity(self.quantizer.qnoise_factor)
      self.quantizer.update_qnoise_factor(tf.constant(0.0))

      # Update the EMA
      act_x = self.quantizer(x)  # act_x is the input after the activation
                                 # function, but before the quantizer. This is
                                 # done by using a qnoise_factor of 0
      new_min = tf.squeeze(K.min(act_x, axis=axis, keepdims=True))
      K.moving_average_update(self.ema_min, new_min, self.ema_decay)
      new_max = tf.squeeze(K.max(act_x, axis=axis, keepdims=True))
      K.moving_average_update(self.ema_max, new_max, self.ema_decay)

      # Reset the qnoise factor to the previous value
      self.quantizer.update_qnoise_factor(prev_qnoise_factor)

    # Update the moving average when is_ema_training is True
    tf_utils.smart_cond(
        is_ema_training, true_fn=update_branch, false_fn=lambda: None)

    # Set the integer bits for the quantizer
    integer_bits = _get_integer_bits(
        min_value=self.ema_min,
        max_value=self.ema_max,
        bits=self.total_bits,
        symmetric=self.symmetric,
        keep_negative=self.keep_negative,
        is_clipping=self.po2_rounding)
    self.quantizer.integer.assign(integer_bits)

    return qx

  # Override get_weights since we do not want ema_min or ema_max to be public
  def get_weights(self):
    return []

  # Override set_weights since we do not want ema_min or ema_max to be public
  def set_weights(self, weights):
    return

  def get_config(self):
    config = {
        "activation": self.quantizer.__class__.__name__,
        "total_bits": self.total_bits,
        "current_step": self.step.numpy(),
        "symmetric": self.symmetric,
        "quantization_delay": np.array(self.quantization_delay),
        "ema_freeze_delay": np.array(self.ema_freeze_delay),
        "ema_decay": np.array(self.ema_decay),
        "per_channel": self.per_channel,
        "po2_rounding": self.po2_rounding,
        "relu_neg_slope": self.relu_neg_slope
    }
    base_config = super(QAdaptiveActivation, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_quantization_config(self):
    self.quantizer.integer_bits = np.array(self.quantizer)
    return str(self.quantizer)

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_prunable_weights(self):
    return []


#
# Constraint class to clip weights and bias between -1 and 1 so that:
#    1. quantization approximation is symmetric (b = 0).
#    2. max(x) and min(x) are 1 and -1 respectively.
#
class Clip(Constraint):
  """Clips weight constraint."""

  # This function was modified from Keras minmaxconstraints.
  #
  # Constrains the weights to be between min/max values.
  #   min_value: the minimum norm for the incoming weights.
  #   max_value: the maximum norm for the incoming weights.
  #   constraint: previous constraint to be clipped.
  #   quantizer: quantizer to be applied to constraint.

  def __init__(self, min_value=0.0, max_value=1.0,
               constraint=None, quantizer=None):
    """Initializes Clip constraint class."""

    self.min_value = min_value
    self.max_value = max_value
    self.constraint = constraints.get(constraint)
    # Don't wrap yourself
    if isinstance(self.constraint, Clip):
      self.constraint = None
    self.quantizer = get_quantizer(quantizer)

  def __call__(self, w):
    """Clips values between min and max values."""
    if self.constraint:
      w = self.constraint(w)
      if self.quantizer:
        w = self.quantizer(w)
    w = tf.keras.backend.clip(w, self.min_value, self.max_value)
    return w

  def get_config(self):
    """Returns configuration of constraint class."""
    return {"min_value": self.min_value, "max_value": self.max_value}

  @classmethod
  def from_config(cls, config):
    if isinstance(config.get('constraint', None), Clip):
      config['constraint'] = None
    config['constraint'] = constraints.get(config.get('constraint', None))
    config['quantizer'] = get_quantizer(config.get('quantizer', None))
    return cls(**config)

#
# Definition of Quantized NN classes. These classes were copied
# from the equivalent layers in Keras, and we modified to apply quantization.
# Similar implementations can be seen in the references.
#


class QDense(Dense, PrunableLayer):
  """Implements a quantized Dense layer."""

  # Most of these parameters follow the implementation of Dense in
  # Keras, with the exception of kernel_range, bias_range,
  # kernel_quantizer, bias_quantizer, and kernel_initializer.
  #
  # kernel_quantizer: quantizer function/class for kernel
  # bias_quantizer: quantizer function/class for bias
  # kernel_range/bias_ranger: for quantizer functions whose values
  #   can go over [-1,+1], these values are used to set the clipping
  #   value of kernels and biases, respectively, instead of using the
  #   constraints specified by the user.
  #
  # we refer the reader to the documentation of Dense in Keras for the
  # other parameters.

  def __init__(self,
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
               **kwargs):

    if kernel_range is not None:
      warnings.warn("kernel_range is deprecated in QDense layer.")

    if bias_range is not None:
      warnings.warn("bias_range is deprecated in QDense layer.")

    self.kernel_range = kernel_range
    self.bias_range = bias_range

    self.kernel_quantizer = kernel_quantizer
    self.bias_quantizer = bias_quantizer

    self.kernel_quantizer_internal = get_quantizer(self.kernel_quantizer)
    self.bias_quantizer_internal = get_quantizer(self.bias_quantizer)

    # optimize parameter set to "auto" scaling mode if possible
    if hasattr(self.kernel_quantizer_internal, "_set_trainable_parameter"):
      self.kernel_quantizer_internal._set_trainable_parameter()

    self.quantizers = [
        self.kernel_quantizer_internal, self.bias_quantizer_internal
    ]

    kernel_constraint, kernel_initializer = (
        get_auto_range_constraint_initializer(self.kernel_quantizer_internal,
                                              kernel_constraint,
                                              kernel_initializer))

    if use_bias:
      bias_constraint, bias_initializer = (
          get_auto_range_constraint_initializer(self.bias_quantizer_internal,
                                                bias_constraint,
                                                bias_initializer))
    if activation is not None:
      activation = get_quantizer(activation)

    super(QDense, self).__init__(
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
        **kwargs)

  def call(self, inputs):
    if self.kernel_quantizer:
      quantized_kernel = self.kernel_quantizer_internal(self.kernel)
    else:
      quantized_kernel = self.kernel
    output = tf.keras.backend.dot(inputs, quantized_kernel)
    if self.use_bias:
      if self.bias_quantizer:
        quantized_bias = self.bias_quantizer_internal(self.bias)
      else:
        quantized_bias = self.bias
      output = tf.keras.backend.bias_add(output, quantized_bias,
                                         data_format="channels_last")
    if self.activation is not None:
      output = self.activation(output)
    return output

  def compute_output_shape(self, input_shape):
    assert input_shape and len(input_shape) >= 2
    assert input_shape[-1]
    output_shape = list(input_shape)
    output_shape[-1] = self.units
    return tuple(output_shape)

  def get_config(self):
    config = {
        "units": self.units,
        "activation": activations.serialize(self.activation),
        "use_bias": self.use_bias,
        "kernel_quantizer":
            constraints.serialize(self.kernel_quantizer_internal),
        "bias_quantizer":
            constraints.serialize(self.bias_quantizer_internal),
        "kernel_initializer":
            initializers.serialize(self.kernel_initializer),
        "bias_initializer":
            initializers.serialize(self.bias_initializer),
        "kernel_regularizer":
            regularizers.serialize(self.kernel_regularizer),
        "bias_regularizer":
            regularizers.serialize(self.bias_regularizer),
        "activity_regularizer":
            regularizers.serialize(self.activity_regularizer),
        "kernel_constraint":
            constraints.serialize(self.kernel_constraint),
        "bias_constraint":
            constraints.serialize(self.bias_constraint),
        "kernel_range": self.kernel_range,
        "bias_range": self.bias_range
    }
    base_config = super(QDense, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_quantization_config(self):
    return {
        "kernel_quantizer":
            str(self.kernel_quantizer_internal),
        "bias_quantizer":
            str(self.bias_quantizer_internal),
        "activation":
            str(self.activation),
        "units" : str(self.units)
    }

  def get_quantizers(self):
    return self.quantizers

  def get_prunable_weights(self):
    return [self.kernel]


def get_constraint(identifier, quantizer):
  """Gets the initializer.

  Args:
    identifier: A constraint, which could be dict, string, or callable function.
    quantizer: A quantizer class or quantization function

  Returns:
    A constraint class
  """
  if identifier:
    if isinstance(identifier, dict) and identifier['class_name'] == 'Clip':
      return Clip.from_config(identifier['config'])
    else:
      return constraints.get(identifier)
  else:
    max_value = max(1, quantizer.max()) if hasattr(quantizer, "max") else 1.0
    return Clip(-max_value, max_value, identifier, quantizer)

def get_initializer(identifier):
  """Gets the initializer.

  Args:
    identifier: An initializer, which could be dict, string, or callable function.

  Returns:
    A initializer class

  Raises:
    ValueError: An error occurred when quantizer cannot be interpreted.
  """
  if identifier is None:
    return None
  if isinstance(identifier, dict):
    if identifier['class_name'] == 'QInitializer':
      return QInitializer.from_config(identifier['config'])
    else:
      return initializers.get(identifier)
  elif isinstance(identifier, six.string_types):
    return initializers.get(identifier)
  elif callable(identifier):
    return identifier
  else:
    raise ValueError("Could not interpret initializer identifier: " +
                     str(identifier))
