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
import numpy as np
import warnings
import six
import tensorflow.compat.v2 as tf
from tensorflow.keras import activations
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Layer
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
