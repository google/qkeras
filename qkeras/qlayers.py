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

import copy
import json
import re
import six

from keras import activations
from keras import constraints
from keras import initializers
from keras import regularizers
import keras.backend as K
from keras.constraints import Constraint
from keras.layers import Activation
from keras.layers import AveragePooling2D
from keras.layers import Conv1D
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import InputSpec
from keras.layers import Layer
from keras.models import model_from_json
from keras.utils import conv_utils
import numpy as np
import tensorflow as tf

#
# Library of auxiliary functions
#


def smooth_sigmoid(x):
  """Implements a linear approximation of a sigmoid function."""

  # if we use 2.65 as the clipping point, MSE w.r.t. original sigmoid is
  # smaller than hard_simoid but the arithmetic for it is (x >> 3) +
  # (x >> 4) + 0.5, which is also not bad.

  return K.clip(0.1875 * x + 0.5, 0.0, 1.0)


def hard_sigmoid(x):
  """Computes hard_sigmoid function that saturates between 0 and 1."""

  return K.clip(0.5 * x + 0.5, 0.0, 1.0)


def binary_sigmoid(x):
  """Computes binary_sigmoid."""

  return _round_through(hard_sigmoid(x))


# we use a version of approximated sigmoid everywhere in this code.
# we can set it to hard_sigmoid(x) or smooth_sigmoid(x).

_sigmoid = hard_sigmoid


def set_internal_sigmoid(mode):
  """Sets _sigmoid to either real, hard or smooth."""

  global _sigmoid

  if mode not in ["real", "hard", "smooth"]:
    raise ValueError("mode has to be 'hard' or 'smooth'.")

  if mode == "hard":
    _sigmoid = hard_sigmoid
  elif mode == "smooth":
    _sigmoid = smooth_sigmoid
  elif mode == "real":
    _sigmoid = K.sigmoid


def binary_tanh(x):
  """Computes binary_tanh function that outputs -1 and 1."""

  return 2.0 * binary_sigmoid(x) - 1.0


def hard_tanh(x):
  """Computes hard_tanh function that saturates between -1 and 1."""

  return 2.0 * hard_sigmoid(x) - 1.0


def smooth_tanh(x):
  """Computes smooth_tanh function that saturates between -1 and 1."""

  return 2.0 * smooth_sigmoid(x) - 1.0


def _round_through(x):
  """Rounds x but using straight through estimator.

  We use the trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182).

  Straight through estimator is a biased estimator for the rounding
  operation defined by Hinton"s Coursera Lecture 9c where dL/dx is made
  equal to dL/dy for y = f(x) during gradient computation, where f(x) is
  a non-derivable function. In that case, we assume df/dx = 1 in:

  dL   dL df   dL
  -- = -- -- = --
  dx   df dx   dy

  (https://www.youtube.com/watch?v=LN0xtUuJsEI&list=PLoRl3Ht4JOcdU872GhiYWf6jwrk_SNhz9&index=41)

  Arguments:
    x: tensor to perform round operation with straight through gradient.

  Returns:
    Rounded tensor.
  """

  return x + K.stop_gradient(-x + K.round(x))


def _sign_through(x):
  """Computes the sign operation using the straight through estimator."""

  # K.sign generates -1, 0 or +1, so it should not be used when we attempt
  # to generate -1 and +1.

  k_sign = K.sign(x)

  return x + K.stop_gradient(-x + k_sign)


def _ceil_through(x):
  """Computes the ceiling operation using straight through estimator."""

  return x + K.stop_gradient(-x + tf.ceil(x))


#
# Activation functions for quantized networks.
#
# Please note some of these functions can be used as well
# as quantizer functions for weights of dense and convolutional
# layers.
#


class quantized_bits(object):
  """Quantizes the number to a number of bits.

  In general, we want to use a quantization function like:

  a = (pow(2,bits)-1 - 0) / (max(x) - min(x))
  b = - min(x) * a

  in the equation:

  xq = a x + b

  This requires multiplication, which is undesirable. So, we
  enforce weights to be between -1 and 1 (max(x) = 1 and min(x) = -1),
  and separating the sign from the rest of the number as we make this function
  symmetric, thus resulting in the following approximation.

  1) max(x) = +1, min(x) = -1
  2) max(x) = -min(x)

  a = pow(2,bits-1)
  b = 0

  Finally, just remember that to represent the number with sign, the
  largest representation is -pow(2,bits) to pow(2, bits-1)

  Symmetric and keep_negative allow us to generate numbers that are symmetric
  (same number of negative and positive representations), and numbers that
  are positive.

  Attributes:
    bits: number of bits to perform quantization.
    integer: number of bits to the left of the decimal point.
    symmetric: if true, we will have the same number of values for positive
      and negative numbers.
    keep_negative: if true, we do not clip negative numbers.

  Returns:
    Function that computes fixed-point quantization with bits.
  """

  def __init__(self, bits=8, integer=0, symmetric=0, keep_negative=1):
    self.bits = bits
    self.integer = integer
    self.symmetric = symmetric
    self.keep_negative = (keep_negative > 0)

  def __call__(self, x):
    """Computes fixedpoint quantization of x."""

    unsigned_bits = self.bits - self.keep_negative
    m = pow(2, unsigned_bits)
    m_i = pow(2, self.integer)
    p = x * m / m_i
    xq = m_i * K.clip(
        _round_through(p), self.keep_negative * (-m + self.symmetric),
        m - 1) / m
    return x + K.stop_gradient(-x + xq)


class bernoulli(object):
  """Computes a Bernoulli sample with probability sigmoid(x).

  This computation uses ST approximation.

  To do that, we compute sigmoid(x) and a random sample z ~ U[0,1]. As
  p in [0,1] and z in [0,1], p - z in [-1,1]. However, -1 will
  never appear because to get -1 we would need sigmoid(-inf) - z == 1.
  As a result, the range will be in practical terms [0,1].

  The noise introduced by z can be seen as a regularizer to the weights W of
  y = Wx as y = Wx + Wz for some noise z with mean mu(z) and var(z). As a
  result, W**2 var(z) to the variance of y, which has the same effect as a
  regularizer on L2 with lambda = var(z), as presented in Hinton"s Coursera
  Lecture 9c.

  Remember that E[dL/dy] = E[dL/dx] once we add stochastic sampling.

  Attributes:
    alpha: allows one to specify multiplicative factor for number generation.

  Returns:
    Computation of round with stochastic sampling with straight through
    gradient.
  """

  def __init__(self, alpha=1.0):
    self.alpha = alpha

  def __call__(self, x):
    p = _sigmoid(x / self.alpha)
    k_sign = K.sign(p - K.random_uniform(K.shape(p)))
    k_sign += (1.0 - K.abs(k_sign))
    return x + K.stop_gradient(-x + self.alpha * (k_sign + 1.0) / 2.0)


class stochastic_ternary(object):
  """Computes a stochastic activation function returning -alpha, 0 or +alpha.

  Computes straight-through approximation using random sampling to make
  E[dL/dy] = E[dL/dx], and computing the sign function. See explanation above.

  Attributes:
    x: tensor to perform sign opertion with stochastic sampling.
    alpha: ternary is -alpha or +alpha.`
    threshold: (1-threshold) specifies the spread of the +1 and -1 values.

  Returns:
    Computation of sign with stochastic sampling with straight through gradient.
  """

  def __init__(self, alpha=1.0, threshold=0.25):
    self.bits = 2
    self.alpha = alpha
    self.threshold = threshold
    assert threshold != 1.0

  def __call__(self, x):
    # we right now use the following distributions for fm1, f0, fp1
    #
    # fm1 = ((1-T)-p)/(1-T)             for p <= (1-T)
    #  f0 = 2*p/clip(0.5+T,0.5,1.0)     for p <= 0.5
    #       2*(1-p)/clip(0.5+T,0.5,1.0) for p > 0.5
    # fp1 = (p-T)/(1-T)                 for p >= T
    #
    # threshold (1-T) determines the spread of -1 and +1
    # for T < 0.5 we need to fix the distribution of f0
    # to make it bigger when compared to the other
    # distributions.

    p = _sigmoid(x / self.alpha)

    T = self.threshold

    ones = K.ones_like(p)
    zeros = K.zeros_like(p)

    T0 = np.clip(0.5 + T, 0.5, 1.0)

    fm1 = tf.where(p <= (1 - T), ((1 - T) - p) / (1 - T), zeros)
    f0 = tf.where(p <= 0.5, 2 * p, 2 * (1 - p)) / T0
    fp1 = tf.where(p <= T, zeros, (p - T) / (1 - T))

    f_all = fm1 + f0 + fp1

    c_fm1 = fm1 / f_all
    c_f0 = (fm1 + f0) / f_all

    r = K.random_uniform(K.shape(p))

    return x + K.stop_gradient(-x + self.alpha * tf.where(
        r <= c_fm1, -1 * ones, tf.where(r <= c_f0, zeros, ones)))


class ternary(object):
  """Computes an activation function returning -alpha, 0 or +alpha.

  Attributes:
    x: tensor to perform sign opertion with stochastic sampling.
    alpha: ternary is -alpha or +alpha. Threshold is also scaled by alpha.
    threshold: threshold to apply "dropout" or dead band (0 value).

  Returns:
    Computation of sign within the threshold.
  """

  def __init__(self, alpha=1.0, threshold=0.33):
    self.alpha = alpha
    self.bits = 2
    self.threshold = threshold

  def __call__(self, x):
    return x + K.stop_gradient(-x + self.alpha * tf.where(
        K.abs(x) < self.threshold, K.zeros_like(x), K.sign(x)))


class stochastic_binary(object):
  """Computes a stochastic activation function returning -alpha or +alpha.

  Computes straight-through approximation using random sampling to make
  E[dL/dy] = E[dL/dx], and computing the sign function. See explanation above.

  Attributes:
    x: tensor to perform sign opertion with stochastic sampling.
    alpha: binary is -alpha or +alpha.`

  Returns:
    Computation of sign with stochastic sampling with straight through gradient.
  """

  def __init__(self, alpha=1.0):
    self.alpha = alpha
    self.bits = 1

  def __call__(self, x):
    assert self.alpha != 0
    p = _sigmoid(x / self.alpha)
    k_sign = K.sign(p - tf.random_uniform(tf.shape(x)))
    # we should not need this, but if K.sign is not safe if input is
    # exactly 0.0
    k_sign += (1.0 - K.abs(k_sign))
    return x + K.stop_gradient(-x + self.alpha * k_sign)


class binary(object):
  """Computes the sign(x) returning a value between -alpha and alpha.

  Although we cannot guarantee E[dL/dy] = E[dL/dx] if we do not use the
  stochastic sampling, we still use the ST approximation.

  Modified from original binary to match QNN implementation.

  Attributes:
    x: tensor to perform sign_through.
    use_01: if True, return {0,1} instead of {-1,+1}.
    alpha: binary is -alpha or +alpha.

  Returns:
    Computation of sign operation with straight through gradient.
  """

  def __init__(self, use_01=False, alpha=1.0):
    self.use_01 = use_01
    self.bits = 1
    self.alpha = alpha

  def __call__(self, x):
    assert self.alpha != 0

    k_sign = K.sign(x)
    k_sign += (1.0 - K.abs(k_sign))
    if self.use_01:
      k_sign = (k_sign + 1.0) / 2.0
    return x + K.stop_gradient(-x + self.alpha * k_sign)


class quantized_relu(object):
  """Computes a quantized relu to a number of bits.

  Modified from:

  [https://github.com/BertMoons/QuantizedNeuralNetworks-Keras-Tensorflow]

  Assume h(x) = +1 with p = sigmoid(x), -1 otherwise, the expected value of
  h(x) is:

  E[h(x)] = +1 P(p <= sigmoid(x)) - 1 P(p > sigmoid(x))
          = +1 P(p <= sigmoid(x)) - 1 ( 1 - P(p <= sigmoid(x)) )
          = 2 P(p <= sigmoid(x)) - 1
          = 2 sigmoid(x) - 1, if p is sampled from a uniform distribution U[0,1]

  If use_sigmoid is 0, we just keep the positive numbers up to
  2**integer * (1 - 2**(-bits)) instead of normalizing them, which is easier
  to implement in hardware.

  Attributes:
    bits: number of bits to perform quantization.
    integer: number of bits to the left of the decimal point.
    use_sigmoid: if true, we apply sigmoid to input to normalize it.

  Returns:
    Function that performs relu + quantization to bits >= 0.
  """

  def __init__(self, bits=8, integer=0, use_sigmoid=0):
    self.bits = bits
    self.integer = integer
    self.use_sigmoid = use_sigmoid

  def __call__(self, x):
    m = pow(2, self.bits)
    m_i = pow(2, self.integer)

    if self.use_sigmoid:
      p = _sigmoid(x / m_i) * m
      xq = m_i * K.clip(2.0 * (_round_through(p) / m) - 1.0, 0.0, 1.0 - 1.0 / m)
    else:
      p = x * m / m_i
      xq = m_i * K.clip(_round_through(p) / m, 0.0, 1.0 - 1.0 / m)
    return xq


class quantized_ulaw(object):
  """Computes a u-law quantization.

  Attributes:
    bits: number of bits to perform quantization.
    integer: number of bits to the left of the decimal point.
    symmetric: if true, we will have the same number of values for positive
      and negative numbers.
    u: parameter of u-law

  Returns:
    Function that performs ulaw + quantization to bits in the range -1.0 to 1.0.
  """

  def __init__(self, bits=8, integer=0, symmetric=0, u=255.0):
    self.bits = bits
    self.integer = integer
    self.symmetric = symmetric
    self.u = u

  def __call__(self, x):
    non_sign_bits = self.bits - 1
    m = pow(2, non_sign_bits)
    m_i = pow(2, self.integer)
    p = _sigmoid(x / m_i) * m
    rp = 2.0 * (_round_through(p) / m) - 1.0
    u_law_p = K.sign(rp) * K.log(1 + self.u * K.abs(rp)) / K.log(1 + self.u)
    xq = m_i * K.clip(u_law_p, -1.0 + (1.0 * self.symmetric) / m, 1.0 - 1.0 / m)
    return xq


class quantized_tanh(object):
  """Computes a quantized tanh to a number of bits.

  Modified from:

  [https://github.com/BertMoons/QuantizedNeuralNetworks-Keras-Tensorflow]

  Attributes:
    bits: number of bits to perform quantization.
    integer: number of bits to the left of the decimal point.
    symmetric: if true, we will have the same number of values for positive
      and negative numbers.

  Returns:
    Function that performs tanh + quantization to bits in the range -1.0 to 1.0.
  """

  def __init__(self, bits=8, integer=0, symmetric=0):
    self.bits = bits
    self.integer = integer
    self.symmetric = symmetric

  def __call__(self, x):
    non_sign_bits = self.bits - 1
    m = pow(2, non_sign_bits)
    m_i = pow(2, self.integer)
    p = _sigmoid(x / m_i) * m
    xq = m_i * K.clip(2.0 * (_round_through(p) / m) - 1.0, -1.0 +
                      (1.0 * self.symmetric) / m, 1.0 - 1.0 / m)
    return xq


class quantized_po2(object):
  """Quantizes to the closest power of 2."""

  def __init__(self, bits=8, max_value=-1):
    self.bits = bits
    self.max_value = max_value

  def __call__(self, x):
    non_sign_bits = self.bits - 1
    min_value = 2**(-2**(non_sign_bits - 1))  # 0 is >> by one more than "bits"
    max_value = 2**(2**(non_sign_bits - 1) - 1)

    if self.max_value != -1:
      max_value = self.max_value

    x_sign = K.sign(x)
    x_sign += (1.0 - K.abs(x_sign)) * min_value
    x_abs = K.clip(K.abs(x), min_value, max_value)

    log2 = np.log(2.0)

    return x + K.stop_gradient(
        -x + x_sign * K.pow(2.0, K.round(K.log(x_abs) / log2)))


class quantized_relu_po2(object):
  """Quantizes to the closest power of 2."""

  def __init__(self, bits=8, max_value=-1):
    self.bits = bits
    self.max_value = max_value

  def __call__(self, x):

    min_value = 2**(-2**(self.bits - 1))  # 0 is smallest power of two number
    max_value = 2**(2**(self.bits - 1) - 1)

    if self.max_value != -1:
      max_value = self.max_value

    x_clipped = K.clip(x, min_value, max_value)

    log2 = np.log(2.0)

    return x + K.stop_gradient(-x +
                               K.pow(2.0, K.round(K.log(x_clipped) / log2)))


#
# Because it may be hard to get serialization from activation functions,
# we may be replacing their instantiation by QActivation in the future.
#


class QActivation(Layer):
  """Implements quantized activation layers."""

  def __init__(self, activation, **kwargs):

    super(QActivation, self).__init__(**kwargs)

    self.activation = activation

    if not isinstance(activation, six.string_types):
      self.quantizer = activation
      self.__name__ = self.quantizer.__name__
      return

    self.__name__ = activation
    param_list = []

    if "(" in activation:
      # mode is "quantized_bits(8,2)"
      # we treat ("quantized_bits(8,2)") is the same as ("quantized_bits", 8, 2)

      activation_list = activation.split("(")

      param_list = [float(v) for v in re.findall(r"-?\d+", activation_list[1])]
      activation = activation_list[0]

    if activation == "hard_sigmoid":
      self.quantizer = hard_sigmoid
    if activation == "smooth_sigmoid":
      self.quantizer = smooth_sigmoid
    elif activation == "hard_tanh":
      self.quantizer = hard_tanh
    elif activation == "quantized_bits":
      self.quantizer = quantized_bits(*param_list)
    elif activation == "bernoulli":
      self.quantizer = bernoulli(*param_list)
    elif activation == "ternary":
      self.quantizer = ternary(*param_list)
    elif activation == "stochastic_ternary":
      self.quantizer = stochastic_ternary(*param_list)
    elif activation == "binary":
      self.quantizer = binary(*param_list)
    elif activation == "stochastic_binary":
      self.quantizer = stochastic_binary(*param_list)
    elif activation == "quantized_relu":
      self.quantizer = quantized_relu(*param_list)
    elif activation == "quantized_ulaw":
      self.quantizer = quantized_ulaw(*param_list)
    elif activation == "quantized_tanh":
      self.quantizer = quantized_tanh(*param_list)
    elif activation == "quantized_po2":
      self.quantizer = quantized_po2(*param_list)
    elif activation == "quantized_relu_po2":
      self.quantizer = quantized_relu_po2(*param_list)
    else:
      raise ValueError("invalid activation '{}'".format(activation))

  def call(self, inputs):
    return self.quantizer(inputs)

  def get_config(self):
    config = {"activation": self.activation}
    base_config = super(QActivation, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def compute_output_shape(self, input_shape):
    return input_shape


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

  def __init__(self, min_value=0.0, max_value=1.0):
    """Initializes Clip constraint class."""

    self.min_value = min_value
    self.max_value = max_value

  def __call__(self, w):
    """Clips values between min and max values."""
    w = K.clip(w, self.min_value, self.max_value)
    return w

  def get_config(self):
    """Returns configuration of constraint class."""
    return {"min_value": self.min_value, "max_value": self.max_value}


def get_initializer(w_initializer, w_range):
  """Gets the initializer and scales it by the range."""

  if isinstance(w_initializer, six.string_types):

    if w_initializer == "he_normal":
      return initializers.VarianceScaling(
          scale=2 * w_range, mode="fan_in", distribution="normal", seed=None)
    if w_initializer == "he_uniform":
      return initializers.VarianceScaling(
          scale=2 * w_range, mode="fan_in", distribution="uniform", seed=None)
    elif w_initializer == "glorot_normal":
      return initializers.VarianceScaling(
          scale=w_range, mode="fan_avg", distribution="normal", seed=None)
    elif w_initializer == "glorot_uniform":
      return initializers.VarianceScaling(
          scale=w_range, mode="fan_avg", distribution="uniform", seed=None)
    elif w_initializer == "random_uniform":
      return initializers.RandomUniform(-w_range, w_range)

  return w_initializer


#
# Definition of Quantized NN classes. These classes were copied
# from the equivalent layers in Keras, and we modified to apply quantization.
# Similar implementations can be seen in the references.
#


class QDense(Dense):
  """Implements a quantized Dense layer."""

  # most of these parameters follow the implementation of Dense in
  # Keras, # with the exception of kernel_range, bias_range,
  # kernel_quantizer and bias_quantizer, and kernel_initializer.
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
  #

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
               kernel_range=1.0,
               bias_range=1.0,
               **kwargs):

    self.kernel_range = kernel_range
    self.bias_max = bias_range

    kernel_initializer = get_initializer(kernel_initializer, kernel_range)
    if kernel_quantizer:
      kernel_constraint = Clip(-kernel_range, kernel_range)

    if bias_quantizer:
      bias_constraint = Clip(-bias_range, bias_range)

    self.kernel_quantizer = kernel_quantizer
    self.bias_quantizer = bias_quantizer

    # using eval here is far from ideal. once I figure out another
    # solution to the specification of these quantization functions,
    # I will use them.

    if isinstance(self.kernel_quantizer, six.string_types):
      self.kernel_quantizer_internal = eval(self.kernel_quantizer)
    else:
      self.kernel_quantizer_internal = self.kernel_quantizer

    if isinstance(self.bias_quantizer, six.string_types):
      self.bias_quantizer_internal = eval(self.bias_quantizer)
    else:
      self.bias_quantizer_internal = self.bias_quantizer

    self.quantizers = [
        self.kernel_quantizer_internal, self.bias_quantizer_internal
    ]

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
    output = K.dot(inputs, quantized_kernel)
    if self.use_bias:
      if self.bias_quantizer:
        quantized_bias = self.bias_quantizer_internal(self.bias)
      else:
        quantized_bias = self.bias
      output = K.bias_add(output, quantized_bias, data_format="channels_last")
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
        "units":
            self.units,
        "activation":
            activations.serialize(self.activation),
        "use_bias":
            self.use_bias,
        "kernel_quantizer":
            self.kernel_quantizer,
        "bias_quantizer":
            self.bias_quantizer,
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
        "kernel_range":
            self.kernel_range,
        "bias_range":
            self.bias_range
    }
    base_config = super(QDense, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_quantizers(self):
    return self.quantizers


class QConv1D(Conv1D):
  """1D convolution layer (e.g. spatial convolution over images)."""

  # most of these parameters follow the implementation of Conv1D in Keras,
  # with the exception of kernel_range, bias_range, kernel_quantizer
  # and bias_quantizer, and kernel_initializer.
  #
  # kernel_quantizer: quantizer function/class for kernel
  # bias_quantizer: quantizer function/class for bias
  # kernel_range/bias_ranger: for quantizer functions whose values
  #   can go over [-1,+1], these values are used to set the clipping
  #   value of kernels and biases, respectively, instead of using the
  #   constraints specified by the user.
  #
  # we refer the reader to the documentation of Conv1D in Keras for the
  # other parameters.
  #

  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding="valid",
               data_format="channels_last",
               dilation_rate=1,
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
               kernel_range=1.0,
               bias_range=1.0,
               **kwargs):

    self.kernel_quantizer = kernel_quantizer
    self.bias_quantizer = bias_quantizer
    self.kernel_range = kernel_range
    self.bias_range = bias_range

    # using eval here is far from ideal. once I figure out another
    # solution to the specification of these quantization functions,
    # I will use them.

    if isinstance(self.kernel_quantizer, six.string_types):
      self.kernel_quantizer_internal = eval(self.kernel_quantizer)
    else:
      self.kernel_quantizer_internal = self.kernel_quantizer

    if isinstance(self.bias_quantizer, six.string_types):
      self.bias_quantizer_internal = eval(self.bias_quantizer)
    else:
      self.bias_quantizer_internal = self.bias_quantizer

    self.quantizers = [
        self.kernel_quantizer_internal, self.bias_quantizer_internal
    ]

    if kernel_quantizer:
      kernel_constraint = Clip(-kernel_range, kernel_range)

    if bias_quantizer:
      bias_constraint = Clip(-bias_range, bias_range)

    super(QConv1D, self).__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
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
      if isinstance(self.kernel_quantizer, QActivation):
        quantized_kernel = self.kernel_quantizer_internal(self.kernel)
      else:
        quantized_kernel = self.kernel_quantizer_internal(self.kernel)
    else:
      quantized_kernel = self.kernel

    outputs = K.conv1d(
        inputs,
        quantized_kernel,
        strides=self.strides,
        padding=self.padding,
        data_format=self.data_format,
        dilation_rate=self.dilation_rate)

    if self.use_bias:
      if self.bias_quantizer:
        if isinstance(self.bias_quantizer, QActivation):
          quantized_bias = self.bias_quantizer_internal(self.bias)
        else:
          quantized_bias = self.bias_quantizer_internal(self.bias)
      else:
        quantized_bias = self.bias

      outputs = K.bias_add(
          outputs, quantized_bias, data_format=self.data_format)

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def get_config(self):
    config = {
        "kernel_quantizer": self.kernel_quantizer,
        "bias_quantizer": self.bias_quantizer,
        "kernel_range": self.kernel_range,
        "bias_range": self.bias_range
    }
    base_config = super(QConv1D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_quantizers(self):
    return self.quantizers


class QConv2D(Conv2D):
  """2D convolution layer (e.g. spatial convolution over images)."""

  # most of these parameters follow the implementation of Conv2D in Keras,
  # with the exception of kernel_range, bias_range, kernel_quantizer
  # and bias_quantizer, and kernel_initializer.
  #
  # kernel_quantizer: quantizer function/class for kernel
  # bias_quantizer: quantizer function/class for bias
  # kernel_range/bias_ranger: for quantizer functions whose values
  #   can go over [-1,+1], these values are used to set the clipping
  #   value of kernels and biases, respectively, instead of using the
  #   constraints specified by the user.
  #
  # we refer the reader to the documentation of Conv2D in Keras for the
  # other parameters.
  #

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding="valid",
               data_format="channels_last",
               dilation_rate=(1, 1),
               activation=None,
               use_bias=True,
               kernel_initializer="he_normal",
               bias_initializer="zeros",
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               kernel_range=1.0,
               bias_range=1.0,
               kernel_quantizer=None,
               bias_quantizer=None,
               **kwargs):

    self.kernel_quantizer = kernel_quantizer
    self.bias_quantizer = bias_quantizer
    self.kernel_range = kernel_range
    self.bias_range = bias_range

    kernel_initializer = get_initializer(kernel_initializer, kernel_range)

    # using eval here is far from ideal. once I figure out another
    # solution to the specification of these quantization functions,
    # I will use them.

    if isinstance(self.kernel_quantizer, six.string_types):
      self.kernel_quantizer_internal = eval(self.kernel_quantizer)
    else:
      self.kernel_quantizer_internal = self.kernel_quantizer

    if isinstance(self.bias_quantizer, six.string_types):
      self.bias_quantizer_internal = eval(self.bias_quantizer)
    else:
      self.bias_quantizer_internal = self.bias_quantizer

    self.quantizers = [
        self.kernel_quantizer_internal, self.bias_quantizer_internal
    ]

    if kernel_quantizer:
      kernel_constraint = Clip(-kernel_range, kernel_range)

    if bias_quantizer:
      bias_constraint = Clip(-bias_range, bias_range)

    super(QConv2D, self).__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
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
      if isinstance(self.kernel_quantizer, QActivation):
        quantized_kernel = self.kernel_quantizer_internal(self.kernel)
      else:
        quantized_kernel = self.kernel_quantizer_internal(self.kernel)
    else:
      quantized_kernel = self.kernel

    outputs = K.conv2d(
        inputs,
        quantized_kernel,
        strides=self.strides,
        padding=self.padding,
        data_format=self.data_format,
        dilation_rate=self.dilation_rate)

    if self.use_bias:
      if self.bias_quantizer:
        if isinstance(self.bias_quantizer, QActivation):
          quantized_bias = self.bias_quantizer_internal(self.bias)
        else:
          quantized_bias = self.bias_quantizer_internal(self.bias)
      else:
        quantized_bias = self.bias

      outputs = K.bias_add(
          outputs, quantized_bias, data_format=self.data_format)

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def get_config(self):
    config = {
        "kernel_quantizer": self.kernel_quantizer,
        "bias_quantizer": self.bias_quantizer,
        "kernel_range": self.kernel_range,
        "bias_range": self.bias_range
    }
    base_config = super(QConv2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_quantizers(self):
    return self.quantizers


class QDepthwiseConv2D(Conv2D):
  """Creates quantized depthwise conv2d. Copied from mobilenet."""

  # most of these parameters follow the implementation of DepthwiseConv2D
  # in Keras, # with the exception of depthwise_range, bias_range,
  # depthwise_quantizer # and bias_quantizer, and kernel_initializer.
  #
  # depthwise_quantizer: quantizer function/class for kernel
  # bias_quantizer: quantizer function/class for bias
  # depthwise_range/bias_ranger: for quantizer functions whose values
  #   can go over [-1,+1], these values are used to set the clipping
  #   value of kernels and biases, respectively, instead of using the
  #   constraints specified by the user.
  #
  # we refer the reader to the documentation of DepthwiseConv2D in Keras for the
  # other parameters.
  #

  def __init__(self,
               kernel_size,
               strides=(1, 1),
               padding="valid",
               depth_multiplier=1,
               data_format=None,
               activation=None,
               use_bias=True,
               depthwise_initializer="he_normal",
               bias_initializer="zeros",
               depthwise_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               depthwise_constraint=None,
               bias_constraint=None,
               dilation_rate=(1, 1),
               depthwise_quantizer=None,
               bias_quantizer=None,
               depthwise_range=1.0,
               bias_range=1.0,
               **kwargs):

    if depthwise_quantizer:
      depthwise_constraint = Clip(-depthwise_range, depthwise_range)

    if use_bias and bias_quantizer:
      bias_constraint = Clip(-bias_range, bias_range)

    super(QDepthwiseConv2D, self).__init__(
        filters=None,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        activation=activation,
        use_bias=use_bias,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        bias_constraint=bias_constraint,
        dilation_rate=dilation_rate,
        **kwargs)
    self.depth_multiplier = depth_multiplier
    self.depthwise_initializer = initializers.get(depthwise_initializer)
    self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
    self.depthwise_constraint = constraints.get(depthwise_constraint)
    self.bias_initializer = initializers.get(bias_initializer)

    self.depthwise_constraint = depthwise_constraint
    self.bias_constraint = bias_constraint

    self.depthwise_quantizer = depthwise_quantizer
    self.bias_quantizer = bias_quantizer

    self.depthwise_range = depthwise_range
    self.bias_range = bias_range

    # using eval here is far from ideal. once I figure out another
    # solution to the specification of these quantization functions,
    # I will use them.

    if isinstance(self.depthwise_quantizer, six.string_types):
      self.depthwise_quantizer_internal = eval(self.depthwise_quantizer)
    else:
      self.depthwise_quantizer_internal = self.depthwise_quantizer

    if isinstance(self.bias_quantizer, six.string_types):
      self.bias_quantizer_internal = eval(self.bias_quantizer)
    else:
      self.bias_quantizer_internal = self.bias_quantizer

    self.quantizers = [
        self.depthwise_quantizer_internal, self.bias_quantizer_internal
    ]

  def build(self, input_shape):
    if len(input_shape) < 4:
      raise ValueError(
          "Inputs to `QDepthwiseConv2D` should have rank 4. "
          "Received input shape:", str(input_shape))
    if self.data_format == "channels_first":
      channel_axis = 1
    else:
      channel_axis = 3
    if input_shape[channel_axis] is None:
      raise ValueError("The channel dimension of the inputs to "
                       "`QDepthwiseConv2D` "
                       "should be defined. Found `None`.")
    input_dim = int(input_shape[channel_axis])
    depthwise_kernel_shape = (self.kernel_size[0], self.kernel_size[1],
                              input_dim, self.depth_multiplier)

    self.depthwise_kernel = self.add_weight(
        shape=depthwise_kernel_shape,
        initializer=self.depthwise_initializer,
        name="depthwise_kernel",
        regularizer=self.depthwise_regularizer,
        constraint=self.depthwise_constraint)

    if self.use_bias:
      self.bias = self.add_weight(
          shape=(input_dim * self.depth_multiplier,),
          initializer=self.bias_initializer,
          name="bias",
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint)
    else:
      self.bias = None
    # Set input spec.
    self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
    self.built = True

  def call(self, inputs, training=None):
    if self.depthwise_quantizer:
      if isinstance(self.depthwise_quantizer, QActivation):
        quantized_depthwise_kernel = (
            self.depthwise_quantizer_internal(self.depthwise_kernel))
      else:
        quantized_depthwise_kernel = (
            self.depthwise_quantizer_internal(self.depthwise_kernel))
    else:
      quantized_depthwise_kernel = self.depthwise_kernel
    outputs = K.depthwise_conv2d(
        inputs,
        quantized_depthwise_kernel,
        strides=self.strides,
        padding=self.padding,
        dilation_rate=self.dilation_rate,
        data_format=self.data_format)

    if self.bias:
      if self.bias_quantizer:
        if isinstance(self.bias_quantizer, QActivation):
          quantized_bias = self.bias_quantizer_internal(self.bias)
        else:
          quantized_bias = self.bias_quantizer_internal(self.bias)
      else:
        quantized_bias = self.bias
      outputs = K.bias_add(
          outputs, quantized_bias, data_format=self.data_format)

    if self.activation is not None:
      if isinstance(self.activation, QActivation):
        return self.activation(outputs)
      else:
        return self.activation(outputs)

    return outputs

  def compute_output_shape(self, input_shape):
    if self.data_format == "channels_first":
      rows = input_shape[2]
      cols = input_shape[3]
      out_filters = input_shape[1] * self.depth_multiplier
    elif self.data_format == "channels_last":
      rows = input_shape[1]
      cols = input_shape[2]
      out_filters = input_shape[3] * self.depth_multiplier

    rows = conv_utils.conv_output_length(rows, self.kernel_size[0],
                                         self.padding, self.strides[0])

    cols = conv_utils.conv_output_length(cols, self.kernel_size[1],
                                         self.padding, self.strides[1])

    if self.data_format == "channels_first":
      return (input_shape[0], out_filters, rows, cols)
    elif self.data_format == "channels_last":
      return (input_shape[0], rows, cols, out_filters)

  def get_config(self):
    config = super(QDepthwiseConv2D, self).get_config()
    config.pop("filters")
    config.pop("kernel_initializer")
    config.pop("kernel_regularizer")
    config.pop("kernel_constraint")
    config["depth_multiplier"] = self.depth_multiplier
    config["depthwise_initializer"] = initializers.serialize(
        self.depthwise_initializer)
    config["depthwise_regularizer"] = regularizers.serialize(
        self.depthwise_regularizer)
    config["depthwise_constraint"] = constraints.serialize(
        self.depthwise_constraint)
    config["depthwise_quantizer"] = constraints.serialize(
        self.depthwise_quantizer)
    config["bias_quantizer"] = constraints.serialize(self.bias_quantizer)
    config["depthwise_range"] = self.depthwise_range
    config["bias_range"] = self.bias_range
    return config

  def get_quantizers(self):
    return self.quantizers


def QSeparableConv2D(filters,
                     kernel_size,
                     strides=(1, 1),
                     padding="valid",
                     dilation_rate=(1, 1),
                     depth_multiplier=1,
                     activation=None,
                     use_bias=True,
                     depthwise_initializer="he_normal",
                     pointwise_initializer="he_normal",
                     bias_initializer="zeros",
                     depthwise_regularizer=None,
                     pointwise_regularizer=None,
                     bias_regularizer=None,
                     activity_regularizer=None,
                     depthwise_constraint=None,
                     pointwise_constraint=None,
                     bias_constraint=None,
                     depthwise_quantizer=None,
                     pointwise_quantizer=None,
                     bias_quantizer=None,
                     depthwise_activation=None,
                     depthwise_range=1.0,
                     pointwise_range=1.0,
                     bias_range=1.0,
                     depthwise_dropout_rate=0.0,
                     pw_first=False,
                     name=""):
  """Adds a quantized separableconv2d."""

  # we use here a modified version that appeared in mobilenet that adds
  # quantization to the network, and possibly an intermediate activation
  # layer that acts as a quantizer and possible dropout layer between
  # the depthwise and pointwise convolutions.
  #
  # since this implementation expands into depthwise -> pointwise
  # convolutions, the users will not see a separable convolution operation
  # in model.summary(), but rather a depthwise convolution followed by a
  # pointwise convolution.
  #
  # depthwise_quantizer: depthwise quantization function
  # pointwise_quantizer: pointwise quantization function
  # bias_quantizer: bias quantization function for the pointwise convolution
  # depthwise_range/pointwise_range/bias_range: ranges to be used if
  # quantization values can become greater than -1 and +1.
  # depthwise_dropout_rate: dropout between depthwise and pointwise is added
  #   if rate > 0.0
  # pw_first: this may disappear in the future, but as deep quantized networks
  #   sometimes behave in different ways, if we are using binary or ternary
  #   quantization, it may be better to apply pointwise before depthwise.
  #
  # For the remaining parameters, please refer to Keras implementation of
  # SeparableConv2D.
  #

  def _call(inputs):
    """Internally builds qseparableconv2d."""

    x = inputs

    if pw_first:
      x = QConv2D(
          filters, (1, 1),
          strides=(1, 1),
          padding="same",
          use_bias=use_bias,
          kernel_constraint=pointwise_constraint,
          kernel_initializer=pointwise_initializer,
          kernel_regularizer=pointwise_regularizer,
          kernel_quantizer=pointwise_quantizer,
          bias_quantizer=bias_quantizer,
          bias_regularizer=bias_regularizer,
          bias_initializer=bias_initializer,
          bias_constraint=bias_constraint,
          activity_regularizer=activity_regularizer,
          kernel_range=pointwise_range,
          bias_range=bias_range,
          name=name + "_pw")(
              x)

      if depthwise_activation:
        if isinstance(depthwise_activation, QActivation):
          x = depthwise_activation(x)
        else:
          x = QActivation(depthwise_activation, name=name + "_dw_act")(x)

      if depthwise_dropout_rate > 0.0:
        x = Dropout(rate=depthwise_dropout_rate, name=name + "_dw_dropout")(x)

    x = QDepthwiseConv2D(
        kernel_size,
        strides=strides,
        dilation_rate=dilation_rate,
        padding=padding,
        depth_multiplier=depth_multiplier,
        use_bias=False,
        depthwise_regularizer=depthwise_regularizer,
        depthwise_initializer=depthwise_initializer,
        depthwise_constraint=depthwise_constraint,
        depthwise_quantizer=depthwise_quantizer,
        depthwise_range=depthwise_range,
        name=name + "_dw")(
            x)

    if not pw_first:
      if depthwise_activation:
        if isinstance(depthwise_activation, QActivation):
          x = depthwise_activation(x)
        else:
          x = QActivation(depthwise_activation, name=name + "_dw_act")(x)

      if depthwise_dropout_rate > 0.0:
        x = Dropout(rate=depthwise_dropout_rate, name=name + "_dw_dropout")(x)

      x = QConv2D(
          filters, (1, 1),
          strides=(1, 1),
          padding="same",
          use_bias=use_bias,
          kernel_constraint=pointwise_constraint,
          kernel_initializer=pointwise_initializer,
          kernel_regularizer=pointwise_regularizer,
          kernel_quantizer=pointwise_quantizer,
          bias_quantizer=bias_quantizer,
          bias_regularizer=bias_regularizer,
          bias_initializer=bias_initializer,
          bias_constraint=bias_constraint,
          activity_regularizer=activity_regularizer,
          kernel_range=pointwise_range,
          bias_range=bias_range,
          name=name + "_pw")(
              x)

    if activation:
      if isinstance(activation, QActivation):
        x = activation(x)
      else:
        x = Activation(activation, name=name + "_pw_act")(x)
    return x

  return _call


def QAveragePooling2D(
    pool_size=(2, 2), strides=None, padding="valid", quantizer=None, **kwargs):
  """Computes the quantized version of AveragePooling2D."""

  # this is just a convenient layer, not being actually anything fancy. Just
  # reminds us that we need to quantize average pooling before the next layer.

  def _call(x):
    """Performs inline call to AveragePooling followed by QActivation."""

    x = AveragePooling2D(pool_size, strides, padding, **kwargs)(x)

    if kwargs.get("name", None):
      name = kwargs["name"] + "_act"
    else:
      name = None

    if quantizer:
      x = QActivation(quantizer, name=name)(x)

    return x

  return _call


#
# Model utilities: before saving the weights, we want to apply the quantizers
#


def model_save_quantized_weights(model, filename=None):
  """Quantizes model for inference and save it.

  Takes a model with weights, apply quantization function to weights and
  returns a dictionarty with quantized weights.

  User should be aware that "po2" quantization functions cannot really
  be quantized in meaningful way in Keras. So, in order to preserve
  compatibility with inference flow in Keras, we do not covert "po2"
  weights and biases to exponents + signs (in case of quantize_po2), but
  return instead (-1)**sign*(2**round(log2(x))). In the returned dictionary,
  we will return the pair (sign, round(log2(x))).

  Arguments:
    model: model with weights to be quantized.
    filename: if specified, we will save the hdf5 containing the quantized
      weights so that we can use them for inference later on.

  Returns:
    dictionary containing layer name and quantized weights that can be used
    by a hardware generator.

  """

  saved_weights = {}

  print("... quantizing model")
  for layer in model.layers:
    if hasattr(layer, "get_quantizers"):
      weights = []
      signs = []
      for quantizer, weight in zip(layer.get_quantizers(), layer.get_weights()):
        if quantizer:
          weight = K.constant(weight)
          weight = K.eval(quantizer(weight))

        # If quantizer is power-of-2 (quantized_po2 or quantized_relu_po2),
        # we would like to process it here.
        #
        # However, we cannot, because we will loose sign information as
        # quanized_po2 will be represented by the tuple (sign, log2(abs(w))).
        #
        # In addition, we will not be able to use the weights on the model
        # any longer.
        #
        # So, instead of "saving" the weights in the model, we will return
        # a dictionary so that the proper values can be propagated.

        weights.append(weight)

        has_sign = False
        if quantizer:
          if isinstance(quantizer, six.string_types):
            q_name = quantizer
          if hasattr(quantizer, "__name__"):
            q_name = quantizer.__name__
          elif hasattr(quantizer, "name"):
            q_name = quantizer.name
          elif hasattr(quantizer, "__class__"):
            q_name = quantizer.__class__.__name__
          else:
            q_name = ""
        if quantizer and ("_po2" in q_name):
          # quantized_relu_po2 does not have a sign
          if isinstance(quantizer, quantized_po2):
            has_sign = True
          sign = np.sign(weight)
          # make sure values are -1 or +1 only
          sign += (1.0 - np.abs(sign))
          weight = np.round(np.log2(np.abs(weight)))
          signs.append(sign)
        else:
          signs.append([])

      saved_weights[layer.name] = {"weights": weights}
      if has_sign:
        saved_weights[layer.name]["signs"] = signs

      layer.set_weights(weights)
    else:
      if layer.get_weights():
        print(" ", layer.name, "has not been quantized")

  if filename:
    model.save_weights(filename)

  return saved_weights


def quantize_activation(layer_config, custom_objects, activation_bits):
  """Replaces activation by quantized activation functions."""

  str_act_bits = str(activation_bits)

  # relu -> quantized_relu(bits)
  # tanh -> quantized_tanh(bits)
  #
  # more to come later

  if layer_config["activation"] == "relu":
    layer_config["activation"] = "quantized_relu(" + str_act_bits + ")"
    custom_objects["quantized_relu(" + str_act_bits + ")"] = (
        quantized_relu(activation_bits))

  elif layer_config["activation"] == "tanh":
    layer_config["activation"] = "quantized_tanh(" + str_act_bits + ")"
    custom_objects["quantized_tanh(" + str_act_bits + ")"] = (
        quantized_tanh(activation_bits))


def model_quantize(model,
                   quantizer_config,
                   activation_bits,
                   custom_objects=None,
                   transfer_weights=False):
  """Creates a quantized model from non-quantized model.

  The quantized model translation is based on json interface of Keras,
  which requires a custom_objects dictionary for "string" types.

  Because of the way json works, we pass "string" objects for the
  quantization mechanisms and we perform an eval("string") which
  technically is not safe, but it will do the job.

  The quantizer_config is a dictionary with the following form.
  {
    Dense_layer_name: {
        "kernel_quantizer": "quantizer string",
        "bias_quantizer": "quantizer_string"
    },

    Conv2D_layer_name: {
        "kernel_quantizer": "quantizer string",
        "bias_quantizer": "quantizer_string"
    },

    Activation_layer_name: "quantizer string",

    "QActivation": { "relu": "quantizer_string" },

    "QConv2D": {
        "kernel_quantizer": "quantizer string",
        "bias_quantizer": "quantizer_string"
    }
  }

  In the case of "QActivation", we can modify only certain types of
  activations, for example, a "relu". In this case we represent the
  activation name by a dictionary, or we can modify all activations,
  without representhing as a set.

  We right now require a default case in case we cannot find layer name.
  This simplifies the dictionary because the simplest case, we can just
  say:

  {
    "default": {
        "kernel": "quantized_bits(4)",
        "bias": "quantized_bits(4)"
    }
  }

  and this will quantize all layers' weights and bias to be created with
  4 bits.

  Arguments:
    model: model to be quantized
    quantizer_config: dictionary (as above) with quantized parameters
    activation_bits: number of bits for quantized_relu, quantized_tanh
    custom_objects: dictionary following keras recommendations for json
      translation.
    transfer_weights: if true, weights are to be transfered from model to
      qmodel.

  Returns:
    qmodel with quantized operations and custom_objects.
  """

  if not custom_objects:
    custom_objects = {}

  # let's make a deep copy to make sure our objects are not shared elsewhere
  jm = copy.deepcopy(json.loads(model.to_json()))
  custom_objects = copy.deepcopy(custom_objects)

  config = jm["config"]

  layers = config["layers"]

  custom_objects["QDense"] = QDense
  custom_objects["QConv1D"] = QConv1D
  custom_objects["QConv2D"] = QConv2D
  custom_objects["QDepthwiseConv2D"] = QDepthwiseConv2D
  custom_objects["QAveragePooling2D"] = QAveragePooling2D
  custom_objects["QActivation"] = QActivation

  # just add all the objects from quantizer_config to
  # custom_objects first.

  for layer_name in quantizer_config.keys():

    if isinstance(quantizer_config[layer_name], six.string_types):
      name = quantizer_config[layer_name]
      custom_objects[name] = eval(name)

    else:
      for name in quantizer_config[layer_name].keys():
        custom_objects[quantizer_config[layer_name][name]] = (
            eval(quantizer_config[layer_name][name]))

  for layer in layers:
    layer_config = layer["config"]

    # Dense becomes QDense
    # Activation converts activation functions

    if layer["class_name"] == "Dense":
      layer["class_name"] = "QDense"

      # needs to add kernel/bias quantizers

      kernel_quantizer = quantizer_config.get(
          layer["name"], quantizer_config.get("QDense",
                                              None))["kernel_quantizer"]

      bias_quantizer = quantizer_config.get(
          layer["name"], quantizer_config.get("QDense", None))["bias_quantizer"]

      layer_config["kernel_quantizer"] = kernel_quantizer
      layer_config["bias_quantizer"] = bias_quantizer

      # if activation is present, add activation here

      quantizer = quantizer_config.get(layer["name"],
                                       quantizer_config.get(
                                           "QDense", None)).get(
                                               "activation_quantizer", None)

      if quantizer:
        layer_config["activation"] = quantizer
        custom_objects[quantizer] = eval(quantizer)
      else:
        quantize_activation(layer_config, custom_objects, activation_bits)

    elif layer["class_name"] == "Conv2D":
      layer["class_name"] = "QConv2D"

      # needs to add kernel/bias quantizers

      kernel_quantizer = quantizer_config.get(
          layer["name"], quantizer_config.get("QConv2D", None)).get(
              "kernel_quantizer", None)

      bias_quantizer = quantizer_config.get(
          layer["name"], quantizer_config.get("QConv2D", None)).get(
              "bias_quantizer", None)

      layer_config["kernel_quantizer"] = kernel_quantizer
      layer_config["bias_quantizer"] = bias_quantizer

      # if activation is present, add activation here

      quantizer = quantizer_config.get(layer["name"],
                                       quantizer_config.get(
                                           "QConv2D", None)).get(
                                               "activation_quantizer", None)

      if quantizer:
        layer_config["activation"] = quantizer
        custom_objects[quantizer] = eval(quantizer)
      else:
        quantize_activation(layer_config, custom_objects, activation_bits)

    elif layer["class_name"] == "DepthwiseConv2D":
      layer["class_name"] = "QDepthwiseConv2D"

      # needs to add kernel/bias quantizers

      depthwise_quantizer = quantizer_config.get(
          layer["name"], quantizer_config.get("QDepthwiseConv2D", None)).get(
              "depthwise_quantizer", None)

      bias_quantizer = quantizer_config.get(
          layer["name"], quantizer_config.get("QDepthwiseConv2D", None)).get(
              "bias_quantizer", None)

      layer_config["depthwise_quantizer"] = depthwise_quantizer
      layer_config["bias_quantizer"] = bias_quantizer

      # if activation is present, add activation here

      quantizer = quantizer_config.get(
          layer["name"], quantizer_config.get("QDepthwiseConv2D", None)).get(
              "activation_quantizer", None)

      if quantizer:
        layer_config["activation"] = quantizer
        custom_objects[quantizer] = eval(quantizer)
      else:
        quantize_activation(layer_config, custom_objects, activation_bits)

    elif layer["class_name"] == "Activation":
      quantizer = quantizer_config.get(
          layer["name"], quantizer_config.get("QActivation", None))

      # if quantizer exists in dictionary related to this name,
      # use it, otherwise, use normal transformations

      if not isinstance(quantizer, dict) or quantizer.get(
          layer_config["activation"], None):
        # only change activation layer if we will use a quantized activation

        layer["class_name"] = "QActivation"
        if isinstance(quantizer, dict):
          quantizer = quantizer[layer_config["activation"]]
        if quantizer:
          layer_config["activation"] = quantizer
          custom_objects[quantizer] = eval(quantizer)
        else:
          quantize_activation(layer_config, custom_objects, activation_bits)

    elif layer["class_name"] == "AveragePooling2D":
      layer["class_name"] = "QAveragePooling2D"

      quantizer = quantizer_config.get(layer["name"], None)

      # if quantizer exists in dictionary related to this name,
      # use it, otherwise, use normal transformations

      if quantizer:
        layer_config["activation"] = quantizer
        custom_objects[quantizer] = eval(quantizer)
      else:
        quantize_activation(layer_config, custom_objects, activation_bits)

  # we need to keep a dictionary of custom objects as our quantized library
  # is not recognized by keras.

  qmodel = model_from_json(json.dumps(jm), custom_objects=custom_objects)

  # if transfer_weights is true, we load the weights from model to qmodel

  if transfer_weights:
    for layer, qlayer in zip(model.layers, qmodel.layers):
      if layer.get_weights():
        qlayer.set_weights(copy.deepcopy(layer.get_weights()))

  return qmodel, custom_objects
