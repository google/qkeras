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
# ==============================================================================

import tensorflow.compat.v2 as tf
import numpy as np
import six
import warnings

from tensorflow.keras import initializers
from tensorflow.keras.utils import deserialize_keras_object

from .safe_eval import safe_eval

#
# Library of auxiliary functions
#

def smooth_sigmoid(x):
  """Implements a linear approximation of a sigmoid function."""

  # if we use 2.65 as the clipping point, MSE w.r.t. original sigmoid is
  # smaller than hard_simoid but the arithmetic for it is (x >> 3) +
  # (x >> 4) + 0.5, which is also not bad.

  return tf.keras.backend.clip(0.1875 * x + 0.5, 0.0, 1.0)


def hard_sigmoid(x):
  """Computes hard_sigmoid function that saturates between 0 and 1."""

  return tf.keras.backend.clip(0.5 * x + 0.5, 0.0, 1.0)


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
    _sigmoid = tf.sigmoid


def binary_tanh(x):
  """Computes binary_tanh function that outputs -1 and 1."""
  return 2.0 * binary_sigmoid(x) - 1.0


def hard_tanh(x):
  """Computes hard_tanh function that saturates between -1 and 1."""
  return 2.0 * hard_sigmoid(x) - 1.0


def smooth_tanh(x):
  """Computes smooth_tanh function that saturates between -1 and 1."""
  return 2.0 * smooth_sigmoid(x) - 1.0


def stochastic_round(x):
  """Performs stochastic rounding to the first decimal point."""
  s = tf.sign(x)
  s += (1.0 - tf.abs(s)) * (2.0 * tf.round(tf.random.uniform(tf.shape(x))) -
                            1.0)
  t = tf.floor(x) - (s - 1.0) / 2.0
  p = tf.abs(x - t)
  f = s * (tf.sign(p - tf.random.uniform(tf.shape(p))) + 1.0) / 2.0
  return t + f


def stochastic_round_po2(x):
  """Performs stochastic rounding for the power of two."""
  # TODO(hzhuang): test stochastic_round_po2 and constraint.
  # because quantizer is applied after constraint.
  y = tf.abs(x)
  eps = tf.keras.backend.epsilon()
  log2 = tf.keras.backend.log(2.0)
  x_log2 = tf.round(tf.keras.backend.log(y + eps) / log2)
  sign = tf.sign(x)
  po2 = tf.cast(pow(2.0, tf.cast(x_log2, dtype="float32")), dtype="float32")
  left_val = tf.where(po2 > y, x_log2 - 1, x_log2)
  right_val = tf.where(po2 > y, x_log2, x_log2 + 1)
  # sampling in [2**left_val, 2**right_val].
  minval = 2 ** left_val
  maxval = 2 ** right_val
  val = tf.random.uniform(tf.shape(y), minval=minval, maxval=maxval)
  # use y as a threshold to keep the probabliy [2**left_val, y, 2**right_val]
  # so that the mean value of the sample should be y
  x_po2 = tf.where(y < val, left_val, right_val)
  return x_po2


def _round_through(x, use_stochastic_rounding=False):
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
    use_stochastic_rounding: if true, we perform stochastic rounding.

  Returns:
    Rounded tensor.
  """
  if use_stochastic_rounding:
    return x + tf.stop_gradient(-x + stochastic_round(x))
  else:
    return x + tf.stop_gradient(-x + tf.round(x))


def _sign_through(x):
  """Computes the sign operation using the straight through estimator."""

  # tf.sign generates -1, 0 or +1, so it should not be used when we attempt
  # to generate -1 and +1.

  k_sign = tf.sign(x)

  return x + tf.stop_gradient(-x + k_sign)


def _ceil_through(x):
  """Computes the ceiling operation using straight through estimator."""

  return x + tf.stop_gradient(-x + tf.ceil(x))



#
# Activation functions for quantized networks.
#
# Please note some of these functions can be used as well
# as quantizer functions for weights of dense and convolutional
# layers.
#


class quantized_bits(object):  # pylint: disable=invalid-name
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
    use_stochastic_rounding: if true, we perform stochastic rounding.

  Returns:
    Function that computes fixed-point quantization with bits.
  """

  def __init__(self, bits=8, integer=0, symmetric=0, keep_negative=1,
               use_stochastic_rounding=False):
    self.bits = bits
    self.integer = integer
    self.symmetric = symmetric
    self.keep_negative = (keep_negative > 0)
    self.use_stochastic_rounding = use_stochastic_rounding

  def __call__(self, x):
    """Computes fixedpoint quantization of x."""

    unsigned_bits = self.bits - self.keep_negative

    # quantized_bits with "1" bit becomes a binary implementation.

    if unsigned_bits > 0:
      m = pow(2, unsigned_bits)
      m_i = pow(2, self.integer)
      p = x * m / m_i
      xq = m_i * tf.keras.backend.clip(
          _round_through(p, self.use_stochastic_rounding),
          self.keep_negative * (-m + self.symmetric), m - 1) / m
    else:
      xq = tf.sign(x)
      xq += (1.0 - tf.abs(xq))
      if not self.keep_negative:
        xq = (xq + 1.0) / 2.0
    return x + tf.stop_gradient(-x + xq)

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def get_config(self):
    config = {
        "bits":
            self.bits,
        "integer":
            self.integer,
        "symmetric":
            self.symmetric,
        "keep_negative":
            self.keep_negative,
        "use_stochastic_rounding":
            self.use_stochastic_rounding
    }
    return config


class bernoulli(object):  # pylint: disable=invalid-name
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
    k_sign = tf.sign(p - tf.random.uniform(tf.shape(p)))
    k_sign += (1.0 - tf.abs(k_sign))
    return x + tf.stop_gradient(-x + self.alpha * (k_sign + 1.0) / 2.0)

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def get_config(self):
    config = {"alpha": self.alpha}
    return config


class stochastic_ternary(object):  # pylint: disable=invalid-name
  """Computes a stochastic activation function returning -alpha, 0 or +alpha.

  Computes straight-through approximation using random sampling to make
  E[dL/dy] = E[dL/dx], and computing the sign function. See explanation above.

  Attributes:
    x: tensor to perform sign opertion with stochastic sampling.
    bits: number of bits to perform quantization.
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

    p = _sigmoid(x / self.alpha)  # pylint: disable=invalid-name

    T = self.threshold  # pylint: disable=invalid-name

    ones = tf.ones_like(p)
    zeros = tf.zeros_like(p)

    T0 = np.clip(0.5 + T, 0.5, 1.0)  # pylint: disable=invalid-name

    fm1 = tf.where(p <= (1 - T), ((1 - T) - p) / (1 - T), zeros)
    f0 = tf.where(p <= 0.5, 2 * p, 2 * (1 - p)) / T0
    fp1 = tf.where(p <= T, zeros, (p - T) / (1 - T))

    f_all = fm1 + f0 + fp1

    c_fm1 = fm1 / f_all
    c_f0 = (fm1 + f0) / f_all

    r = tf.random.uniform(tf.shape(p))

    return x + tf.stop_gradient(-x + self.alpha * tf.where(
        r <= c_fm1, -1 * ones, tf.where(r <= c_f0, zeros, ones)))

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def get_config(self):
    config = {
        "alpha":
            self.alpha,
        "threshold":
            self.threshold,
    }
    return config


class ternary(object):  # pylint: disable=invalid-name
  """Computes an activation function returning -alpha, 0 or +alpha.

  Attributes:
    x: tensor to perform sign opertion with stochastic sampling.
    bits: number of bits to perform quantization.
    alpha: ternary is -alpha or +alpha. Threshold is also scaled by alpha.
    threshold: threshold to apply "dropout" or dead band (0 value).
    use_stochastic_rounding: if true, we perform stochastic rounding.

  Returns:
    Computation of sign within the threshold.
  """

  def __init__(self, alpha=1.0, threshold=0.33, use_stochastic_rounding=False):
    self.alpha = alpha
    self.bits = 2
    self.threshold = threshold
    self.use_stochastic_rounding = use_stochastic_rounding

  def __call__(self, x):
    if self.use_stochastic_rounding:
      x = _round_through(
          x, use_stochastic_rounding=self.use_stochastic_rounding)
    return x + tf.stop_gradient(
        -x + self.alpha * tf.where(tf.abs(x) < self.threshold,
                                   tf.zeros_like(x), tf.sign(x)))

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def get_config(self):
    config = {
        "alpha":
            self.alpha,
        "threshold":
            self.threshold,
        "use_stochastic_rounding":
            self.use_stochastic_rounding
    }
    return config


class stochastic_binary(object):  # pylint: disable=invalid-name
  """Computes a stochastic activation function returning -alpha or +alpha.

  Computes straight-through approximation using random sampling to make
  E[dL/dy] = E[dL/dx], and computing the sign function. See explanation above.

  Attributes:
    x: tensor to perform sign opertion with stochastic sampling.
    alpha: binary is -alpha or +alpha.`
    bits: number of bits to perform quantization.

  Returns:
    Computation of sign with stochastic sampling with straight through gradient.
  """

  def __init__(self, alpha=1.0):
    self.alpha = alpha
    self.bits = 1

  def __call__(self, x):
    assert self.alpha != 0
    p = _sigmoid(x / self.alpha)
    k_sign = tf.sign(p - tf.random.uniform(tf.shape(x)))
    # we should not need this, but if tf.sign is not safe if input is
    # exactly 0.0
    k_sign += (1.0 - tf.abs(k_sign))
    return x + tf.stop_gradient(-x + self.alpha * k_sign)

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def get_config(self):
    config = {"alpha": self.alpha}
    return config


class binary(object):  # pylint: disable=invalid-name
  """Computes the sign(x) returning a value between -alpha and alpha.

  Although we cannot guarantee E[dL/dy] = E[dL/dx] if we do not use the
  stochastic sampling, we still use the ST approximation.

  Modified from original binary to match QNN implementation.

  Attributes:
    x: tensor to perform sign_through.
    bits: number of bits to perform quantization.
    use_01: if True, return {0,1} instead of {-1,+1}.
    alpha: binary is -alpha or +alpha.
    use_stochastic_rounding: if true, we perform stochastic rounding.

  Returns:
    Computation of sign operation with straight through gradient.
  """

  def __init__(self, use_01=False, alpha=1.0, use_stochastic_rounding=False):
    self.use_01 = use_01
    self.bits = 1
    self.alpha = alpha
    self.use_stochastic_rounding = use_stochastic_rounding

  def __call__(self, x):
    assert self.alpha != 0
    if self.use_stochastic_rounding:
      x = self.alpha * _round_through(
          x / self.alpha, use_stochastic_rounding=self.use_stochastic_rounding)

    k_sign = tf.sign(x)
    if self.use_stochastic_rounding:
      k_sign += (1.0 - tf.abs(k_sign)) * (
          2.0 * tf.round(tf.random.uniform(tf.shape(x))) - 1.0)
    else:
      k_sign += (1.0 - tf.abs(k_sign))
    if self.use_01:
      k_sign = (k_sign + 1.0) / 2.0
    return x + tf.stop_gradient(-x + self.alpha * k_sign)

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def get_config(self):
    config = {
        "use_01":
            self.use_01,
        "alpha":
            self.alpha,
        "use_stochastic_rounding":
            self.use_stochastic_rounding
    }
    return config


class quantized_relu(object):  # pylint: disable=invalid-name
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
    use_stochastic_rounding: if true, we perform stochastic rounding.

  Returns:
    Function that performs relu + quantization to bits >= 0.
  """

  def __init__(self, bits=8, integer=0, use_sigmoid=0,
               use_stochastic_rounding=False):
    self.bits = bits
    self.integer = integer
    self.use_sigmoid = use_sigmoid
    self.use_stochastic_rounding = use_stochastic_rounding

  def __call__(self, x):
    m = pow(2, self.bits)
    m_i = pow(2, self.integer)

    if self.use_sigmoid:
      p = _sigmoid(x / m_i) * m
      xq = m_i * tf.keras.backend.clip(
          2.0 * (_round_through(p, self.use_stochastic_rounding) / m) - 1.0,
          0.0, 1.0 - 1.0 / m)
    else:
      p = x * m / m_i
      xq = m_i * tf.keras.backend.clip(
          _round_through(p, self.use_stochastic_rounding) / m,
          0.0, 1.0 - 1.0 / m)
    return xq

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def get_config(self):
    config = {
        "bits":
            self.bits,
        "integer":
            self.integer,
        "use_sigmoid":
            self.use_sigmoid,
        "use_stochastic_rounding":
            self.use_stochastic_rounding
    }
    return config


class quantized_ulaw(object):  # pylint: disable=invalid-name
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
    u_law_p = tf.sign(rp) * tf.keras.backend.log(
        1 + self.u * tf.abs(rp)) / tf.keras.backend.log(1 + self.u)
    xq = m_i * tf.keras.backend.clip(u_law_p, -1.0 +
                                     (1.0 * self.symmetric) / m, 1.0 - 1.0 / m)
    return xq

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def get_config(self):
    config = {
        "bits":
            self.bits,
        "integer":
            self.integer,
        "symmetric":
            self.symmetric,
        "u":
            self.u
    }
    return config


class quantized_tanh(object):  # pylint: disable=invalid-name
  """Computes a quantized tanh to a number of bits.

  Modified from:

  [https://github.com/BertMoons/QuantizedNeuralNetworks-Keras-Tensorflow]

  Attributes:
    bits: number of bits to perform quantization.
    integer: number of bits to the left of the decimal point.
    symmetric: if true, we will have the same number of values for positive
               and negative numbers.
    use_stochastic_rounding: if true, we perform stochastic rounding.

  Returns:
    Function that performs tanh + quantization to bits in the range -1.0 to 1.0.
  """

  def __init__(self, bits=8, integer=0, symmetric=0,
               use_stochastic_rounding=False):
    self.bits = bits
    self.integer = integer
    self.symmetric = symmetric
    self.use_stochastic_rounding = use_stochastic_rounding

  def __call__(self, x):
    non_sign_bits = self.bits - 1
    m = pow(2, non_sign_bits)
    m_i = pow(2, self.integer)
    p = _sigmoid(x / m_i) * m
    xq = m_i * tf.keras.backend.clip(
        2.0 *
        (_round_through(p, self.use_stochastic_rounding) / m) - 1.0, -1.0 +
        (1.0 * self.symmetric) / m, 1.0 - 1.0 / m)
    return xq

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def get_config(self):
    config = {
        "bits":
            self.bits,
        "integer":
            self.integer,
        "symmetric":
            self.symmetric,
        "use_stochastic_rounding":
            self.use_stochastic_rounding
    }
    return config


class quantized_po2(object):  # pylint: disable=invalid-name
  """Quantizes to the closest power of 2."""

  def __init__(self,
               bits=8,
               max_value=-1,
               use_stochastic_rounding=False,
               quadratic_approximation=False):
    self.bits = bits
    self.max_value = max_value
    self.use_stochastic_rounding = use_stochastic_rounding

    # if True, round to the exponent for sqrt(x),
    # so that the return value can be divided by two without remainder.
    self.quadratic_approximation = quadratic_approximation

  def __call__(self, x):

    need_exponent_sign_bit = (self.max_value > 1)
    non_sign_bits = self.bits - 1
    min_exp = -2**(non_sign_bits - need_exponent_sign_bit)
    max_exp = 2**(non_sign_bits - need_exponent_sign_bit) - 1
    eps = tf.keras.backend.epsilon()
    if min_exp < np.log2(eps):
      warnings.warn(
          "QKeras: min_exp in po2 quantizer is smaller than tf.epsilon()")

    if self.max_value != -1:
      max_exp = np.round(np.log2(self.max_value + eps))

    x_sign = tf.sign(x)
    x_sign += (1.0 - tf.abs(x_sign))
    log2 = np.log(2.0)

    # if True, round to the exponent for sqrt(x),
    # so that the return value can be divided by two without remainder.
    if self.quadratic_approximation:
      q_factor = 2.0
    else:
      q_factor = 1.0

    if self.use_stochastic_rounding:
      if self.quadratic_approximation:
        x_log2 = stochastic_round_po2(tf.sqrt(x))
      else:
        x_log2 = stochastic_round_po2(x)
    else:
      if self.quadratic_approximation:
        x_log2 = _round_through(tf.keras.backend.log(tf.sqrt(x) + eps) / log2)
      else:
        x_log2 = _round_through(tf.keras.backend.log(tf.abs(x) + eps) / log2)
    x_clipped = q_factor * tf.keras.backend.clip(x_log2, min_exp, max_exp)
    return x + tf.stop_gradient(-x + x_sign * pow(2.0, x_clipped))

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def get_config(self):
    config = {
        "bits":
            self.bits,
        "max_value":
            self.max_value,
        "use_stochastic_rounding":
            self.use_stochastic_rounding,
        "quadratic_approximation":
            self.quadratic_approximation
    }
    return config


class quantized_relu_po2(object):  # pylint: disable=invalid-name
  """Quantizes to the closest power of 2."""

  def __init__(self, bits=8, max_value=-1, use_stochastic_rounding=False,
               quadratic_approximation=False):
    self.bits = bits
    self.max_value = max_value
    self.use_stochastic_rounding = use_stochastic_rounding

    # if True, round to the exponent for sqrt(x),
    # so that the return value can be divided by two without remainder.
    self.quadratic_approximation = quadratic_approximation

  def __call__(self, x):

    need_exponent_sign_bit = (self.max_value > 1)
    min_exp = -2**(self.bits - need_exponent_sign_bit)
    max_exp = 2**(self.bits - need_exponent_sign_bit) - 1
    eps = tf.keras.backend.epsilon()

    if min_exp < np.log2(eps):
      warnings.warn(
          "QKeras: min_exp in quantized_relu_po2 quantizer "
          "is smaller than tf.epsilon()")

    log2 = np.log(2.0)

    if self.max_value != -1:
      max_exp = np.round(np.log2(self.max_value + eps))

    if self.quadratic_approximation:
      q_factor = 2.0
    else:
      q_factor = 1.0
    x = tf.maximum(x, 0)

    if self.use_stochastic_rounding:
      # if True, approximate the power of two to the sqrt(x)
      # use q_factor to recover the value in x_clipped.
      if self.quadratic_approximation:
        x_log2 = stochastic_round_po2(tf.sqrt(x))
      else:
        x_log2 = stochastic_round_po2(x)
    else:
      if self.quadratic_approximation:
        x_log2 = _round_through(tf.keras.backend.log(tf.sqrt(x) + eps) / log2)
      else:
        x_log2 = _round_through(tf.keras.backend.log(tf.abs(x) + eps) / log2)
    x_clipped = q_factor * tf.keras.backend.clip(x_log2, min_exp, max_exp)
    return x + tf.stop_gradient(-x + pow(2.0, x_clipped))

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def get_config(self):
    config = {
        "bits":
            self.bits,
        "max_value":
            self.max_value,
        "use_stochastic_rounding":
            self.use_stochastic_rounding,
        "quadratic_approximation":
            self.quadratic_approximation
    }
    return config


def get_quantizer(identifier):
  if identifier is None:
    return None
  if isinstance(identifier, dict):
    return deserialize_keras_object(identifier,
      module_objects=globals(),
      printable_module_name='quantizer')
  elif isinstance(identifier, six.string_types):
    return safe_eval(identifier, globals())
  elif callable(identifier):
    return identifier
  else:
    raise ValueError('Could not interpret quantizer identifier: ' + str(identifier))


def get_quantized_initializer(w_initializer, w_range):
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

