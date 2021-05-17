# Lint as: python3
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
"""atomic quantizer implementation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import math

import numpy as np
from qkeras import quantizers

FLOATINGPOINT_BITS = 32


def get_exp(quantizer):
  """get max/min exp value for relu_po2 or quantized_po2."""

  if quantizer.is_signed:
    non_sign_bits = quantizer.bits - 1
  else:
    non_sign_bits = quantizer.bits

  min_exp = -2 ** (non_sign_bits - 1)
  max_exp_orig = 2 ** (non_sign_bits - 1) - 1

  max_exp = max_exp_orig
  # max_value caps how many int_bits actually allowed
  if quantizer.max_val_po2 != -1:
    if quantizer.max_val_po2 <= 0:
      max_exp = 0
    else:
      max_exp = math.ceil(np.log2(quantizer.max_val_po2))
      max_exp = min(max_exp, max_exp_orig)

  # if max_exp<0. no need to expand int_bits
  max_exp = max(0, max_exp)

  return (-min_exp, max_exp)


class IQuantizer(abc.ABC):
  """abstract class for quantizer."""

  def __init__(self):
    self.mode = -1
    self.bits = -1
    self.int_bits = -1
    self.is_signed = 0
    self.is_floating_point = False
    self.max_val_po2 = -1
    self.is_po2 = 0
    self.name = None
    self.op_type = "quantizer"


class QuantizedBits(IQuantizer):
  """quantized bits.

  Attributes:
    mode: index of the current quantizer in
          MultiplierFactory.multiplier_impl_table
    bits: total bits
    int_bits: integer bits
    is_signed: if a signed number
    name: quantizer name
  """

  def __init__(self):
    super().__init__()
    self.mode = 0
    self.is_signed = 1
    self.name = "quantized_bits"

  def convert_qkeras_quantizer(
      self, quantizer: quantizers.quantized_bits):
    self.mode = 0
    self.bits = quantizer.bits
    self.int_bits = quantizer.integer
    self.is_signed = quantizer.keep_negative

  def convert_to_qkeras_quantizer(
      self, symmetric=1, alpha=None, use_stochastic_rounding=False,
      scale_axis=None, qnoise_factor=1.0):
    """convert qtools quantizer to qkeras quantizer."""

    return quantizers.quantized_bits(
        bits=self.bits, integer=self.int_bits, keep_negative=self.is_signed,
        symmetric=symmetric, alpha=alpha,
        use_stochastic_rounding=use_stochastic_rounding,
        scale_axis=scale_axis, qnoise_factor=qnoise_factor)


class QuantizedTanh(QuantizedBits):
  """same as quantized bits."""

  def __init__(self):
    super().__init__()
    self.name = "quantized_tanh"

  def convert_qkeras_quantizer(
      self, quantizer: quantizers.quantized_tanh):
    self.mode = 0
    self.bits = quantizer.bits
    self.is_signed = 1

  def convert_to_qkeras_quantizer(
      self, symmetric=False, use_stochastic_rounding=False):
    """convert qtools quantizer to qkeras quantizer."""

    return quantizers.quantized_tanh(
        bits=self.bits, use_stochastic_rounding=use_stochastic_rounding,
        symmetric=symmetric)


class QuantizedUlaw(QuantizedBits):
  """quantized ulaw type."""

  # same as quantized bits
  def __init__(self):
    super().__init__()
    self.name = "quantized_ulaw"

  def convert_qkeras_quantizer(
      self, quantizer: quantizers.quantized_ulaw):
    self.mode = 0
    self.bits = quantizer.bits
    self.int_bits = quantizer.integer
    self.is_signed = 1

  def convert_to_qkeras_quantizer(self, symmetric=0, u=255.0):
    """convert qtools quantizer to qkeras quantizer."""

    return quantizers.quantized_ulaw(
        bits=self.bits, integer=self.int_bits, symmetric=symmetric, u=u)


class Binary(IQuantizer):
  """binary quantizer."""

  def __init__(self, use_01=False):
    super().__init__()
    if use_01:
      self.mode = 4
      self.is_signed = 0
    else:
      self.mode = 3
      self.is_signed = 1

    self.bits = 1
    self.int_bits = 1
    self.use_01 = use_01
    self.name = "binary"

  def convert_qkeras_quantizer(self, quantizer: quantizers.binary):
    if quantizer.use_01:
      self.mode = 4
      self.is_signed = 0
    else:
      self.mode = 3
      self.is_signed = 1

    self.use_01 = quantizer.use_01

  def convert_to_qkeras_quantizer(self, alpha=None,
                                  use_stochastic_rounding=False):
    """convert qtools quantizer to qkeras quantizer."""

    return quantizers.binary(use_01=self.use_01, alpha=alpha,
                             use_stochastic_rounding=use_stochastic_rounding)


class StochasticBinary(Binary):
  """stochastic binary quantizer."""

  # same as binary(-1, 1)
  def __init__(self):
    super().__init__(use_01=False)
    self.name = "stochastic_binary"

  def convert_qkeras_quantizer(
      self, quantizer: quantizers.stochastic_binary):
    """convert qkeras quantizer to qtools quantizer."""

    pass

  def convert_to_qkeras_quantizer(self, alpha=None, temperature=6.0,
                                  use_real_sigmoid=True):
    """convert qtools quantizer to qkeras quantizer."""

    return quantizers.stochastic_binary(alpha=alpha, temperature=temperature,
                                        use_real_sigmoid=use_real_sigmoid)


class Bernoulli(Binary):
  """bernoulli quantizer. same as binary(0, 1)."""

  def __init__(self):
    super().__init__(use_01=True)
    self.name = "bernoulli"

  def convert_qkeras_quantizer(self, quantizer: quantizers.bernoulli):
    pass

  def convert_to_qkeras_quantizer(self, alpha=None, temperature=6.0,
                                  use_real_sigmoid=True):
    """convert qtools quantizer to qkeras quantizer."""

    return quantizers.bernoulli(alpha=alpha, temperature=temperature,
                                use_real_sigmoid=use_real_sigmoid)


class QuantizedRelu(IQuantizer):
  """quantized relu quantizer."""

  def __init__(self):
    super().__init__()
    self.is_signed = 0
    self.name = "quantized_relu"

  def convert_qkeras_quantizer(
      self, quantizer: quantizers.quantized_relu):
    """convert from qkeras quantizer."""

    bits = quantizer.bits
    int_bits = quantizer.integer

    if bits == 1 and int_bits == 1:
      mode = 4
    else:
      mode = 0

    self.mode = mode
    self.bits = bits
    self.int_bits = int_bits
    if hasattr(quantizer, "negative_slope") and quantizer.negative_slope != 0:
      self.is_signed = 1

  def convert_to_qkeras_quantizer(
      self, use_sigmoid=0, negative_slope=0.0, use_stochastic_rounding=False,
      relu_upper_bound=None, is_quantized_clip=True, qnoise_factor=1.0):
    """convert qtools quantizer to qkeras quantizer."""

    return quantizers.quantized_relu(
        bits=self.bits, integer=self.int_bits, use_sigmoid=use_sigmoid,
        negative_slope=negative_slope,
        use_stochastic_rounding=use_stochastic_rounding,
        relu_upper_bound=relu_upper_bound,
        is_quantized_clip=is_quantized_clip,
        qnoise_factor=qnoise_factor)


class Ternary(IQuantizer):
  """ternary(0, 1, -1)."""

  def __init__(self):
    super().__init__()
    self.mode = 2
    self.bits = 2
    self.int_bits = 2
    self.is_signed = 1
    self.name = "ternary"

  def convert_qkeras_quantizer(
      self, quantizer: quantizers.ternary):
    pass

  def convert_to_qkeras_quantizer(
      self, alpha=None, threshold=None, use_stochastic_rounding=False,
      number_of_unrolls=5):
    """convert qtools quantizer to qkeras quantizer."""

    return quantizers.ternary(
        alpha=alpha, threshold=threshold,
        use_stochastic_rounding=use_stochastic_rounding,
        number_of_unrolls=number_of_unrolls)


class StochasticTernary(Ternary):
  """stochastic ternary."""

  def __init__(self):
    super().__init__()
    self.name = "stochastic_ternary"

  # same as ternary
  def convert_qkeras_quantizer(
      self, quantizer: quantizers.stochastic_ternary):
    pass

  def convert_to_qkeras_quantizer(
      self, alpha=None, threshold=None, temperature=8.0,
      use_real_sigmoid=True, number_of_unrolls=5):
    """convert qtools quantizer to qkeras quantizer."""

    return quantizers.stochastic_ternary(
        alpha=alpha, threshold=threshold, temperature=temperature,
        use_real_sigmoid=use_real_sigmoid,
        number_of_unrolls=number_of_unrolls)


class FloatingPoint(IQuantizer):
  """float32."""

  def __init__(self, bits):
    super().__init__()
    self.mode = 5
    self.bits = bits
    self.int_bits = -1
    self.is_signed = 1
    self.is_floating_point = True
    self.name = "floating_point"

  def convert_qkeras_quantizer(self, bits):
    pass

  def convert_to_qkeras_quantizer(self, bits):
    pass


class PowerOfTwo(IQuantizer):
  """po2."""

  def __init__(self, is_signed=True):
    super().__init__()
    self.mode = 1
    self.is_po2 = 1
    self.is_signed = is_signed
    self.inference_value_counts = -1

    if is_signed:
      self.name = "quantized_po2"
    else:
      self.name = "quantized_relu_po2"

  def convert_qkeras_quantizer(self, quantizer):
    """convert qkeras quantizer to qtools quantizer."""

    assert "po2" in quantizer.__class__.__name__

    if quantizer.__class__.__name__ == "quantized_po2":
      self.is_signed = 1
      self.name = "quantized_po2"

    elif quantizer.__class__.__name__ == "quantized_relu_po2":
      super().__init__()
      self.is_signed = 0
      self.name = "quantized_relu_po2"

    bits = quantizer.bits
    max_val_po2 = quantizer.max_value
    if not max_val_po2:
      self.max_val_po2 = -1
    else:
      self.max_val_po2 = max_val_po2
    self.bits = bits
    self.int_bits = bits

  def convert_to_qkeras_quantizer(
      self, negative_slope=0, use_stochastic_rounding=False,
      quadratic_approximation=False):
    """convert qtools quantizer to qkeras quantizer."""

    if self.is_signed:
      # quantized_po2
      return quantizers.quantized_po2(
          bits=self.bits,
          max_value=self.max_val_po2 if self.max_val_po2 >= 0 else None,
          use_stochastic_rounding=use_stochastic_rounding,
          quadratic_approximation=quadratic_approximation)
    else:
      # quantized_relu_po2
      return quantizers.quantized_relu_po2(
          bits=self.bits,
          max_value=self.max_val_po2 if self.max_val_po2 >= 0 else None,
          negative_slope=negative_slope,
          use_stochastic_rounding=use_stochastic_rounding,
          quadratic_approximation=quadratic_approximation)

  def get_min_max_exp(self):
    return get_exp(self)

  def quantizer_bits_calculator(self, val):
    """calculate how many bits needed."""

    # calculate how many bits are required to represent a po2 value.
    # val can be +/- values, can be integer or franctional number.
    # needs to be dealt seperately.

    sign_bit = val < 0

    # get rid of sign
    val = abs(val)

    if val == 0:
      # val of 0 is special case; qkeras uses mininmum
      # number to represent 0
      non_sign_bits = self.bits - sign_bit
    else:
      exp_value = np.log2(val)

      # exp_value should be integer
      if abs(np.round(exp_value) - exp_value) > 0:
        raise ValueError("ERROR: {} is not a po2 value!".format(val))

      exp_value = int(exp_value)

      # for n bits, the range of values it can represent is:
      # min_val = -2 ** (n - 1)
      # max_val = 2 ** (n - 1) - 1
      if exp_value == 0:
        non_sign_bits = 1
      elif exp_value > 0:
        # e.g., 16 needs 5 bits + 1 exp sign bit,
        # 15 needs 4 bits + 1 exp sign bit
        non_sign_bits = math.floor(np.log2(exp_value)) + 1 + 1
      else:
        # e.g., -16 needs 4 bits + 1 exp sign bit
        non_sign_bits = math.ceil(np.log2(abs(exp_value))) + 1

    return (sign_bit, non_sign_bits)

  def update_quantizer(self, val, reset=False):
    """update quantizer bits according to the input value.

    Args:
      val: input value
      reset: True->disregard current quantizer bits and reset
        it according to the given value; False-> update the quantizer
        bits with given value.
        quantizer.bits = min(existing_bits, bits required by val)

    Returns:
      Update existing po2 quantizer bits by val.
       quantizer.bits = min(existing_bits, bits required by val)
    """
    (sign_bit, non_sign_bits) = self.quantizer_bits_calculator(val)

    if reset:
      self.bits = sign_bit + non_sign_bits
    else:
      # avoid input value exceeding quantizer limit
      self.bits = min(self.bits, sign_bit + non_sign_bits)

    self.int_bits = self.bits
    self.max_val_po2 = min(val, self.max_val_po2)
    self.is_signed = sign_bit

    if sign_bit:
      self.name = "quantized_po2"
    else:
      self.name = "quantized_relu_po2"

  def update_inference_values(self, weights):
    """find how many different values in weights in the po2 quantizer."""

    inference_value_counts = len(set(weights.flatten()))
    self.inference_value_counts = inference_value_counts


class ReluPowerOfTwo(PowerOfTwo):
  """relu po2."""

  def __init__(self):
    super().__init__()
    self.mode = 1
    self.is_po2 = 1
    self.is_signed = 0
    self.name = "quantized_relu_po2"

  def convert_qkeras_quantizer(
      self, quantizer: quantizers.quantized_relu_po2):

    self.bits = quantizer.bits
    self.int_bits = quantizer.bits
    if not quantizer.max_value:
      self.max_val_po2 = -1
    else:
      self.max_val_po2 = quantizer.max_value

  def convert_to_qkeras_quantizer(
      self, negative_slope=0, use_stochastic_rounding=False,
      quadratic_approximation=False):
    """convert qtools quantizer to qkeras quantizer."""

    # quantized_relu_po2
    return quantizers.quantized_relu_po2(
        bits=self.bits,
        max_value=self.max_val_po2 if self.max_val_po2 >= 0 else None,
        negative_slope=negative_slope,
        use_stochastic_rounding=use_stochastic_rounding,
        quadratic_approximation=quadratic_approximation)
