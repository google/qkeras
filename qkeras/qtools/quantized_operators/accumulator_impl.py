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
"""Accumulator operation implementation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from absl import logging
import numpy as np

from qkeras.qtools.quantized_operators import multiplier_impl
from qkeras.qtools.quantized_operators import quantizer_impl


def po2_to_qbits(quantizer: quantizer_impl.IQuantizer):
  """convert po2 type to qbits type."""

  (min_exp, max_exp) = quantizer.get_min_max_exp()
  # min_exp is number of bits needed on the right in qbits
  # max_exp is number of bits needed on the left in qbits
  unsigned_bits = min_exp + max_exp
  int_bits = max_exp
  sign_bit = quantizer.is_signed
  bits = sign_bit + unsigned_bits

  return (int(bits), int(int_bits))


class IAccumulator(abc.ABC):
  """abstract class for accumulator."""

  @staticmethod
  @abc.abstractmethod
  def implemented_as():
    pass


class FloatingPointAccumulator(IAccumulator):
  """class for floating point accumulator."""

  def __init__(
      self,
      multiplier: multiplier_impl.IMultiplier
  ):
    super().__init__()

    self.multiplier = multiplier
    self.output = quantizer_impl.FloatingPoint(
        bits=self.multiplier.output.bits)
    self.output.bits = self.multiplier.output.bits
    self.output.int_bits = -1
    self.output.is_signed = self.multiplier.output.is_signed
    self.output.is_floating_point = True
    self.output.op_type = "accumulator"

  @staticmethod
  def implemented_as():
    return "add"


class FixedPointAccumulator(IAccumulator):
  """class for fixed point accumulator."""

  def __init__(
      self,
      kernel_shape,
      multiplier: multiplier_impl.IMultiplier,
      use_bias=True
  ):
    super().__init__()

    if len(kernel_shape) not in (
        2,
        4,
    ):
      logging.fatal(
          "unsupported kernel shape, "
          "it is neither a dense kernel of length 2,"
          " nor a convolution kernel of length 4")

    kernel_shape_excluding_output_dim = kernel_shape[:-1]
    kernel_add_ops = np.prod(kernel_shape_excluding_output_dim)

    # bias are associate with filters; each filter adds 1 bias
    bias_add = 1 if use_bias else 0

    add_ops = kernel_add_ops + bias_add
    self.log_add_ops = int(np.ceil(np.log2(add_ops)))

    self.multiplier = multiplier
    self.output = quantizer_impl.QuantizedBits()
    self.output.bits = self.log_add_ops + self.multiplier.output.bits
    self.output.int_bits = self.log_add_ops + self.multiplier.output.int_bits
    self.output.is_signed = self.multiplier.output.is_signed
    self.output.op_type = "accumulator"

    assert not self.multiplier.output.is_floating_point
    self.output.is_floating_point = False

  @staticmethod
  def implemented_as():
    return "add"


class Po2Accumulator(FixedPointAccumulator):
  """accumulator for po2."""

  # multiplier is po2. multiplier output needs to convert
  # to Fixedpoint before Accumulator.

  def __init__(
      self,
      kernel_shape,
      multiplier: multiplier_impl.IMultiplier,
      use_bias=True
  ):
    super().__init__(kernel_shape, multiplier, use_bias)

    assert multiplier.output.is_po2
    # convert multiplier output from po2 to quantized_bits
    (bits_from_po2multiplier, int_bits_from_po2multiplier) = po2_to_qbits(
        multiplier.output)

    self.output.bits = self.log_add_ops + int(bits_from_po2multiplier)
    self.output.int_bits = self.log_add_ops + int(int_bits_from_po2multiplier)
    self.output.op_type = "accumulator"

  @staticmethod
  def implemented_as():
    return "add"
