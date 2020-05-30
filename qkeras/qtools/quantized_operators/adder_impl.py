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
"""adder operation implementation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from qkeras.qtools.quantized_operators import accumulator_impl
from qkeras.qtools.quantized_operators import quantizer_impl


def po2_qbits_converter(po2_quantizer: quantizer_impl.IQuantizer):
  """convert a po2 quantizer to fixedpoint quantizer."""

  (bits_from_po2, int_bits_from_po2) = accumulator_impl.po2_to_qbits(
      po2_quantizer)
  qbits_quantizer = quantizer_impl.QuantizedBits()
  qbits_quantizer.bits = bits_from_po2
  qbits_quantizer.int_bits = int_bits_from_po2
  qbits_quantizer.is_signed = po2_quantizer.is_signed

  return qbits_quantizer


class IAdderImpl(abc.ABC):
  """abstract class for adder."""

  @staticmethod
  @abc.abstractmethod
  def implemented_as():
    pass


class FixedPointAdder(IAdderImpl):
  """adder for fixed point."""

  def __init__(self, quantizer_1, quantizer_2):
    self.output = quantizer_impl.QuantizedBits()
    self.output.bits = max(quantizer_1.bits, quantizer_2.bits) + 1
    self.output.int_bits = max(quantizer_1.int_bits,
                               quantizer_2.int_bits) + 1
    self.output.is_signed = quantizer_1.is_signed | quantizer_2.is_signed
    self.output.mode = 0
    self.output.is_floating_point = False
    self.output.is_po2 = 0

  @staticmethod
  def implemented_as():
    return "add"


class FloatingPointAdder(IAdderImpl):
  """floating point adder."""

  def __init__(self, quantizer_1, quantizer_2):
    bits = max(quantizer_1.bits, quantizer_2.bits)
    self.output = quantizer_impl.FloatingPoint(
        bits=bits)

  @staticmethod
  def implemented_as():
    return "add"


class Po2FixedPointAdder(IAdderImpl):
  """adder between po2 and fixed point."""

  def __init__(self, quantizer_1, quantizer_2):

    if quantizer_1.is_po2:
      po2_quantizer = quantizer_1
      fixedpoint_quantizer = quantizer_2
    else:
      po2_quantizer = quantizer_2
      fixedpoint_quantizer = quantizer_1

    # convert po2 to qbits first
    po2_qbits_quantizer = po2_qbits_converter(po2_quantizer)

    # qbits + qbits -> FixedPointAdder
    self.output = FixedPointAdder(po2_qbits_quantizer,
                                  fixedpoint_quantizer).output

  @staticmethod
  def implemented_as():
    return "add"


class Po2Adder(IAdderImpl):
  """adder for po2 type."""

  def __init__(self, quantizer_1, quantizer_2):
    qbits_quantizer_1 = po2_qbits_converter(quantizer_1)
    qbits_quantizer_2 = po2_qbits_converter(quantizer_2)
    self.output = FixedPointAdder(qbits_quantizer_1,
                                  qbits_quantizer_2).output

  @staticmethod
  def implemented_as():
    return "add"
