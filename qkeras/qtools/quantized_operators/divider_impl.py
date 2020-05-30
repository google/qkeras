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
"""Divider operation implementation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np


class IDividerImpl(abc.ABC):
  """abstract class for divider."""

  def __init__(self, numerator_quantizer, denominator_quantizer,
               output_quantizer):
    self.numerator_quantizier = numerator_quantizer
    self.denominator_quantizer = denominator_quantizer
    self.output = output_quantizer

  @staticmethod
  @abc.abstractmethod
  def implemented_as():
    pass


class FloatingPointDivider(IDividerImpl):
  """floating point divider."""

  def __init__(self, numerator_quantizer, denominator_quantizer,
               output_quantizer):

    super().__init__(numerator_quantizer, denominator_quantizer,
                     output_quantizer)
    if self.output.bits is None:
      # decide f16/f32 according to numerator/denominator type
      bits = 0
      if numerator_quantizer.is_floating_point:
        bits = max(bits, numerator_quantizer.bits)
      if denominator_quantizer.is_floating_point:
        bits = max(bits, denominator_quantizer.bits)

      self.output.bits = bits

    self.gate_bits = self.output.bits
    self.gate_factor = 1

  @staticmethod
  def implemented_as():
    # TODO(lishanok): change cost from "mul" to "divide"
    return "mul"


class Shifter(IDividerImpl):
  """shifter type."""

  # other_datatype/po2
  def __init__(self, numerator_quantizer, denominator_quantizer,
               output_quantizer):
    super().__init__(numerator_quantizer, denominator_quantizer,
                     output_quantizer)

    qbit_quantizer = numerator_quantizer
    po2_quantizer = denominator_quantizer

    (min_exp, max_exp) = po2_quantizer.get_min_max_exp()

    # since it's a divider, min_exp and max_exp swap
    # for calculating right and left shift
    tmp = min_exp
    min_exp = max_exp
    max_exp = tmp

    qbits_bits = qbit_quantizer.bits
    qbits_int_bits = qbit_quantizer.int_bits

    self.output.bits = int(qbits_bits + max_exp + min_exp)
    if (not qbit_quantizer.is_signed) and po2_quantizer.is_signed:
      # if qbit is signed, qbits_bits already has the sign_bit,
      # no need to +1,
      # if qbit is un_signed, po2 is unsigned, no need to +1
      # if qbit is un_signed, po2 is signed, min_exp and max_exp
      # didnot include sign_bit,
      # therefore need to +1
      self.output.bits += 1

    self.output.int_bits = int(qbits_int_bits + max_exp)
    self.output.is_signed = qbit_quantizer.is_signed |\
                            po2_quantizer.is_signed
    self.output.is_floating_point = False

    if po2_quantizer.inference_value_counts > 0:
      # during qbn inference, count number of unique values
      self.gate_factor = po2_quantizer.inference_value_counts * 0.3
      self.gate_bits = qbits_bits
    else:
      # programmable shifter, similar to sum gate
      self.gate_factor = 1
      b = np.sqrt(2 ** po2_quantizer.bits * qbits_bits)
      self.gate_bits = b * np.log10(b)

  @staticmethod
  def implemented_as():
    return "shifter"


class Subtractor(IDividerImpl):
  """subtractor quantizer."""

  # subtractor is only possible when numerator and denominator
  # are both po2 quantizers.

  def __init__(self, numerator_quantizer, denominator_quantizer,
               output_quantizer):
    super().__init__(numerator_quantizer, denominator_quantizer,
                     output_quantizer)

    self.output.bits = max(numerator_quantizer.bits,
                           denominator_quantizer.bits) + 1
    self.output.int_bits = max(numerator_quantizer.int_bits,
                               denominator_quantizer.int_bits) + 1
    self.output.is_signed = 1
    self.output.is_floating_point = False
    self.output.is_po2 = 1

    if (numerator_quantizer.max_val_po2 == -1 or
        denominator_quantizer.max_val_po2 == -1):
      self.output.max_val_po2 = -1
    else:
      # Adder is two po2_value multiply with each other
      self.output.max_val_po2 = numerator_quantizer.max_val_po2 /\
                                denominator_quantizer.max_val_po2

    if "po2" in output_quantizer.name:
      # po2 * po2
      if self.output.is_signed:
        output_quantizer.name = "quantized_po2"
      else:
        output_quantizer.name = "quantized_relu_po2"

    self.gate_bits = self.output.bits
    self.gate_factor = 1

  @staticmethod
  def implemented_as():
    return "add"
