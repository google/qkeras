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
"""multiplier operation implementations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np

from qkeras.qtools.quantized_operators import quantizer_impl


class IMultiplier(abc.ABC):
  """abstract class for multiplier.

  This class is about how multiplier is implemented in hardware, which can be
     mux gate, shifter, adder, etc.
  """

  def __init__(self, weight_quantizer: quantizer_impl.IQuantizer,
               input_quantizer: quantizer_impl.IQuantizer,
               output_quantizer: quantizer_impl.IQuantizer):
    self.input = input_quantizer
    self.weights = weight_quantizer
    self.output = output_quantizer
    self.output.op_type = "multiplier"

  @staticmethod
  @abc.abstractmethod
  def implemented_as():
    pass

  def name(self) -> str:
    return self.output.name

  def output_quantizer(self):
    return self.output


def assert_neither_input_and_weights_is_floating_point(
    multiplier: IMultiplier):
  """assert non float type."""

  assert not multiplier.input.is_floating_point
  assert not multiplier.weights.is_floating_point


class Mux(IMultiplier):
  """Use mux for the hardware implementation of multiplier."""

  # binary(1,-1)/ternary * other_datatype
  def __init__(self, weight_quantizer: quantizer_impl.IQuantizer,
               input_quantizer: quantizer_impl.IQuantizer,
               output_quantizer: quantizer_impl.IQuantizer):
    super().__init__(weight_quantizer, input_quantizer,
                     output_quantizer)
    self.output.is_signed = self.input.is_signed | self.weights.is_signed

    if any(s in weight_quantizer.name for s in ["binary", "ternary"]):
      self.output.bits = input_quantizer.bits
      self.output.int_bits = input_quantizer.int_bits
      if not input_quantizer.is_signed and weight_quantizer.is_signed:
        self.output.bits += 1

      # multiplier factor for gate counts
      # gate_factor is the relative energy of given gate comparing
      # to an Add gate, giving that Add gate is 1
      if "binary" in weight_quantizer.name:
        self.gate_factor = 0.3
      else:
        self.gate_factor = 2 * 0.3
      self.gate_bits = input_quantizer.bits

    else:
      self.output.bits = weight_quantizer.bits
      self.output.int_bits = weight_quantizer.int_bits
      if not weight_quantizer.is_signed and input_quantizer.is_signed:
        self.output.bits += 1

      # multiplier factor for gate counts
      if input_quantizer.name == "binary":
        self.gate_factor = 0.3
      else:
        self.gate_factor = 2 * 0.3
      self.gate_bits = weight_quantizer.bits

    if "po2" in output_quantizer.name:
      if self.output.is_signed:
        output_quantizer.name = "quantized_po2"
      else:
        output_quantizer.name = "quantized_relu_po2"

      if "po2" in weight_quantizer.name:
        self.output.max_val_po2 = weight_quantizer.max_val_po2
      else:
        self.output.max_val_po2 = input_quantizer.max_val_po2

      self.output.int_bits = self.output.bits

  @staticmethod
  def implemented_as():
    return "mux"


class XorGate(IMultiplier):
  """Use XorGate for hardware implementation of a multiplier."""

  def __init__(self, weight_quantizer: quantizer_impl.IQuantizer,
               input_quantizer: quantizer_impl.IQuantizer,
               output_quantizer: quantizer_impl.IQuantizer):
    super().__init__(weight_quantizer, input_quantizer, output_quantizer)

    if output_quantizer.name != "ternary":
      self.output.bits = max(self.input.bits, self.weights.bits)
      self.output.int_bits = max(self.input.int_bits, self.weights.int_bits)
      self.output.is_signed = self.input.is_signed | self.weights.is_signed
      assert_neither_input_and_weights_is_floating_point(self)
      self.output.is_floating_point = False

    self.gate_factor = 0.3
    self.gate_bits = 1

  @staticmethod
  def implemented_as():
    return "xor"


class Shifter(IMultiplier):
  """shifter gate.

  po2*qbit is implemented as a shifter. output is qbits type.

  determin number of bits in the output qbits type:
    1. min_exp in po2: number of bits to be expanded on the
        right (decimal bits) in qbits
        for example, min_exp = -2 -> po2 =2^min_exp = 2^(-2) :
        this means, po2*qbit -> qbit value right shifted for 2 bits
    2. max_exp in po2: number of bits to be expanded on
        the left (int_bits) in qbits

  How to calculate min_exp and max_exp:
    1.if po2 is_signed (quantized_po2)
      *one bit for sign for the entire po2 value;
      *exp has non_sign_bits = bits - 1 number of bits,
      *furthermore, 1 bit from non_sign_bits is used as sign bit in exp;
      *value range for exp is [-2 ** (non_sign_bits - 1),
       2 ** (non_sign_bits - 1) - 1]
    2.if not_signed (quantized_relu_po2)
      * 0 bit for the entire po2 value
      * exp has non_sign_bits = bits
      * rest is the same as above

  determine sign bit in the output qbits:
    1. qbits no_sign and po2 is_sign: since max_exp and min_exp
        are computed without sign bit
       we need to add 1 sign bit to the final result;
    2. qbits is_sign: since qbits already has a sign bit,
        no extra sign bit needed
    3. qbits no_sign and po2 no_sign: no extra sign bit needed

  Attributes:
    input: input_quantizer
    weight: weight_quantizer
    output: output_quantizer
    gate_factor: relative energy comparing to an Adder
    gate_bits: number of bits for energy calculation.
  """

  def __init__(
      self, weight_quantizer: quantizer_impl.IQuantizer,
      input_quantizer: quantizer_impl.IQuantizer,
      output_quantizer: quantizer_impl.IQuantizer
  ):
    super().__init__(weight_quantizer, input_quantizer, output_quantizer)

    # locate the po2 quantizer
    mode_w = weight_quantizer.mode
    if mode_w == 1:
      po2_quantizer = weight_quantizer
      qbit_quantizer = input_quantizer
    else:
      po2_quantizer = input_quantizer
      qbit_quantizer = weight_quantizer

    # find min_exp and max_exp of po2 quantizer
    (min_exp, max_exp) = po2_quantizer.get_min_max_exp()
    qbits_bits = qbit_quantizer.bits
    qbits_int_bits = qbit_quantizer.int_bits

    self.output.bits = int(qbits_bits + max_exp + min_exp)
    if (not qbit_quantizer.is_signed) and po2_quantizer.is_signed:
      # if qbit is signed, qbits_bits already has the sign_bit, no need to +1
      # if qbit is un_signed, po2 is unsigned, no need to +1
      # if qbit is un_signed, po2 is signed, min_exp and max_exp
      # didnot include sign_bit,
      # therefore need to +1
      self.output.bits += 1

    self.output.int_bits = int(qbits_int_bits + max_exp)
    self.output.is_signed = qbit_quantizer.is_signed | po2_quantizer.is_signed

    assert_neither_input_and_weights_is_floating_point(self)
    self.output.is_floating_point = False

    if po2_quantizer.inference_value_counts > 0:
      self.gate_factor = po2_quantizer.inference_value_counts * 0.3
      self.gate_bits = qbits_bits
    else:
      self.gate_factor = 1
      b = np.sqrt(2 ** po2_quantizer.bits * qbits_bits)
      self.gate_bits = b * np.log10(b)

  @staticmethod
  def implemented_as():
    return "shifter"


class AndGate(IMultiplier):
  """and gate implementation."""

  # binary(0,1) * any_datatype
  def __init__(
      self, weight_quantizer: quantizer_impl.IQuantizer,
      input_quantizer: quantizer_impl.IQuantizer,
      output_quantizer: quantizer_impl.IQuantizer
  ):
    super().__init__(weight_quantizer, input_quantizer, output_quantizer)

    # if output is ternary, no need for further computation
    if self.output.name != "ternary":
      self.output.bits = max(self.input.bits, self.weights.bits)

      self.output.is_signed = self.input.is_signed | self.weights.is_signed
      self.output.is_floating_point = self.input.is_floating_point |\
                                      self.weights.is_floating_point

      if weight_quantizer.name == "binary" and weight_quantizer.use_01:
        # binary(0,1) * datatype -> int_bits = datatype.int_bits
        self.output.int_bits = input_quantizer.int_bits
      else:
        self.output.int_bits = weight_quantizer.int_bits

      if "po2" in output_quantizer.name:
        # binary * po2
        if self.output.is_signed:
          output_quantizer.name = "quantized_po2"
        else:
          output_quantizer.name = "quantized_relu_po2"

        if "po2" in weight_quantizer.name:
          self.output.max_val_po2 = weight_quantizer.max_val_po2
        else:
          self.output.max_val_po2 = input_quantizer.max_val_po2

    self.gate_bits = self.output.bits
    self.gate_factor = 0.1

  @staticmethod
  def implemented_as():
    return "and"


class Adder(IMultiplier):
  """adder implementation."""

  def __init__(self, weight_quantizer: quantizer_impl.IQuantizer,
               input_quantizer: quantizer_impl.IQuantizer,
               output_quantizer: quantizer_impl.IQuantizer):
    super().__init__(weight_quantizer, input_quantizer,
                     output_quantizer)
    self.output.bits = max(self.input.bits, self.weights.bits) + 1
    self.output.int_bits = max(self.input.int_bits,
                               self.weights.int_bits) + 1
    self.output.is_signed = self.input.is_signed | self.weights.is_signed
    assert_neither_input_and_weights_is_floating_point(self)
    self.output.is_floating_point = False
    self.output.is_po2 = 1

    if self.input.max_val_po2 == -1 or self.weights.max_val_po2 == -1:
      self.output.max_val_po2 = -1
    else:
      # Adder is two po2_value multiply with each other
      self.output.max_val_po2 = self.input.max_val_po2 * self.weights.max_val_po2

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


class FloatingPointMultiplier(IMultiplier):
  """multiplier for floating point."""

  def __init__(self, weight_quantizer: quantizer_impl.IQuantizer,
               input_quantizer: quantizer_impl.IQuantizer,
               output_quantizer: quantizer_impl.IQuantizer):
    super().__init__(weight_quantizer, input_quantizer,
                     output_quantizer)

    self.output.bits = max(
        self.input.bits * self.input.is_floating_point,
        self.weights.bits * self.weights.is_floating_point,
    )
    self.output.int_bits = -1
    self.output.is_signed = 1

    assert self.input.is_floating_point | self.weights.is_floating_point
    self.output.is_floating_point = True

    self.gate_factor = 1
    self.gate_bits = self.output.bits

  @staticmethod
  def implemented_as():
    return "mul"


class FixedPointMultiplier(IMultiplier):
  """multiplier for fixed point."""

  def __init__(self, weight_quantizer: quantizer_impl.IQuantizer,
               input_quantizer: quantizer_impl.IQuantizer,
               output_quantizer: quantizer_impl.IQuantizer):
    super().__init__(weight_quantizer, input_quantizer,
                     output_quantizer)

    self.output.bits = self.input.bits + self.weights.bits
    self.output.int_bits = self.input.int_bits + self.weights.int_bits
    self.output.is_signed = self.input.is_signed | self.weights.is_signed

    assert_neither_input_and_weights_is_floating_point(self)
    self.output.is_floating_point = False

    self.gate_factor = 1
    self.gate_bits = np.sqrt(self.input.bits * self.weights.bits)

  @staticmethod
  def implemented_as():
    return "mul"
