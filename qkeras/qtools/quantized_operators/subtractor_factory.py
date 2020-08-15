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
""""create subtractor quantizer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from qkeras.qtools.quantized_operators import adder_factory
from qkeras.qtools.quantized_operators import adder_impl
from qkeras.qtools.quantized_operators import quantizer_impl


class ISubtractor(adder_factory.IAdder):
  """Create a subtractor instance.

  The methods in subtractor is mostly inherited from adder
  with a few exceptions.
  """

  def make_quantizer(self, quantizer_1: quantizer_impl.IQuantizer,
                     quantizer_2: quantizer_impl.IQuantizer):
    """make an ISubtractor instance.

    if quantizer1 and quantizer2 are both non-signed, result should change
    to signed; else since a sign bit is already present,
    no need to add extra sign bit

    Args:
      quantizer_1: first operand
      quantizer_2: second operand

    Returns:
      An ISubtractor instance
    """
    quantizer = super().make_quantizer(quantizer_1, quantizer_2)

    if not isinstance(quantizer, adder_impl.FloatingPoint_Adder):
      if not quantizer_1.is_signed and not quantizer_2.is_signed:
        quantizer.output.is_signed = 1
        quantizer.output.bits += 1

    return quantizer
