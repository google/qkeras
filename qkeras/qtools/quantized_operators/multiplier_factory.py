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
"""Create multiplier quantizer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
from qkeras.qtools.quantized_operators import multiplier_impl
from qkeras.qtools.quantized_operators import quantizer_impl


class MultiplierFactory:
  """determine which multiplier implementation to use."""

  def __init__(self):
    # the table below is found in this slides:
    # https://docs.google.com/presentation/d/1pcmoB6ZpX0IqjhSwgzO-oQwpMRYwIcDe/edit#slide=id.p40
    # also attached the output datatype in the table
    self.multiplier_impl_table = [
        [
            (
                multiplier_impl.FixedPointMultiplier,
                quantizer_impl.QuantizedBits()
            ),
            (multiplier_impl.Shifter, quantizer_impl.QuantizedBits()),
            (multiplier_impl.Mux, quantizer_impl.QuantizedBits()),
            (multiplier_impl.Mux, quantizer_impl.QuantizedBits()),
            (multiplier_impl.AndGate, quantizer_impl.QuantizedBits()),
            (
                multiplier_impl.FloatingPointMultiplier,
                quantizer_impl.FloatingPoint(
                    bits=None)
            )
        ],
        [
            (multiplier_impl.Shifter, quantizer_impl.QuantizedBits()),
            (multiplier_impl.Adder, quantizer_impl.PowerOfTwo()),
            (multiplier_impl.Mux, quantizer_impl.PowerOfTwo()),
            (multiplier_impl.Mux, quantizer_impl.PowerOfTwo()),
            (multiplier_impl.AndGate, quantizer_impl.PowerOfTwo()),
            (multiplier_impl.FloatingPointMultiplier,
             quantizer_impl.FloatingPoint(bits=None)
            )
        ],
        [
            (multiplier_impl.Mux, quantizer_impl.QuantizedBits()),
            (multiplier_impl.Mux, quantizer_impl.PowerOfTwo()),
            (multiplier_impl.Mux, quantizer_impl.Ternary()),
            (multiplier_impl.Mux, quantizer_impl.Ternary()),
            (multiplier_impl.AndGate, quantizer_impl.Ternary()),
            (multiplier_impl.FloatingPointMultiplier,
             quantizer_impl.FloatingPoint(bits=None))
        ],
        [
            (multiplier_impl.Mux, quantizer_impl.QuantizedBits()),
            (multiplier_impl.Mux, quantizer_impl.PowerOfTwo()),
            (multiplier_impl.Mux, quantizer_impl.Ternary()),
            (multiplier_impl.XorGate, quantizer_impl.Binary(
                use_01=False)),
            (multiplier_impl.AndGate, quantizer_impl.Ternary()),
            (multiplier_impl.FloatingPointMultiplier,
             quantizer_impl.FloatingPoint(bits=None))
        ],
        [
            (multiplier_impl.AndGate, quantizer_impl.QuantizedBits()),
            (multiplier_impl.AndGate, quantizer_impl.PowerOfTwo()),
            (multiplier_impl.AndGate, quantizer_impl.Ternary()),
            (multiplier_impl.AndGate, quantizer_impl.Ternary()),
            (multiplier_impl.AndGate, quantizer_impl.Binary(
                use_01=True)),
            (multiplier_impl.FloatingPointMultiplier,
             quantizer_impl.FloatingPoint(bits=None))
        ],
        [
            (
                multiplier_impl.FloatingPointMultiplier,
                quantizer_impl.FloatingPoint(bits=None)
            ),
            (
                multiplier_impl.FloatingPointMultiplier,
                quantizer_impl.FloatingPoint(bits=None)
            ),
            (
                multiplier_impl.FloatingPointMultiplier,
                quantizer_impl.FloatingPoint(bits=None)
            ),
            (
                multiplier_impl.FloatingPointMultiplier,
                quantizer_impl.FloatingPoint(bits=None)
            ),
            (
                multiplier_impl.FloatingPointMultiplier,
                quantizer_impl.FloatingPoint(bits=None)
            ),
            (
                multiplier_impl.FloatingPointMultiplier,
                quantizer_impl.FloatingPoint(bits=None)
            )
        ]
    ]

  def make_multiplier(
      self, weight_quantizer: quantizer_impl.IQuantizer,
      input_quantizer: quantizer_impl.IQuantizer
  ) -> multiplier_impl.IMultiplier:
    """Create a multiplier instance.

    The type and bit width of the multiplier is deteremined from the
    quantizer type of both the kernel (weight) and input tensor.

    The table below illustrates the rule of inferring multiplier type from the
    quantizer type of both the kernel (weight) and input tensor

                                        x
                      qb(n)   +/-,exp  t(-1,0,+1) b(-1,+1) b(0,1) float32
        qb(n)            *     << >>,-     ?,-       ?,-       ?
        +/-,exp        << >>,-   +         ?,-        ^      ?,-
      w t(-1,0,+1)      ?,-     ?,-        ?,^       ?,^      ^
        b(-1,+1)        ?,-      ^         ?,^        ^       ^
        b(0,1)           ?      ?,-         ^         ^       ^      &
        float32

    Args:
      weight_quantizer: weight quantizer type
      input_quantizer: input quantizer type

    Returns:
      An IMultiplier instance.
    """

    assert weight_quantizer is not None
    assert input_quantizer is not None

    (multiplier_impl_class, output_quantizer) = self.multiplier_impl_table[
        weight_quantizer.mode][input_quantizer.mode]

    logging.debug(
        "multiplier implemented as class %s",
        multiplier_impl_class.implemented_as())

    assert issubclass(multiplier_impl_class, multiplier_impl.IMultiplier)

    return multiplier_impl_class(
        weight_quantizer,
        input_quantizer,
        output_quantizer
    )
