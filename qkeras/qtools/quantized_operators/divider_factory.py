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
""""create divider quantizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from absl import logging
from qkeras.qtools.quantized_operators import divider_impl
from qkeras.qtools.quantized_operators import quantizer_impl


class UnacceptedQuantizerError(ValueError):
  pass


class IDivider(abc.ABC):
  """abstract class for divider."""

  def __init__(self):
    # also attached the output datatype in the table
    self.divider_impl_table = [
        [
            # when qbits is denominator, use default bits for float result
            (divider_impl.FloatingPointDivider, quantizer_impl.FloatingPoint(
                bits=quantizer_impl.FLOATINGPOINT_BITS)),
            (divider_impl.Shifter, quantizer_impl.QuantizedBits()),
            (None, None),
            (None, None),
            (None, None),
            # when bits sets to None, will decide f16/f32 according
            # to input quantizer
            (divider_impl.FloatingPointDivider, quantizer_impl.FloatingPoint(
                bits=None))
        ],
        [
            (divider_impl.FloatingPointDivider, quantizer_impl.FloatingPoint(
                bits=quantizer_impl.FLOATINGPOINT_BITS)),
            (divider_impl.Subtractor, quantizer_impl.PowerOfTwo()),
            (None, None),
            (None, None),
            (None, None),
            (divider_impl.FloatingPointDivider, quantizer_impl.FloatingPoint(
                bits=None))
        ],
        [
            (divider_impl.FloatingPointDivider, quantizer_impl.FloatingPoint(
                bits=quantizer_impl.FLOATINGPOINT_BITS)),
            (divider_impl.Shifter, quantizer_impl.QuantizedBits()),
            (None, None),
            (None, None),
            (None, None),
            (divider_impl.FloatingPointDivider, quantizer_impl.FloatingPoint(
                bits=None))
        ],
        [
            (divider_impl.FloatingPointDivider, quantizer_impl.FloatingPoint(
                bits=quantizer_impl.FLOATINGPOINT_BITS)),
            (divider_impl.Shifter, quantizer_impl.PowerOfTwo()),
            (None, None),
            (None, None),
            (None, None),
            (divider_impl.FloatingPointDivider, quantizer_impl.FloatingPoint(
                bits=None))
        ],
        [
            (divider_impl.FloatingPointDivider, quantizer_impl.FloatingPoint(
                bits=quantizer_impl.FLOATINGPOINT_BITS)),
            (divider_impl.Shifter, quantizer_impl.PowerOfTwo()),
            (None, None),
            (None, None),
            (None, None),
            (divider_impl.FloatingPointDivider, quantizer_impl.FloatingPoint(
                bits=None))
        ],
        [
            (divider_impl.FloatingPointDivider, quantizer_impl.FloatingPoint(
                bits=None)),
            (divider_impl.FloatingPointDivider, quantizer_impl.FloatingPoint(
                bits=None)),
            (None, None),
            (None, None),
            (None, None),
            (divider_impl.FloatingPointDivider, quantizer_impl.FloatingPoint(
                bits=None))
        ]
    ]

  def make_quantizer(self, numerator_quantizer: quantizer_impl.IQuantizer,
                     denominator_quantizer: quantizer_impl.IQuantizer):
    """make the quantizer."""

    self.numerator_quantizer = numerator_quantizer
    self.denominator_quantizer = denominator_quantizer

    mode1 = numerator_quantizer.mode
    mode2 = denominator_quantizer.mode

    (divider_impl_class, output_quantizer) = self.divider_impl_table[
        mode1][mode2]

    if divider_impl_class is None:
      raise UnacceptedQuantizerError(
          "denominator quantizer {} not accepted!".format(
              denominator_quantizer.name))

    logging.debug(
        "qbn adder implemented as class %s",
        divider_impl_class.implemented_as())

    return divider_impl_class(
        numerator_quantizer,
        denominator_quantizer,
        output_quantizer
    )
