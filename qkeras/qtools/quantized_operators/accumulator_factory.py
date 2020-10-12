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
"""Create accumulator quantizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from qkeras.qtools.quantized_operators import accumulator_impl
from qkeras.qtools.quantized_operators import multiplier_impl


class AccumulatorFactory:
  """interface for accumulator type."""

  def make_accumulator(
      self, kernel_shape,
      multiplier: multiplier_impl.IMultiplier,
      use_bias=True
  ) -> accumulator_impl.IAccumulator:
    """Create an accumulator instance."""

    # The type and bit width of the accumulator is deteremined from the
    # multiplier implementation, and the shape of both kernel and bias

    if multiplier.output.is_floating_point:
      accumulator = accumulator_impl.FloatingPointAccumulator(
          multiplier)

    # po2*po2 is implemented as Adder; output type is po2
    # in multiplier, po2 needs to be converted to FixedPoint
    elif multiplier.output.is_po2:
      accumulator = accumulator_impl.Po2Accumulator(
          kernel_shape, multiplier, use_bias)

    # fixed point
    else:
      accumulator = accumulator_impl.FixedPointAccumulator(
          kernel_shape, multiplier, use_bias)

    return accumulator
