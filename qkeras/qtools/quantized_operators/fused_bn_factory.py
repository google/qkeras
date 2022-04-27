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
"""quantized batch normliaztion quantizer implementation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np
import copy
from qkeras.qtools.quantized_operators import adder_factory
from qkeras.qtools.quantized_operators import divider_factory
from qkeras.qtools.quantized_operators import multiplier_factory
from qkeras.qtools.quantized_operators import quantizer_impl
from qkeras.qtools import qtools_util
from qkeras import quantizers

class FusedBNFactory:
  """determine which quantizer implementation to use.

  Create an fused bn instance. The type and bit width of the output_quantizer
  is deteremined from both the previous layer and batchnorm weight types:

  z = bn(y) = bn_inv * x - fused_bias is the output of the previous
  layer and the following bn layer, with:
    bn_inv = gamma * rsqrt(variance^2+epsilon) is computed from the
      bn layer weights with inverse_quantizer datatype
    x is the previous layer's output
    fused_bias = bn_inv * bias + beta - bn_inv*mean where bias is
      the bias term from the previous layer, beta and mean are the bn
      layer weights.
  """

  def make_quantizer(
      self, prev_output_quantizer: quantizer_impl.IQuantizer,
      beta_quantizer: quantizer_impl.IQuantizer,
      mean_quantizer: quantizer_impl.IQuantizer,
      inverse_quantizer: quantizer_impl.IQuantizer,
      prev_bias_quantizer: quantizer_impl.IQuantizer,
      use_beta: bool,
      use_bias: bool,
      qkeras_inverse_quantizer:quantizers.BaseQuantizer
  ):
    """Makes a fused_bn quantizer.

    Args:
      prev_output_quantizer: IQuantizer type. Previous layer output quantizer
      beta_quantizer: IQuantizer type. bn layer beta quantizer
      mean_quantizer: IQuantizer type.  layer mean quantizer
      inverse_quantizer: IQuantizer type. bn layer inverse quantizer
      prev_bias_quantizer: IQuantizer type. conv layer bias quantizer
      use_beta: Bool. whether enabling beta in batch_normalization layer
      use_bias: Bool. Whether bias is used in conv layer.
      qkeras_inverse_quantizer: QKeras quantizer type. bn layer inverse
        quantizer with QKeras quantizer type
    Returns:
      None
    """

    assert not isinstance(inverse_quantizer, quantizer_impl.FloatingPoint), (
        "inverse_quantizer in batchnorm layer has to be set for "
        "fused bn inference in hardware!")

    # bn_inv * x
    multiplier_instance = multiplier_factory.MultiplierFactory()
    multiplier_x = multiplier_instance.make_multiplier(
        inverse_quantizer, prev_output_quantizer)

    qtools_util.adjust_multiplier_for_auto_po2(
        multiplier_x, qkeras_inverse_quantizer)

    # fused_bias = bn_inv * bias + beta - bn_inv*mean
    # This step derives the datatype for bn_inv * mean
    multiplier_mean = multiplier_instance.make_multiplier(
        inverse_quantizer, mean_quantizer)

    qtools_util.adjust_multiplier_for_auto_po2(
        multiplier_mean, qkeras_inverse_quantizer)

    adder_instance = adder_factory.IAdder()
    if use_bias:
      # Derives datatype of bn_inv*bias
      multiplier_bias = multiplier_instance.make_multiplier(
          inverse_quantizer, prev_bias_quantizer)

      qtools_util.adjust_multiplier_for_auto_po2(
          multiplier_bias, qkeras_inverse_quantizer)

      # Derives datatype of bn_inv*bias - bn_inv*mean
      adder_1 = adder_instance.make_quantizer(
          multiplier_bias.output, multiplier_mean.output)
    else:
      # There is no bias from the previous layer,
      # therefore datatype of bn_inv*bias - bn_inv*mean is the same
      # as bn_inv*mean
      adder_1 = multiplier_mean

    if use_beta:
      # Derives datatype of fused_bias = bn_inv * bias + beta - bn_inv*mean
      adder_bias = adder_instance.make_quantizer(
          adder_1.output, beta_quantizer)
    else:
      # Since beta is not used, fused_bias = bn_inv * bias - bn_inv*mean
      adder_bias = adder_1

    # bn_inv * x - fused_bias
    adder = adder_instance.make_quantizer(
        multiplier_x.output, adder_bias.output)
    self.internal_accumulator = adder
    self.internal_output = adder
