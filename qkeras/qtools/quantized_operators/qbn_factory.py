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
"""quantized batch normliaztion quantizer implementation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np
from qkeras.qtools.quantized_operators import adder_factory
from qkeras.qtools.quantized_operators import divider_factory
from qkeras.qtools.quantized_operators import multiplier_factory
from qkeras.qtools.quantized_operators import quantizer_impl


class QBNFactory:
  """determine which quantizer implementation to use.

  Create an qbn instance. The type and bit width of the output_quantizer
  is deteremined from gamma, beta, mean and variance quantizer
  y = gamma * (x - mean)/stddev + beta
  """

  def make_quantizer(
      self, input_quantizer: quantizer_impl.IQuantizer,
      gamma_quantizer: quantizer_impl.IQuantizer,
      beta_quantizer: quantizer_impl.IQuantizer,
      mean_quantizer: quantizer_impl.IQuantizer,
      variance_quantizer: quantizer_impl.IQuantizer,
      use_scale,
      use_center
  ):
    """make a qbn quantizer."""

    self.input_quantizer = input_quantizer
    self.gamma_quantizer = gamma_quantizer
    self.beta_quantizer = beta_quantizer
    self.mean_quantizer = mean_quantizer
    self.variance_quantizer = variance_quantizer
    self.use_scale = use_scale
    self.use_center = use_center

    multiplier = None
    accumulator = None

    # convert variance po2 quantizer to stddev po2 quantizer
    stddev_quantizer = copy.deepcopy(variance_quantizer)
    if stddev_quantizer.is_po2:
      if variance_quantizer.max_val_po2 >= 0:
        stddev_quantizer.max_val_po2 = np.round(math.sqrt(
            variance_quantizer.max_val_po2))
      else:
        stddev_quantizer.max_val_po2 = variance_quantizer.max_val_po2

      stddev_quantizer.bits = variance_quantizer.bits - 1
      stddev_quantizer.int_bits = stddev_quantizer.bits

    divider_instance = divider_factory.IDivider()

    if use_scale:
      # gamma/var
      divider = divider_instance.make_quantizer(
          gamma_quantizer, stddev_quantizer)

      # update the actual number of values in divider quantizer during inference
      count = -1
      if gamma_quantizer.is_po2 and gamma_quantizer.inference_value_counts > 0:
        count = gamma_quantizer.inference_value_counts
        if stddev_quantizer.is_po2 and stddev_quantizer.inference_value_counts > 0:
          count *= stddev_quantizer.inference_value_counts
        else:
          count = -1
      if count > 0:
        divider.output.inference_value_counts = count

      # gamma/var * x
      multiplier_instance = multiplier_factory.MultiplierFactory()
      multiplier = multiplier_instance.make_multiplier(
          divider.output, input_quantizer)
      accumulator_input = multiplier

    else:
      # x/var
      divider = divider_instance.make_quantizer(
          input_quantizer, stddev_quantizer)
      accumulator_input = divider

    if use_center:
      # y = gamma/var * x + beta
      accumulator_instance = adder_factory.IAdder()
      accumulator = accumulator_instance.make_quantizer(
          accumulator_input.output, beta_quantizer)
      output_q = accumulator
    else:
      output_q = accumulator_input

    self.internal_divide_quantizer = divider
    self.internal_multiplier = multiplier
    self.internal_accumulator = accumulator
    self.internal_output = output_q
