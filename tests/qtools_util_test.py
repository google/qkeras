# Copyright 2020 Google LLC
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
"""Tests for qtools_util module."""

import json

import numpy as np
import pytest
import tensorflow.keras as keras
import tensorflow as tf

from qkeras import quantizers
from qkeras.qtools import qtools_util

from qkeras.qtools import quantized_operators
from qkeras.qtools.quantized_operators import quantizer_factory as quantizer_factory_module


@pytest.mark.parametrize(
    "w_bits, w_int_bits, weight_quantizer_scale_type, "
    "expected_bits_before_adjustment, expected_int_bits_before_adjustment, "
    "expected_bits_after_adjustment, expected_int_bits_after_adjustment",
    [
        (8, 0, "1.0", 11, 2, 11, 2),
        (4, 2, "auto_po2", 7, 4, 10, 5),
        (4, 0, "post_training_scale", 7, 2, 10, 5),
    ],
)
def test_adjust_multiplier_for_auto_po2(
    w_bits, w_int_bits, weight_quantizer_scale_type,
    expected_bits_before_adjustment, expected_int_bits_before_adjustment,
    expected_bits_after_adjustment, expected_int_bits_after_adjustment):
  """Test adjust_multiplier_for_auto_po2 with auto_po2 weight quantizer."""

  multiplier_factory = quantized_operators.MultiplierFactory()
  quantizer_factory = quantizer_factory_module.QuantizerFactory()

  qkeras_input_quantizer = quantizers.quantized_bits(4, 2, 1)

  # Generate the weight quantizer.
  if weight_quantizer_scale_type in ["auto_po2", "post_training_scale"]:
    # Compute the scale for auto_po2 quantizer.
    qkeras_weight_quantizer = quantizers.quantized_bits(
        bits=w_bits, integer=w_int_bits, keep_negative=True,
        symmetric=True, alpha="auto_po2")
    weight_arr = np.array([1.07, -1.7, 3.06, 1.93, 0.37, -2.43, 6.3, -2.9]
                          ).reshape((2, 4))
    qkeras_weight_quantizer(weight_arr)

    if weight_quantizer_scale_type == "post_training_scale":
      # Set the post_training_scale as fixed scale.
      auto_po2_scale = qkeras_weight_quantizer.scale.numpy()
      qkeras_weight_quantizer = quantizers.quantized_bits(
          bits=w_bits, integer=w_int_bits, alpha="auto_po2",
          post_training_scale=auto_po2_scale)
  else:
    qkeras_weight_quantizer = quantizers.quantized_bits(w_bits, w_int_bits)

  input_quantizer = quantizer_factory.make_quantizer(
      qkeras_input_quantizer)
  weight_quantizer = quantizer_factory.make_quantizer(
      qkeras_weight_quantizer)

  multiplier = multiplier_factory.make_multiplier(
      weight_quantizer, input_quantizer)

  np.testing.assert_equal(multiplier.output.bits,
                          expected_bits_before_adjustment)
  np.testing.assert_equal(multiplier.output.int_bits,
                          expected_int_bits_before_adjustment)

  qtools_util.adjust_multiplier_for_auto_po2(
      multiplier, qkeras_weight_quantizer)
  print(f"after adjustment: {multiplier.output.bits}, {multiplier.output.int_bits}")
  np.testing.assert_equal(multiplier.output.bits,
                          expected_bits_after_adjustment)
  np.testing.assert_equal(multiplier.output.int_bits,
                          expected_int_bits_after_adjustment)


if __name__ == "__main__":
  pytest.main([__file__])
