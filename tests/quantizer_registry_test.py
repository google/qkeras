# Copyright 2024 Google LLC
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
"""Unit tests for QKeras quantizer registry."""

import numpy as np
import pytest

from qkeras import quantizer_registry
from qkeras import quantizers


@pytest.mark.parametrize(
    "quantizer_name",
    [
        "quantized_linear",
        "quantized_bits",
        "bernoulli",
        "ternary",
        "stochastic_ternary",
        "binary",
        "stochastic_binary",
        "quantized_relu",
        "quantized_ulaw",
        "quantized_tanh",
        "quantized_sigmoid",
        "quantized_po2",
        "quantized_relu_po2",
        "quantized_hswish",
    ],
)
def test_lookup(quantizer_name):
  quantizer = quantizer_registry.lookup_quantizer(quantizer_name)
  is_class_instance = isinstance(quantizer, type)
  np.testing.assert_equal(is_class_instance, True)


if __name__ == "__main__":
  pytest.main([__file__])
