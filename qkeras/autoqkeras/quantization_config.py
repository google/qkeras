# ==============================================================================
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
"""Definition of default quantization configuration."""

default_quantization_config = {
    "kernel": {
        "binary": 1,
        "stochastic_binary": 1,
        "ternary": 2,
        "stochastic_ternary": 2,
        "quantized_bits(2,1,1,alpha=1.0)": 2,
        "quantized_bits(4,0,1)": 4,
        "quantized_bits(8,0,1)": 8,
        "quantized_po2(4,1)": 4
    },
    "bias": {
        "quantized_bits(4,0,1)": 4,
        "quantized_bits(8,3,1)": 8,
        "quantized_po2(4,8)": 4
    },
    "activation": {
        "binary": 1,
        "binary(alpha='auto_po2')": 1,
        "ternary": 2,
        "quantized_relu(3,1)": 3,
        "quantized_relu(4,2)": 4,
        "quantized_relu(8,2)": 8,
        "quantized_relu(8,4)": 8,
        "quantized_relu(16,8)": 16,
        "quantized_relu_po2(4,4)": 4
    },
    "linear": {
        "binary": 1,
        "ternary": 2,
        "quantized_bits(4,1)": 4,
        "quantized_bits(8,2)": 8,
        "quantized_bits(16,10)": 16,
        "quantized_po2(6,4)": 6
    }
}
