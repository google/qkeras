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
"""configuration file for external usage."""

config_settings = {
    "default_source_quantizer": "quantized_bits(8, 0, 1)",
    "default_interm_quantizer": "quantized_bits(8, 0, 1)",

    "horowitz": {
        "fpm_add": [0.003125, 0],
        "fpm_mul": [0.002994791667, 0.001041666667, 0],
        "fp16_add": [0.4],
        "fp16_mul": [1.1],
        "fp32_add": [0.9],
        "fp32_mul": [3.7],
        "sram_rd": [9.02427321e-04, -2.68847858e-02, 2.08900804e-01, 0.0],
        "dram_rd": [20.3125, 0]
    },

    "include_energy": {
        "QActivation": ["outputs"],
        "QAdaptiveActivation": ["outputs"],
        "Activation": ["outputs"],
        "QBatchNormalization": ["parameters"],
        "BatchNormalization": ["parameters"],
        "Add": ["op_cost"],
        "Subtract": ["op_cost"],
        "MaxPooling2D": ["op_cost"],
        "default": ["inputs", "parameters", "op_cost"]
    }
}
