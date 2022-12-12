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
"""Exports experimental quantizers."""

import tensorflow as tf

from qkeras.experimental.quantizers.quantizers_po2 import quantized_bits_learnable_po2  
from qkeras.experimental.quantizers.quantizers_po2 import quantized_bits_msqe_po2  

__version__ = "0.9.0"
