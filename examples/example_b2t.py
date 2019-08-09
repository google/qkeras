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
"""Implements total/partial Binary to Thermometer decoder."""

import numpy as np
from qkeras import BinaryToThermometer

if __name__ == "__main__":
  np.random.seed(42)
  x = np.array(range(8))
  b = BinaryToThermometer(x, 2, 8)
  print(b)
  b = BinaryToThermometer(x, 2, 8, 1)
  print(b)
  b = BinaryToThermometer(x, 2, 8, 1, use_two_hot_encoding=1)
  print(b)
  b = BinaryToThermometer(x, 4, 8)
  print(b)
  b = BinaryToThermometer(x, 4, 8, 1)
  print(b)
  b = BinaryToThermometer(x, 4, 8, 1, use_two_hot_encoding=1)
  print(b)
  x = np.random.randint(0, 255, (100, 28, 28, 1))
  print(x[0, 0, 0:5])
  b = BinaryToThermometer(x, 8, 256, 0)
  print(x.shape, b.shape)
  print(b[0, 0, 0:5])
  b = BinaryToThermometer(x, 8, 256, 1)
  print(b[0, 0, 0:5])
  x = np.random.randint(0, 255, (100, 28, 28, 2))
  b = BinaryToThermometer(x, 8, 256, 0, 1)
  print(x.shape, b.shape)
  print(x[0, 0, 0, 0:2])
  print(b[0, 0, 0, 0:8])
  print(b[0, 0, 0, 8:16])
