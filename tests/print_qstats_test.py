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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pytest
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from qkeras.estimate import print_qstats
from qkeras.utils import model_quantize


def create_network():
  xi = Input((28, 28, 1))
  x = Conv2D(32, (3, 3))(xi)
  x = Activation("relu")(x)
  x = Conv2D(32, (3, 3), activation="relu")(x)
  x = Activation("softmax")(x)
  return Model(inputs=xi, outputs=x)


def test_conversion_print_qstats():
  # this tests if references in tensorflow are working properly.
  m = create_network()
  d = {
      "QConv2D": {
          "kernel_quantizer": "binary",
          "bias_quantizer": "binary"
      },
      "QActivation": {
          "relu": "ternary"
      }
  }
  qq = model_quantize(m, d, 4)
  qq.summary()
  print_qstats(qq)


if __name__ == "__main__":
  pytest.main([__file__])
