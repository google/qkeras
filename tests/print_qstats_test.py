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
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from qkeras.estimate import print_qstats
from qkeras.utils import model_quantize
from qkeras import QConv2D
from qkeras.quantizers import *


def create_network():
  xi = Input((28, 28, 1))
  x = Conv2D(32, (3, 3))(xi)
  x = Activation("relu")(x)
  x = Conv2D(32, (3, 3), activation="relu")(x)
  x = Activation("softmax")(x)
  return Model(inputs=xi, outputs=x)


def create_mix_network():

  xi = Input((28, 28, 1))
  x = QConv2D(32, (3, 3), kernel_quantizer=binary())(xi)
  x = Activation("relu")(x)
  x = Conv2D(32, (3, 3))(x)
  x = Activation("softmax")(x)
  return Model(inputs=xi, outputs=x)


def create_network_with_bn():
  """Creates a network contains both QConv2D and QDepthwiseConv2D layers."""

  xi = Input((28, 28, 1))
  x = Conv2D(32, (3, 3))(xi)
  x = BatchNormalization()(x)
  x = Activation("relu")(x)
  x = DepthwiseConv2D((3, 3), activation="relu")(x)
  x = BatchNormalization()(x)
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

  # test if print_qstats works with unquantized layers
  print_qstats(m)

  # test if print_qstats works with mixture of quantized and unquantized layers
  m1 = create_mix_network()
  print_qstats(m1)

  m2 = create_network_with_bn()
  d2 = {
      "QConv2D": {
          "kernel_quantizer": "binary",
          "bias_quantizer": "binary"
      },
      "QActivation": {
          "relu": "ternary"
      },
      "QConv2DBatchnorm": {
          "kernel_quantizer": "ternary",
          "bias_quantizer": "ternary",
      },
      "QDepthwiseConv2DBatchnorm": {
          "depthwise_quantizer": "ternary",
          "bias_quantizer": "ternary",
      },
  }
  m2 = model_quantize(m2, d2, 4, enable_bn_folding=True)
  m2.summary()
  print_qstats(m2)


if __name__ == "__main__":
  pytest.main([__file__])
