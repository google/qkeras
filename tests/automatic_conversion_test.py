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
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from qkeras import *
from qkeras.utils import model_quantize


def create_network():
  xi = Input((28,28,1))
  x = Conv2D(32, (3, 3))(xi)
  x = Activation("relu")(x)
  x = Conv2D(32, (3, 3), activation="relu")(x)
  x = Activation("softmax")(x)
  x = QConv2D(32, (3, 3), activation="quantized_relu(4)")(x)
  return Model(inputs=xi, outputs=x)

def create_network_sequential():
  model = Sequential([
    Conv2D(32, (3, 3), input_shape=(28,28,1)),
    Activation('relu'),
    Conv2D(32, (3, 3), activation="relu"),
    Activation('softmax'),
    QConv2D(32, (3, 3), activation="quantized_relu(4)")
  ])
  return model

def test_linear_activation():
  m = create_network()

  assert m.layers[1].activation.__name__ == "linear", "test failed"


def test_linear_activation_conversion():
  m = create_network()

  d = {
      "QConv2D": {
          "kernel_quantizer": "binary",
          "bias_quantizer": "binary",
          "activation_quantizer": "binary"
      }
  }
  qq = model_quantize(m, d, 4)

  assert str(qq.layers[1].activation) == "binary()"


def test_no_activation_conversion_to_quantized():
  m = create_network()
  d = {"QConv2D": {"kernel_quantizer": "binary", "bias_quantizer": "binary"}}
  qq = model_quantize(m, d, 4)
  assert qq.layers[2].__class__.__name__ == "Activation"
  assert qq.layers[4].__class__.__name__ == "Activation"


def test_automatic_conversion_from_relu_to_qr():
  m = create_network()
  d = {
      "QConv2D": {
          "kernel_quantizer": "binary",
          "bias_quantizer": "binary"
      }}
  qq = model_quantize(m, d, 4)
  assert str(qq.layers[3].activation) == "quantized_relu(4,0)"


def test_conversion_from_relu_activation_to_qr_qactivation():
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
  assert qq.layers[2].__class__.__name__ == "QActivation"
  assert str(qq.layers[2].quantizer) == "ternary()"
  assert qq.layers[4].__class__.__name__ == "Activation"


def test_sequential_model_conversion():
  m = create_network_sequential()
  d = {
      "QConv2D": {
          "kernel_quantizer": "binary",
          "bias_quantizer": "binary"
      }}
  qq = model_quantize(m, d, 4)
  assert str(qq.layers[2].activation) == "quantized_relu(4,0)"

if __name__ == "__main__":
  pytest.main([__file__])
