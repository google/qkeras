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
  x = Activation("relu", name='relu_act')(x)
  x = Conv2D(32, (3, 3), activation="relu")(x)
  x = Activation("softmax")(x)
  x = QConv2D(32, (3, 3), activation="quantized_relu(4)")(x)
  return Model(inputs=xi, outputs=x)

def create_network_with_bn():
  xi = Input((28,28,1))
  x = Conv2D(32, (3, 3))(xi)
  x = BatchNormalization(axis=-1)(x)
  x = Activation("relu", name='relu_act')(x)
  x = Conv2D(32, (3, 3), activation="relu")(x)
  x = Activation("softmax")(x)
  x = DepthwiseConv2D((3, 3))(x)
  x = BatchNormalization(axis=-1)(x)
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


def test_conversion_from_relu_activation_to_qadaptiveactivation():
  m = create_network()
  d = {
      "QConv2D": {
          "kernel_quantizer": "binary",
          "bias_quantizer": "binary"
      },
      "QAdaptiveActivation": {
          "relu": "quantized_relu(8)"
      }
  }
  qq = model_quantize(m, d, 4)
  assert qq.layers[2].__class__.__name__ == "QAdaptiveActivation"
  assert str(qq.layers[2].quantizer).startswith("quantized_relu(8,")
  assert qq.layers[4].__class__.__name__ == "Activation"


def test_conversion_qadaptiveactivation_with_preference():
  m = create_network()
  d = {
      "QConv2D": {
          "kernel_quantizer": "binary",
          "bias_quantizer": "binary"
      },
      "relu_act": {
          "relu": "quantized_relu(8)"
      }
  }

  # Test with QActivation preference
  qq1 = model_quantize(m, d, 4, prefer_qadaptiveactivation=False)
  assert qq1.layers[2].__class__.__name__ == "QActivation"
  assert str(qq1.layers[2].quantizer).startswith("quantized_relu(8,")
  assert qq1.layers[4].__class__.__name__ == "Activation"

  # Test with QAdaptiveActivation preference
  qq2 = model_quantize(m, d, 4, prefer_qadaptiveactivation=True)
  assert qq2.layers[2].__class__.__name__ == "QAdaptiveActivation"
  assert str(qq2.layers[2].quantizer).startswith("quantized_relu(8,")
  assert qq2.layers[4].__class__.__name__ == "Activation"


def test_sequential_model_conversion():
  m = create_network_sequential()
  d = {
      "QConv2D": {
          "kernel_quantizer": "binary",
          "bias_quantizer": "binary"
      }}
  qq = model_quantize(m, d, 4)
  assert str(qq.layers[2].activation) == "quantized_relu(4,0)"


def test_folded_layer_conversion():
  # create a sequential model with conv2d layer and activation layers
  m1 = create_network()

  # create a sequantial model with conv2d layer followed by bn layer
  m2 = create_network_with_bn()

  # quantization config
  d = {
      "QConv2D": {
          "kernel_quantizer": "binary",
          "bias_quantizer": "binary"
      },
      "QDepthwiseConv2D": {
          "depthwise_quantizer": "binary",
          "bias_quantizer": "binary"
      },
      "QConv2DBatchnorm": {
          "kernel_quantizer": "ternary",
          "bias_quantizer": "ternary",
      },
      "QDepthwiseConv2DBatchnorm": {
          "depthwise_quantizer": "ternary",
          "bias_quantizer": "ternary",
      },
      "relu_act": {
          "relu": "quantized_relu(8)"
      }
  }

  # test when model has no layer to fold
  # desired behavior: un-folded layers
  qq1 = model_quantize(m1, d, 4, enable_bn_folding=True)
  assert qq1.layers[1].__class__.__name__ == "QConv2D"
  assert str(qq1.layers[1].quantizers[0]).startswith("binary")

  # test when the 1st conv2d layers needs to fold but the 2nd conv2d layer
  # does not (not followed by bn layer)
  # desired behavior: 1st conv2d is folded, 2nd conv2d unfolded
  # also test the depthwiseconv2d layer should fold
  qq2 = model_quantize(m2, d, 4, enable_bn_folding=True)
  assert qq2.layers[1].__class__.__name__ == "QConv2DBatchnorm"
  assert str(qq2.layers[1].quantizers[0]).startswith("ternary")
  assert qq2.layers[3].__class__.__name__ == "QConv2D"
  assert str(qq2.layers[3].quantizers[0]).startswith("binary")
  assert qq2.layers[5].__class__.__name__ == "QDepthwiseConv2DBatchnorm"
  assert str(qq2.layers[5].quantizers[0]).startswith("ternary")

  # test when there are layers to fold but folding is disabled
  # desired behavior: all conv2d/depthwise2d layers are not folded
  qq3 = model_quantize(m2, d, 4, enable_bn_folding=False)
  assert qq3.layers[1].__class__.__name__ == "QConv2D"
  assert str(qq3.layers[1].quantizers[0]).startswith("binary")
  assert qq3.layers[2].__class__.__name__ == "BatchNormalization"
  assert str(qq3.layers[3].quantizer).startswith("quantized_relu")
  assert qq3.layers[6].__class__.__name__ == "QDepthwiseConv2D"
  assert str(qq3.layers[6].quantizers[0]).startswith("binary")

  # test when QConv2DBatchnorm quantizer, e.g., is not given in config
  # desired behavior: quantizers for QConv2DBatchnorm layer fall back to QConv2D
  #   quantizers
  d = {
      "QConv2D": {
          "kernel_quantizer": "binary",
          "bias_quantizer": "binary"
      },
      "QDepthwiseConv2D": {
          "depthwise_quantizer": "binary",
          "bias_quantizer": "binary"
      },
      "relu_act": {
          "relu": "quantized_relu(8)"
      }
  }
  qq4 = model_quantize(m2, d, 4, enable_bn_folding=True)
  assert qq4.layers[1].__class__.__name__ == "QConv2DBatchnorm"
  assert str(qq4.layers[1].quantizers[0]).startswith("binary")
  assert qq4.layers[3].__class__.__name__ == "QConv2D"
  assert str(qq4.layers[3].quantizers[0]).startswith("binary")
  assert qq4.layers[5].__class__.__name__ == "QDepthwiseConv2DBatchnorm"
  assert str(qq4.layers[5].quantizers[0]).startswith("binary")


if __name__ == "__main__":
  pytest.main([__file__])
