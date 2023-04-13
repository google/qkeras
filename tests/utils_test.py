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
"""Tests for methods in utils.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from qkeras import *
from qkeras.utils import get_model_sparsity
from qkeras.utils import model_quantize
from qkeras.utils import convert_to_folded_model
from qkeras.utils import is_TFOpLambda_layer
from qkeras.utils import find_bn_fusing_layer_pair
from qkeras.utils import add_bn_fusing_weights


def create_quantized_network():
  """Creates a simple quantized conv net model."""
  # Create a simple model
  xi = Input((28, 28, 1))
  x = Conv2D(32, (3, 3))(xi)
  x = Activation("relu")(x)
  x = Conv2D(32, (3, 3), activation="relu")(x)
  x = Activation("softmax")(x)
  model = Model(inputs=xi, outputs=x)

  # Quantize the model
  quantizer_config = {
      "QConv2D": {
          "kernel_quantizer": "quantized_bits(4)",
          "bias_quantizer": "quantized_bits(4)"
      },
      "QActivation": {
          "relu": "ternary"
      }
  }
  activation_bits = 4
  qmodel = model_quantize(model, quantizer_config, activation_bits)
  return qmodel


def create_quantized_po2_network():
  """Creates a simple quantized conv net model with po2 quantizers."""
  xi = Input((28, 28, 1))
  x = QConv2D(32, (3, 3), kernel_quantizer=quantized_po2(4))(xi)
  x = QActivation(quantized_bits(8))(x)
  x = QConv2D(32, (3, 3), kernel_quantizer=quantized_po2(4))(x)
  x = QActivation(quantized_bits(8))(x)
  qmodel = Model(xi, x, name='simple_po2_qmodel')
  return qmodel


def set_network_sparsity(model, sparsity):
  """Set the sparsity of the given model using random weights."""

  for layer in model.layers:
    new_weights = []
    for w in layer.get_weights():
      # Create weights with desired sparsity
      sparse_weights = np.random.rand(w.size)+0.1
      sparse_weights[:int(w.size*sparsity)] = 0
      np.random.shuffle(sparse_weights)
      new_weights.append(sparse_weights.reshape(w.shape))
    layer.set_weights(new_weights)
  return model


def test_get_model_sparsity():
  """Tests if the method get_model_sparsity in utils.py works correctly."""
  qmodel = create_quantized_network()

  # Generate sparsity levels to test
  sparsity_levels = np.concatenate((np.random.rand(10), [1.0, 0.0])).round(2)

  # Test various sparsity levels
  for true_sparsity in sparsity_levels:
    qmodel = set_network_sparsity(qmodel, true_sparsity)
    calc_sparsity = get_model_sparsity(qmodel)
    assert np.abs(calc_sparsity - true_sparsity) < 0.01


def test_get_po2_model_sparsity():
  """Tests get_model_sparsity on a po2-quantized model.

  Models quantized with po2 quantizers should have a sparsity near 0 because
  if the exponent is set to 0, the value of the weight will equal 2^0 == 1 != 0
  """
  qmodel = create_quantized_po2_network()
  qmodel.use_legacy_config = True

  # Generate sparsity levels to test
  sparsity_levels = np.concatenate((np.random.rand(10), [1.0, 0.0])).round(2)

  # Test various sparsity levels
  for set_sparsity in sparsity_levels:
    qmodel = set_network_sparsity(qmodel, set_sparsity)
    calc_sparsity = get_model_sparsity(qmodel)
    assert np.abs(calc_sparsity - 0) < 0.01


def test_convert_to_folded_model():
  """Test convert_to_folded_model to work properly on non-sequential model."""

  def get_add_model():
    x = x_in = Input(shape=(4, 4, 1), name="input")
    x1 = Conv2D(4, kernel_size=(2, 2), padding="valid", strides=(1, 1),
                name="conv2d_1")(x)
    x1 = BatchNormalization(name="bn_1")(x1)
    x1 = Activation("relu", name="relu_1")(x1)
    x2 = Conv2D(4, kernel_size=(2, 2), padding="valid", strides=(1, 1),
                name="conv2d_2")(x)
    x2 = BatchNormalization(name="bn_2")(x2)
    x2 = Activation("relu", name="relu_2")(x2)
    x = Add(name="add")([x1, x2])
    x = Softmax()(x)

    return Model(inputs=[x_in], outputs=[x])

  model = get_add_model()

  fmodel, _ = convert_to_folded_model(model)

  assert fmodel.layers[5].name == "add"

  # test if convert_to_folded_model work with TFOpLambda layers
  def hard_sigmoid(x):
    return ReLU(6.)(x + 3.) * (1. / 6.)

  def hard_swish(x):
    return Multiply()([hard_sigmoid(x), x])

  def get_lambda_model():
    x = x_in = Input(shape=(4, 4, 1), name="input")
    x = Conv2D(
        4, kernel_size=(2, 2), padding="valid", strides=(1, 1),
        name="conv2d_1")(x)
    x = hard_swish(x)

    return Model(inputs=[x_in], outputs=[x])

  model = get_lambda_model()
  fmodel, _ = convert_to_folded_model(model)

  assert is_TFOpLambda_layer(model.layers[2])
  assert is_TFOpLambda_layer(model.layers[4])
  assert isinstance(fmodel.layers[5], Multiply)


def test_find_bn_fusing_layer_pair():
  x = x_in = Input((23, 23, 1), name="input")
  x1 = QConv2D(
      2, 2, 1,
      kernel_quantizer=quantized_bits(4, 0, 1),
      bias_quantizer=quantized_bits(4, 0, 1),
      use_bias=False,
      name="conv1")(x)
  x1 = QBatchNormalization(
      mean_quantizer=quantized_bits(4, 0, 1),
      gamma_quantizer=None,
      variance_quantizer=None,
      beta_quantizer=quantized_bits(4, 0, 1),
      inverse_quantizer=quantized_bits(8, 0, 1), name="bn1")(x1)

  x2 = QConv2D(
      2, 2, 1,
      kernel_quantizer=quantized_bits(3, 0),
      bias_quantizer=quantized_bits(3, 2),
      name="conv2")(x)

  x2 = QBatchNormalization(
      mean_quantizer=quantized_bits(4, 0, 1),
      gamma_quantizer=None,
      variance_quantizer=None,
      beta_quantizer=quantized_bits(4, 0, 1),
      inverse_quantizer=quantized_bits(8, 0, 1), name="bn2")(x2)

  x = Add(name="add")([x1, x2])
  model = Model(inputs=[x_in], outputs=[x])

  (conv_bn_pair_dict, _) = find_bn_fusing_layer_pair(model)
  assert conv_bn_pair_dict["conv1"] == "bn1"
  assert conv_bn_pair_dict["conv2"] == "bn2"

  conv_layer = model.layers[1]
  bn_layer = model.layers[3]

  conv_layer.set_weights([
      np.array([[[[0.5, 0.75]], [[1.5, -0.625]]],
                [[[-0.875, 1.25]], [[-1.25, -2.5]]]])
  ])
  bn_layer.set_weights([
      np.array([1., 0.25]),
      np.array([0.5, 1.0]),
      np.array([0.5, 2.5]),
      np.array([1.5, 1.])
  ])
  saved_weights = {}
  saved_weights[conv_layer.name] = {}
  add_bn_fusing_weights(conv_layer, bn_layer, saved_weights)

  d = saved_weights[conv_layer.name]
  assert d["enable_bn_fusing"]
  assert d["fused_bn_layer_name"] == "bn1"
  assert np.all(d["bn_inv"] == np.array([0.8125, 0.25]))
  assert np.all(d["fused_bias"] == np.array([0.09375, 0.65625]))


if __name__ == "__main__":
  pytest.main([__file__])
