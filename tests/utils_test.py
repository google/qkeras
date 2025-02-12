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
import os
import tempfile
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from qkeras import *
from qkeras.utils import get_model_sparsity
from qkeras.utils import model_quantize
from qkeras.utils import convert_to_folded_model
from qkeras.utils import is_TFOpLambda_layer
from qkeras.utils import find_bn_fusing_layer_pair
from qkeras.utils import add_bn_fusing_weights
from qkeras.utils import clone_model_and_freeze_auto_po2_scale
from qkeras.utils import load_qmodel


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


def create_test_model_for_scale_freezing(bias_quantizer):
  def _create_simple_model(bias_quantizer):
    x = x_in = tf.keras.Input((4, 4, 1), name="input")
    x = QConv2D(
        filters=4, kernel_size=2, strides=2,
        kernel_quantizer=quantized_bits(4, 2, 1, alpha="auto_po2"),
        bias_quantizer=quantized_bits(4, 2, 1),
        use_bias=False,
        name="conv")(x)
    x = QDepthwiseConv2D(
        kernel_size=2, strides=1,
        depthwise_quantizer=quantized_bits(6, 3, 1, alpha="auto_po2"),
        use_bias=False,
        bias_quantizer=quantized_bits(4, 2, 1),
        name="dw_conv")(x)
    x = QBatchNormalization(
        mean_quantizer=quantized_bits(4, 2, 1),
        gamma_quantizer=None,
        variance_quantizer=None,
        beta_quantizer=quantized_bits(4, 0, 1),
        inverse_quantizer=quantized_bits(8, 0, 1, alpha="auto_po2"),
        name="bn")(x)

    x = QActivation(activation=quantized_bits(4, 0), name="relu")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = QDense(units=2,
               kernel_quantizer=quantized_bits(4, 2, 1, alpha="auto_po2"),
               bias_quantizer=bias_quantizer, name="dense")(x)
    model = tf.keras.Model(inputs=x_in, outputs=x)

    return model

  def _set_weights(model):
    conv_w = [np.array(
        [0.23, 2.76, 0.1, 0.33, 0.53, 0.16, 0.3, 1.7, -0.9,
         1.43, 2.31, -0.2, -1.7, 0.39, -2.03, 1.79]).reshape(2, 2, 1, 4)]

    dw_conv_w = [np.array([
        0.03, 3.6, 2.1, 1.2, 0.13, 1.3, -0.3, 1.2, -0.7,
        -10.3, 11.7, -0.92, -10.7, 0.59, -1.93, 2.8]).reshape((2, 2, 4, 1))]

    bn_w = [np.array([0.28, 1.33, 2.27, 3.36]),
            np.array([0.31, 0.1, 0.03, 4.26]),
            np.array([0.89, -0.21, 1.97, 2.06]),
            np.array([1.2, 0.9, 13.2, 10.9])]

    dense_w = np.array(
        [0.13, 0.66, 0.21, 0.23, 1.07, -0.79, 1.83, 1.81])
    dense_w = [dense_w.reshape((4, 2)), np.array([-1.3, 0.7])]

    model.get_layer("conv").set_weights(conv_w)
    model.get_layer("dw_conv").set_weights(dw_conv_w)
    model.get_layer("bn").set_weights(bn_w)
    model.get_layer("dense").set_weights(dense_w)

  orig_model = _create_simple_model(bias_quantizer)
  _set_weights(orig_model)

  return orig_model


def test_clone_model_and_freeze_auto_po2_scale():
  """Test clone_model_and_freeze_auto_po2_scale to work properly."""

  orig_model = create_test_model_for_scale_freezing(quantized_bits(4, 2, 1))
  _, new_hw = clone_model_and_freeze_auto_po2_scale(
      orig_model, quantize_model_weights=True)

  # Check if the new model's weights and scales are derived properly.
  np.testing.assert_array_equal(
      new_hw["conv"]["weights"][0],
      np.array(
          [[[[0.5, 6, 0, 0.5]], [[1, 0, 0.5, 3.5]]],
           [[[-2., 3., 3.5, -0.5]], [[-3.5, 1., -3.5, 3.5]]]]))

  np.testing.assert_array_equal(
      new_hw["conv"]["scales"][0], np.array([[[[0.25, 0.5, 0.25, 0.25]]]]))

  np.testing.assert_array_equal(
      new_hw["dw_conv"]["weights"][0].numpy().flatten(),
      np.array([
          0., 14, 8, 4, 0, 6, -2, 4, -2, -42, 46, -4, -42, 2, -8, 12]))

  np.testing.assert_array_equal(
      new_hw["dense"]["scales"][0], np.array([[0.25, 0.25]]))


def test_clone_model_and_freeze_auto_po2_scale_serialization():
  # Test if the cloned model can be saved and loaded properly.
  orig_model = create_test_model_for_scale_freezing(quantized_bits(4, 2, 1))
  new_model, _ = clone_model_and_freeze_auto_po2_scale(
      orig_model, quantize_model_weights=True)

  fd, fname = tempfile.mkstemp(".hdf5")
  new_model.save(fname)
  _ = load_qmodel(fname)
  os.close(fd)
  os.remove(fname)


def test_clone_model_and_freeze_auto_po2_scale_error():
  orig_model = create_test_model_for_scale_freezing(
      quantized_bits(4, 2, 1, alpha="auto_po2"))
  # Test if the function raises an error when there are more than one
  # auto_po2 quantizers in a layer.
  with pytest.raises(ValueError):
    clone_model_and_freeze_auto_po2_scale(
        orig_model, quantize_model_weights=False)


if __name__ == "__main__":
  pytest.main([__file__])
