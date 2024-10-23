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
"""Test layers from qconvolutional.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
import tensorflow as tf

from qkeras import QDepthwiseConv2DTranspose
from qkeras import quantized_bits


# Predicted output from float model.
_FLOAT_PREDICTED_OUTPUT = np.array([[
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ],
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ],
    [
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ],
    [
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ],
    [
        [2.0, 4.0, 6.0],
        [2.0, 4.0, 6.0],
        [2.0, 4.0, 6.0],
        [2.0, 4.0, 6.0],
        [2.0, 4.0, 6.0],
        [2.0, 4.0, 6.0],
        [2.0, 4.0, 6.0],
        [2.0, 4.0, 6.0],
    ],
    [
        [2.0, 4.0, 6.0],
        [2.0, 4.0, 6.0],
        [2.0, 4.0, 6.0],
        [2.0, 4.0, 6.0],
        [2.0, 4.0, 6.0],
        [2.0, 4.0, 6.0],
        [2.0, 4.0, 6.0],
        [2.0, 4.0, 6.0],
    ],
    [
        [3.0, 6.0, 9.0],
        [3.0, 6.0, 9.0],
        [3.0, 6.0, 9.0],
        [3.0, 6.0, 9.0],
        [3.0, 6.0, 9.0],
        [3.0, 6.0, 9.0],
        [3.0, 6.0, 9.0],
        [3.0, 6.0, 9.0],
    ],
    [
        [3.0, 6.0, 9.0],
        [3.0, 6.0, 9.0],
        [3.0, 6.0, 9.0],
        [3.0, 6.0, 9.0],
        [3.0, 6.0, 9.0],
        [3.0, 6.0, 9.0],
        [3.0, 6.0, 9.0],
        [3.0, 6.0, 9.0],
    ],
]])


def create_model(group_size=1):
  x = img_input = tf.keras.layers.Input(shape=(4, 4, 3))
  x = QDepthwiseConv2DTranspose(
      filters=2,
      kernel_size=(2, 2),
      strides=(2, 2),
      padding="same",
      name="conv2d_tran",
      depthwise_activation=None,
      depthwise_kernel_quantizer=None,
      bias_quantizer=None,
      group_size=group_size,
  )(x)

  model = tf.keras.Model(inputs=img_input, outputs=x)

  return model


def create_quantized_model(group_size=1):
  x = img_input = tf.keras.layers.Input(shape=(4, 4, 3))
  x = QDepthwiseConv2DTranspose(
      filters=2,
      kernel_size=(2, 2),
      strides=(1, 1),
      padding="same",
      name="conv2d_tran",
      depthwise_activation="quantized_bits(10, 6, 1)",
      depthwise_kernel_quantizer=quantized_bits(1, 0, 1, alpha=1.0),
      bias_quantizer=quantized_bits(2, 2, 1, alpha=1.0),
      group_size=group_size,
  )(x)

  model = tf.keras.Model(inputs=img_input, outputs=x)

  return model


def test_qseparable_conv2d_transpose():
  # By setting the weights and input values manually, we can test
  # the correctness of the output.

  # Input is (1, 4, 4, 3), with 3 output channels. For i-th channel,
  # with shape (1, 4, 4, 1), it will convolve with the depthwise kernel at
  # i-th channel. Depthwise outputs are (1, 8, 8, 3).

  # Create model.
  model = create_model()

  output_shape = model.output_shape
  ws = model.layers[1].weights

  x = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])
  inputs = np.concatenate([x, x, x], axis=-1)
  inputs = tf.constant(inputs.reshape((1, 4, 4, -1)), dtype=tf.float32)

  # depthwise kernel of shape (2, 2, 3, 1)
  dw_kernel = np.array([
      [[[1.0], [2.0], [3.0]], [[1.0], [2.0], [3.0]]],
      [[[1.0], [2.0], [3.0]], [[1.0], [2.0], [3.0]]],
  ])

  bias = tf.zeros((2,))

  model.layers[1].set_weights([dw_kernel, bias])

  actual_output = model.predict(inputs).astype(np.float16)
  assert_equal(output_shape[1:], (8, 8, 3))
  assert_equal(len(ws), 2)

  # Test if the depthwise conv kernel shape is correct.
  assert_equal(ws[0].shape, (2, 2, 3, 1))

  # Test if the bias shape is correct.
  assert_equal(ws[1].shape, (2,))

  # Test if overall output is correct.
  assert_equal(actual_output, _FLOAT_PREDICTED_OUTPUT)


def test_quantization_in_separable_conv2d_transpose():
  # Test if quantization is applied correctly.

  # Create model with quantization.
  model = create_quantized_model()

  x = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])
  inputs = np.concatenate([x, x, x], axis=-1)
  inputs = tf.constant(inputs.reshape((1, 4, 4, -1)), dtype=tf.float32)

  # depthwise kernel of shape (2, 2, 3, 1)
  dw_kernel = np.array([
      [[[1.0], [2.0], [3.0]], [[1.0], [2.0], [3.0]]],
      [[[1.0], [2.0], [3.0]], [[1.0], [2.0], [3.0]]],
  ])

  bias = tf.ones((2,))

  model.layers[1].set_weights([dw_kernel, bias])

  actual_output = model.predict(inputs).astype(np.float16)

  qs = model.layers[1].get_quantizers()
  assert_equal(len(qs), 3)
  assert_equal(str(qs[0]), "quantized_bits(1,0,1,alpha=1.0)")
  assert_equal(str(qs[1]), "quantized_bits(2,2,1,alpha=1.0)")
  assert_equal(str(qs[2]), "quantized_bits(10,6,1)")

  expected = np.array([[
      [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
      [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
      [[3.0, 3.0, 3.0], [6.0, 6.0, 6.0], [6.0, 6.0, 6.0], [6.0, 6.0, 6.0]],
      [
          [5.0, 5.0, 5.0],
          [10.0, 10.0, 10.0],
          [10.0, 10.0, 10.0],
          [10.0, 10.0, 10.0],
      ],
  ]])

  assert_equal(actual_output, expected)


def test_qseparable_conv2d_transpose_with_groups():
  model = create_model(group_size=3)

  output_shape = model.output_shape
  ws = model.layers[1].weights

  x = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])
  inputs = np.concatenate([x, x, x], axis=-1)
  inputs = tf.constant(inputs.reshape((1, 4, 4, -1)), dtype=tf.float32)

  # depthwise kernel of shape (2, 2, 3, 3)
  dw_kernel = np.array([
      [
          [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
          [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
      ],
      [
          [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
          [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
      ],
  ])

  bias = tf.zeros((2,))

  model.layers[1].set_weights([dw_kernel, bias])

  actual_output = model.predict(inputs).astype(np.float16)

  predicted = _FLOAT_PREDICTED_OUTPUT * 3.0  # kernel values replicated 3 times

  assert_equal(output_shape[1:], (8, 8, 3))
  assert_equal(len(ws), 2)

  # Test if the depthwise conv kernel shape is correct.
  assert_equal(ws[0].shape, (2, 2, 3, 3))

  # Test if the bias shape is correct.
  assert_equal(ws[1].shape, (2,))

  # Test if overall output is correct.
  assert_equal(actual_output, predicted)


def test_save_and_load_model():
  # Test if the model can be loaded from a saved model.
  model = create_quantized_model(group_size=3)

  fd, fname = tempfile.mkstemp(".hdf5")
  model.save(fname)

  custom_object = {
      "QDepthwiseConv2DTranspose": QDepthwiseConv2DTranspose,
  }

  model_loaded = tf.keras.models.load_model(
      fname, custom_objects=custom_object)

  # Clean the h5 file after loading the model
  os.close(fd)
  os.remove(fname)

  model_weights = model.layers[1].weights
  loaded_model_weights = model_loaded.layers[1].weights

  assert_equal(len(model_weights), len(loaded_model_weights))
  for i, model_weight in enumerate(model_weights):
    assert_equal(model_weight.numpy(), loaded_model_weights[i].numpy())


if __name__ == "__main__":
  pytest.main([__file__])
