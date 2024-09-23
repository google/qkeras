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
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
import tempfile
import os

import tensorflow as tf

from qkeras import quantized_bits
from qkeras import QSeparableConv2DTransposeTPU
from qkeras import QSeparableConv2DTransposeCPU


def create_model(for_tpu=True):
  x = img_input = tf.keras.layers.Input(shape=(4, 4, 3))

  if for_tpu:
    x = QSeparableConv2DTransposeTPU(
        filters=2, kernel_size=(2, 2),
        strides=(2, 2),
        padding="same", name="conv2d_tran",
        depthwise_activation=None,
        pointwise_activation=None,
        depthwise_kernel_quantizer=None,
        pointwise_kernel_quantizer=None,
        bias_quantizer=None,
        )(x)
  else:
    x = QSeparableConv2DTransposeCPU(
        filters=2, kernel_size=(2, 2),
        strides=(2, 2),
        padding="same", name="conv2d_tran",
        depthwise_activation=None,
        pointwise_activation=None,
        depthwise_kernel_quantizer=None,
        pointwise_kernel_quantizer=None,
        bias_quantizer=None,
        )(x)

  model = tf.keras.Model(inputs=img_input, outputs=x)

  return model


def create_quantized_model(for_tpu=True):
  x = img_input = tf.keras.layers.Input(shape=(4, 4, 3))

  if for_tpu:
    x = QSeparableConv2DTransposeTPU(
        filters=2, kernel_size=(2, 2),
        strides=(1, 1),
        padding="same", name="conv2d_tran",
        depthwise_activation="quantized_bits(10, 6, 1)",
        pointwise_activation="quantized_bits(5, 3, 1)",
        depthwise_kernel_quantizer=quantized_bits(1, 0, 1, alpha=1.0),
        pointwise_kernel_quantizer=quantized_bits(1, 0, 1, alpha=1.0),
        bias_quantizer=quantized_bits(2, 2, 1, alpha=1.0)
        )(x)
  else:
    x = QSeparableConv2DTransposeCPU(
        filters=2, kernel_size=(2, 2),
        strides=(1, 1),
        padding="same", name="conv2d_tran",
        depthwise_activation="quantized_bits(10, 6, 1)",
        pointwise_activation="quantized_bits(5, 3, 1)",
        depthwise_kernel_quantizer=quantized_bits(1, 0, 1, alpha=1.0),
        pointwise_kernel_quantizer=quantized_bits(1, 0, 1, alpha=1.0),
        bias_quantizer=quantized_bits(2, 2, 1, alpha=1.0)
        )(x)

  model = tf.keras.Model(inputs=img_input, outputs=x)

  return model


def test_qseparable_conv2d_transpose():
  # By setting the weights and input values manually, we can test
  # the correctness of the output.

  # Input is (1, 4, 4, 3), with 3 output channels. For i-th channel,
  # with shape (1, 4, 4, 1), it will convolve with the depthwise kernel at
  # i-th channel. Depthwise outputs are (1, 4, 4, 3). DW output is then
  # mapped from input channel(3) to output channel (2) by pointwise conv.
  # Pointwise conv output is (1, 8, 8, 2).

  # Create model using CPU version: QSeparableConv2DTransposeCPU.
  model = create_model(for_tpu=False)

  output_shape = model.output_shape
  ws = model.layers[1].weights

  x = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])
  inputs = np.concatenate([x, x, x], axis=-1)
  inputs = tf.constant(inputs.reshape((1, 4, 4, -1)), dtype=tf.float32)

  k = tf.ones((2, 2, 1, 1))
  dw_kernel = tf.concat([k, 1+k, 2+k], axis=-1)

  k = tf.ones((1, 1, 1, 3))
  pt_kernel = tf.concat([k, 1+k], axis=-2)

  bias = tf.zeros((2,))

  model.layers[1].set_weights([dw_kernel, pt_kernel, bias])

  actual_output = model.predict(inputs).astype(np.float16)

  predicted = np.array(
      [[[[0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.],
         [0., 0.], [0., 0.]],
        [[0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.],
         [0., 0.], [0., 0.]],
        [[6., 12.], [6., 12.], [6., 12.], [6., 12.], [6., 12.], [6., 12.],
         [6., 12.], [6., 12.]],
        [[6., 12.], [6., 12.], [6., 12.], [6., 12.], [6., 12.],
         [6., 12.], [6., 12.], [6., 12.]],
        [[12., 24.], [12., 24.], [12., 24.], [12., 24.], [12., 24.],
         [12., 24.], [12., 24.], [12., 24.]],
        [[12., 24.], [12., 24.], [12., 24.], [12., 24.], [12., 24.],
         [12., 24.], [12., 24.], [12., 24.]],
        [[18., 36.], [18., 36.], [18., 36.], [18., 36.], [18., 36.],
         [18., 36.], [18., 36.], [18., 36.]],
        [[18., 36.], [18., 36.], [18., 36.], [18., 36.], [18., 36.],
         [18., 36.], [18., 36.], [18., 36.]]]])

  assert_equal(output_shape[1:], (8, 8, 2))
  assert_equal(len(ws), 3)

  # Test if the depthwise conv kernel shape is correct.
  assert_equal(ws[0].shape, (2, 2, 1, 3))

  # Test if the pointwise conv kernel shape is correct.
  assert_equal(ws[1].shape, (1, 1, 2, 3))

  # Test if the bias shape is correct.
  assert_equal(ws[2].shape, (2,))

  # Test if overall output is correct.
  assert_equal(actual_output, predicted)


def test_quantization_in_separable_conv2d_transpose():
  # Test if quantization is applied correctly.

  # Create model using CPU version: QSeparableConv2DTransposeCPU
  # with quantization.
  model = create_quantized_model(for_tpu=False)

  x = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])
  inputs = np.concatenate([x, x, x], axis=-1)
  inputs = tf.constant(inputs.reshape((1, 4, 4, -1)), dtype=tf.float32)

  k = tf.ones((2, 2, 1, 1))
  dw_kernel = tf.concat([k, 1+k, 2+k], axis=-1)

  k = tf.ones((1, 1, 1, 3))
  pt_kernel = tf.concat([k, 1+k], axis=-2)

  bias = tf.ones((2,))

  model.layers[1].set_weights([dw_kernel, pt_kernel, bias])

  actual_output = model.predict(inputs).astype(np.float16)

  qs = model.layers[1].get_quantizers()
  assert_equal(len(qs), 5)
  assert_equal(str(qs[0]), "quantized_bits(1,0,1,alpha=1.0)")
  assert_equal(str(qs[1]), "quantized_bits(1,0,1,alpha=1.0)")
  assert_equal(str(qs[2]), "quantized_bits(2,2,1,alpha=1.0)")
  assert_equal(str(qs[3]), "quantized_bits(10,6,1)")
  assert_equal(str(qs[4]), "quantized_bits(5,3,1)")

  expected = np.array(
      [[[[0., 0.], [0., 0.], [0., 0.], [0., 0.]],
        [[3., 3.], [6., 6.], [6., 6.], [6., 6.]],
        [[7.5, 7.5], [7.5, 7.5], [7.5, 7.5], [7.5, 7.5]],
        [[7.5, 7.5], [7.5, 7.5], [7.5, 7.5], [7.5, 7.5]]]]
  )

  assert_equal(actual_output, expected)


def test_save_and_load_model():
  # Test if the model can be loaded from a saved model.
  model = create_quantized_model(for_tpu=True)

  fd, fname = tempfile.mkstemp(".hdf5")
  model.save(fname)

  custom_object = {
      "QSeparableConv2DTransposeTPU": QSeparableConv2DTransposeTPU,
      "QSeparableConv2DTransposeCPU": QSeparableConv2DTransposeCPU,
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
