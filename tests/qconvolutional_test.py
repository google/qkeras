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

import os
import tempfile
import numpy as np
from numpy.testing import assert_allclose
import pytest
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.backend import clear_session

from qkeras import QActivation
from qkeras import QDense
from qkeras import QConv1D
from qkeras import QConv2D
from qkeras import QSeparableConv2D
from qkeras import quantized_bits
from qkeras.utils import model_save_quantized_weights
from qkeras.utils import quantized_model_from_json
from qkeras.utils import load_qmodel
from qkeras import print_qstats
from qkeras import extract_model_operations

def test_qnetwork():
  x = x_in = Input((28, 28, 1), name='input')
  x = QSeparableConv2D(
      32, (2, 2),
      strides=(2, 2),
      depthwise_quantizer="binary",
      pointwise_quantizer=quantized_bits(4, 0, 1),
      depthwise_activation=quantized_bits(6, 2, 1),
      bias_quantizer=quantized_bits(4, 0, 1),
      name='conv2d_0_m')(
          x)
  x = QActivation('quantized_relu(6,2,1)', name='act0_m')(x)
  x = QConv2D(
      64, (3, 3),
      strides=(2, 2),
      kernel_quantizer="ternary",
      bias_quantizer=quantized_bits(4, 0, 1),
      name='conv2d_1_m')(
          x)
  x = QActivation('quantized_relu(6, 3, 1)', name='act1_m')(x)
  x = QConv2D(
      64, (2, 2),
      strides=(2, 2),
      kernel_quantizer=quantized_bits(6, 2, 1),
      bias_quantizer=quantized_bits(4, 0, 1),
      name='conv2d_2_m')(
          x)
  x = QActivation('quantized_relu(6,4,1)', name='act2_m')(x)
  x = Flatten(name='flatten')(x)
  x = QDense(
      10,
      kernel_quantizer=quantized_bits(6, 2, 1),
      bias_quantizer=quantized_bits(4, 0, 1),
      name='dense')(
          x)
  x = Activation('softmax', name='softmax')(x)

  model = Model(inputs=[x_in], outputs=[x])

  # reload the model to ensure saving/loading works
  json_string = model.to_json()
  clear_session()
  model = quantized_model_from_json(json_string)

  # generate same output for weights

  np.random.seed(42)
  for layer in model.layers:
    all_weights = []
    for i, weights in enumerate(layer.get_weights()):
      input_size = np.prod(layer.input.shape.as_list()[1:])
      if input_size is None:
        input_size = 576 * 10  # to avoid learning sizes
      shape = weights.shape
      assert input_size > 0, 'input size for {} {}'.format(layer.name, i)
      # he normal initialization with a scale factor of 2.0
      all_weights.append(
          10.0 * np.random.normal(0.0, np.sqrt(2.0 / input_size), shape))
    if all_weights:
      layer.set_weights(all_weights)

  # apply quantizer to weights
  model_save_quantized_weights(model)

  all_weights = []

  for layer in model.layers:
    for i, weights in enumerate(layer.get_weights()):
      w = np.sum(weights)
      all_weights.append(w)

  all_weights = np.array(all_weights)

  # test_qnetwork_weight_quantization
  all_weights_signature = np.array(
      [2., -6.75, -0.625, -2., -0.25, -56., 1.125, -1.625, -1.125])

  assert all_weights.size == all_weights_signature.size
  assert np.all(all_weights == all_weights_signature)

  # test_qnetwork_forward:
  expected_output = np.array([[0.e+00, 0.e+00, 0.e+00, 0.e+00, 0.e+00,
                 0.e+00, 1.e+00, 0.e+00, 0.e+00, 0.e+00],
                [0.e+00, 0.e+00, 0.e+00, 0.e+00, 0.e+00,
                 0.e+00, 1.e+00, 0.e+00, 0.e+00, 0.e+00],
                [0.e+00, 0.e+00, 0.e+00, 0.e+00, 0.e+00,
                 0.e+00, 0.e+00, 0.e+00, 6.e-08, 1.e+00],
                [0.e+00, 0.e+00, 0.e+00, 0.e+00, 0.e+00,
                 0.e+00, 1.e+00, 0.e+00, 0.e+00, 0.e+00],
                [0.e+00 ,0.e+00, 0.e+00, 0.e+00, 0.e+00,
                 0.e+00, 1.e+00, 0.e+00, 0.e+00, 0.e+00],
                [0.e+00, 0.e+00, 0.e+00, 0.e+00, 0.e+00,
                 0.e+00, 0.e+00, 0.e+00, 5.e-07, 1.e+00],
                [0.e+00, 0.e+00, 0.e+00, 0.e+00, 0.e+00,
                 0.e+00 ,1.e+00, 0.e+00, 0.e+00, 0.e+00],
                [0.e+00, 1.e+00, 0.e+00, 0.e+00, 0.e+00,
                 0.e+00 ,0.e+00, 0.e+00, 0.e+00, 0.e+00],
                [0.e+00, 0.e+00, 0.e+00, 0.e+00, 1.e+00,
                 0.e+00, 0.e+00, 0.e+00, 0.e+00, 0.e+00],
                [0.e+00, 0.e+00, 0.e+00, 0.e+00, 0.e+00,
                 1.e+00, 0.e+00, 0.e+00, 0.e+00, 0.e+00]]).astype(np.float16)

  inputs = 2 * np.random.rand(10, 28, 28, 1)
  actual_output = model.predict(inputs).astype(np.float16)
  assert_allclose(actual_output, expected_output, rtol=1e-4)


def test_qconv1d():
  np.random.seed(33)
  x = Input((4, 4,))
  y = QConv1D(
      2, 1,
      kernel_quantizer=quantized_bits(6, 2, 1),
      bias_quantizer=quantized_bits(4, 0, 1),
      name='qconv1d')(
          x)
  model = Model(inputs=x, outputs=y)

  #Extract model operations
  model_ops = extract_model_operations(model)

  # Assertion about the number of operations for this Conv1D layer
  assert model_ops['qconv1d']["number_of_operations"] == 32

  # Print qstats to make sure it works with Conv1D layer
  print_qstats(model) 

  # reload the model to ensure saving/loading works
  json_string = model.to_json()
  clear_session()
  model = quantized_model_from_json(json_string)

  for layer in model.layers:
    all_weights = []
    for i, weights in enumerate(layer.get_weights()):
      input_size = np.prod(layer.input.shape.as_list()[1:])
      if input_size is None:
        input_size = 10 * 10
      shape = weights.shape
      assert input_size > 0, 'input size for {} {}'.format(layer.name, i)
      all_weights.append(
          10.0 * np.random.normal(0.0, np.sqrt(2.0 / input_size), shape))
    if all_weights:
      layer.set_weights(all_weights)
    
  # Save the model as an h5 file using Keras's model.save()
  fd, fname = tempfile.mkstemp('.h5')
  model.save(fname)
  del model  # Delete the existing model

  # Returns a compiled model identical to the previous one
  model = load_qmodel(fname)

  #Clean the created h5 file after loading the model
  os.close(fd)
  os.remove(fname)

  # apply quantizer to weights
  model_save_quantized_weights(model)

  inputs = np.random.rand(2, 4, 4)
  p = model.predict(inputs).astype(np.float16)
  '''
  y = np.array([[[0.1309, -1.229], [-0.4165, -2.639], [-0.08105, -2.299],
                 [1.981, -2.195]],
                [[-0.3174, -3.94], [-0.3352, -2.316], [0.105, -0.833],
                 [0.2115, -2.89]]]).astype(np.float16)
  '''
  y = np.array([[[-2.441, 3.816], [-3.807, -1.426], [-2.684, -1.317],
                 [-1.659, 0.9834]],
                [[-4.99, 1.139], [-2.559, -1.216], [-2.285, 1.905],
                 [-2.652, -0.467]]]).astype(np.float16)
  assert np.all(p == y)



if __name__ == '__main__':
  pytest.main([__file__])
