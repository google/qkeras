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
"""Test layers from qlayers.py."""

import numpy as np

from qkeras import binary
from qkeras import model_save_quantized_weights
from qkeras import QActivation
from qkeras import QConv1D
from qkeras import QConv2D
from qkeras import QDense
from qkeras import QSeparableConv2D
from qkeras import quantized_bits
from qkeras import ternary

import numpy as np
from numpy.testing import assert_allclose
import pytest
from keras import backend as K
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model


def qdense_util(layer_cls,
                kwargs=None,
                input_data=None,
                weight_data=None,
                expected_output=None):
  """qlayer test utility."""
  input_shape = input_data.shape
  input_dtype = input_data.dtype
  layer = layer_cls(**kwargs)
  x = Input(shape=input_shape[1:], dtype=input_dtype)
  y = layer(x)
  layer.set_weights(weight_data)
  model = Model(x, y)
  actual_output = model.predict(input_data)
  if expected_output is not None:
    assert_allclose(actual_output, expected_output, rtol=1e-4)


@pytest.mark.parametrize(
    'layer_kwargs, input_data, weight_data, bias_data, expected_output', [
        ({
            'units': 2,
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros'
        }, np.array([[1, 1, 1, 1]], dtype=K.floatx()),
         np.array([[10, 20], [10, 20], [10, 20], [10, 20]],
                  dtype=K.floatx()), np.array([0, 0], dtype=K.floatx()),
         np.array([[40, 80]], dtype=K.floatx())),
        ({
            'units': 2,
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_quantizer': 'quantized_bits(2,0)',
            'bias_quantizer': 'quantized_bits(2,0)',
        }, np.array([[1, 1, 1, 1]], dtype=K.floatx()),
         np.array([[10, 20], [10, 20], [10, 20], [10, 20]], dtype=K.floatx()),
         np.array([0, 0], dtype=K.floatx()), np.array([[2, 2]],
                                                      dtype=K.floatx())),
    ])
def test_qdense(layer_kwargs, input_data, weight_data, bias_data,
                expected_output):
  qdense_util(
      layer_cls=QDense,
      kwargs=layer_kwargs,
      input_data=input_data,
      weight_data=[weight_data, bias_data],
      expected_output=expected_output)

def test_qnetwork():
  x = x_in = Input((28, 28, 1), name='input')
  x = QSeparableConv2D(
      32, (2, 2),
      strides=(2, 2),
      depthwise_quantizer=binary(),
      pointwise_quantizer=quantized_bits(4, 0, 1),
      depthwise_activation=quantized_bits(6, 2, 1),
      bias_quantizer=quantized_bits(4, 0, 1),
      name='conv2d_0_m')(
          x)
  x = QActivation('quantized_relu(6,2,1)', name='act0_m')(x)
  x = QConv2D(
      64, (3, 3),
      strides=(2, 2),
      kernel_quantizer=ternary(),
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
  all_weights_signature = np.array([2.0, -6.75, -0.625, -2.0, -0.25, -56.0,
                                    1.125, -2.625, -0.75])
  assert all_weights.size == all_weights_signature.size
  assert np.all(all_weights == all_weights_signature)


  # test_qnetwork_forward:
  y = np.array([[0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
               5.341e-02, 9.468e-01, 0.000e+00, 0.000e+00, 0.000e+00],
              [0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 5.960e-08,
               0.000e+00, 1.919e-01, 0.000e+00, 0.000e+00, 8.081e-01],
              [0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 2.378e-04,
               0.000e+00, 0.000e+00, 0.000e+00, 2.843e-05, 9.995e-01],
              [0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
               0.000e+00, 1.000e+00, 0.000e+00, 0.000e+00, 0.000e+00],
              [0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
               0.000e+00, 1.000e+00, 0.000e+00, 2.623e-06, 0.000e+00],
              [0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
               7.749e-07, 0.000e+00, 0.000e+00, 1.634e-04, 1.000e+00],
              [0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
               0.000e+00, 1.000e+00, 0.000e+00, 0.000e+00, 0.000e+00],
              [0.000e+00, 1.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
               0.000e+00, 6.557e-07, 0.000e+00, 0.000e+00, 0.000e+00],
              [0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 1.000e+00,
               0.000e+00, 5.960e-08, 0.000e+00, 0.000e+00, 0.000e+00],
              [0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.125e-03,
               9.907e-01, 9.418e-06, 0.000e+00, 5.597e-05, 0.000e+00
              ]]).astype(np.float16)

  inputs = 2 * np.random.rand(10, 28, 28, 1)
  p = model.predict(inputs).astype(np.float16)
  assert np.all(p == y)


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

  # apply quantizer to weights
  model_save_quantized_weights(model)

  inputs = np.random.rand(2, 4, 4)
  p = model.predict(inputs).astype(np.float16)

  y = np.array([[[0.1309, -1.229], [-0.4165, -2.639], [-0.08105, -2.299],
                 [1.981, -2.195]],
                [[-0.3174, -3.94], [-0.3352, -2.316], [0.105, -0.833],
                 [0.2115, -2.89]]]).astype(np.float16)

  assert np.all(p == y)


if __name__ == '__main__':
  pytest.main([__file__])
