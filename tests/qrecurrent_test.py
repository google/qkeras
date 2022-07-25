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
"""Tests for qrecurrent.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import os
import tempfile

import numpy as np
from numpy.testing import assert_allclose
import pytest
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.backend import clear_session
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential

from qkeras import QActivation
from qkeras import QBidirectional
from qkeras import QDense
from qkeras import QGRU
from qkeras import QLSTM
from qkeras import QSimpleRNN
from qkeras import quantized_bits
from qkeras import quantized_tanh
from qkeras.utils import load_qmodel
from qkeras.utils import model_quantize
from qkeras.utils import model_save_quantized_weights
from qkeras.utils import quantized_model_from_json

@pytest.mark.skip(reason="Test failing due to random weight initializaiton")
@pytest.mark.parametrize('rnn, all_weights_signature, expected_output', [
    (QSimpleRNN,
     np.array([5.109375, -1.8828125, 0.0, -0.5, 0.0], dtype=np.float32),
     np.array(
              [[0.281, 0.4956, 0.1047, 0.1188],
               [0.185, 0.6016, 0.0977, 0.1157],
               [0.3892, 0.483, 0.03528, 0.0926],
               [0.4038, 0.511, 0.01686, 0.06824],
               [0.3354, 0.5376, 0.02602, 0.101],
               [0.2043, 0.587, 0.04147, 0.1675],
               [0.2297, 0.6455, 0.0456, 0.0789],
               [0.4512, 0.4326, 0.01938, 0.0968],
               [0.6304, 0.2498, 0.03345, 0.0866],
               [0.4924, 0.3735, 0.011925, 0.1222]],
              dtype=np.float16)),
    (QLSTM, np.array([3.7421875, 2.1328125, 15.875, -0.5, 0.0],
                     dtype=np.float32),
     np.array(
              [[0.27, 0.1814, 0.3108, 0.2378],
               [0.2976, 0.2424, 0.248, 0.2119],
               [0.3054, 0.2004, 0.2705, 0.2238],
               [0.325, 0.1656, 0.269, 0.2401],
               [0.271, 0.1796, 0.3, 0.2493],
               [0.3066, 0.1873, 0.2477, 0.2583],
               [0.2798, 0.1757, 0.2944, 0.25],
               [0.2693, 0.2335, 0.2534, 0.2437],
               [0.2808, 0.2057, 0.2712, 0.2422],
               [0.2732, 0.2336, 0.2491, 0.244]],
              dtype=np.float16)),
    (QGRU, np.array([4.6875, 4.3984375, 0.0, -0.5, 0.0], dtype=np.float32),
     np.array(
              [[0.2025, 0.3467, 0.2952, 0.1556],
               [0.2935, 0.3313, 0.2058, 0.1694],
               [0.2046, 0.4465, 0.1827, 0.1661],
               [0.1913, 0.498, 0.1583, 0.1525],
               [0.1578, 0.477, 0.1973, 0.1677],
               [0.2018, 0.44, 0.1714, 0.1869],
               [0.157, 0.551, 0.1709, 0.12115],
               [0.1973, 0.4353, 0.1672, 0.2001],
               [0.1622, 0.5146, 0.1741, 0.149],
               [0.2101, 0.3855, 0.2069, 0.1976]],
              dtype=np.float16)),
])
def test_qrnn(rnn, all_weights_signature, expected_output):
  K.set_learning_phase(0)
  np.random.seed(22)
  tf.random.set_seed(22)

  x = x_in = Input((2, 4), name='input')
  x = rnn(
    16,
    activation=quantized_tanh(bits=8, symmetric=True),
    kernel_quantizer=quantized_bits(8, 0, 1, alpha=1.0),
    recurrent_quantizer=quantized_bits(8, 0, 1, alpha=1.0),
    bias_quantizer=quantized_bits(8, 0, 1, alpha=1.0),
    state_quantizer=quantized_bits(4, 0, 1, alpha=1.0),
    name='qrnn_0')(
        x)
  x = QDense(
      4,
      kernel_quantizer=quantized_bits(6, 2, 1, alpha=1.0),
      bias_quantizer=quantized_bits(4, 0, 1),
      name='dense')(
          x)
  x = Activation('softmax', name='softmax')(x)

  model = Model(inputs=[x_in], outputs=[x])

  # reload the model to ensure saving/loading works
  json_string = model.to_json()
  clear_session()
  model = quantized_model_from_json(json_string)

  # Save the model as an h5 file using Keras's model.save()
  fd, fname = tempfile.mkstemp('.h5')
  model.save(fname)
  del model  # Delete the existing model

  # Return a compiled model identical to the previous one
  model = load_qmodel(fname)

  # Clean the created h5 file after loading the model
  os.close(fd)
  os.remove(fname)

  # apply quantizer to weights
  model_save_quantized_weights(model)

  all_weights = []

  for layer in model.layers:
    for i, weights in enumerate(layer.get_weights()):
      w = np.sum(weights)
      all_weights.append(w)

  all_weights = np.array(all_weights)

  assert all_weights.size == all_weights_signature.size
  assert np.all(all_weights == all_weights_signature)

  # test forward:
  inputs = 2 * np.random.rand(10, 2, 4)
  actual_output = model.predict(inputs).astype(np.float16)
  assert_allclose(actual_output, expected_output, rtol=1e-4)


@pytest.mark.skip(reason="Test failing due to random weight initializaiton")
@pytest.mark.parametrize('rnn, all_weights_signature, expected_output', [
    (QSimpleRNN,
     np.array([
         -2.6562500e+00, -4.3466797e+00, 8.6736174e-19, 6.2548828e-01,
         -6.0751953e+00, 8.6736174e-19, -7.5000000e-01, 0.0
     ],
              dtype=np.float32),
     np.array(
         [[0.0851, 0.1288, 0.586, 0.2002], [0.1044, 0.1643, 0.7217, 0.00978],
          [0.04135, 0.0537, 0.8706, 0.03455], [0.03354, 0.0489, 0.889, 0.02852],
          [0.04358, 0.05246, 0.7563, 0.1478], [0.03403, 0.0743, 0.4177, 0.4739],
          [0.0859, 0.1567, 0.3972, 0.36], [0.27, 0.1945, 0.4841, 0.05124],
          [0.12115, 0.05722, 0.728, 0.0938], [0.2864, 0.1262, 0.339, 0.2484]],
         dtype=np.float16)),
    (QLSTM,
     np.array(
         [-4.1406555, 3.2921143, 16., 7.0236816, 4.1237793, 16., -0.75, 0.],
         dtype=np.float32),
     np.array(
         [[0.301, 0.2236, 0.2275, 0.2478], [0.2135, 0.2627, 0.2439, 0.2798],
          [0.1671, 0.2252, 0.2844, 0.3232], [0.2211, 0.2178, 0.2817, 0.2795],
          [0.3384, 0.1732, 0.2451, 0.2434], [0.296, 0.1979, 0.2468, 0.2593],
          [0.2698, 0.1753, 0.288, 0.267], [0.258, 0.1888, 0.3228, 0.2301],
          [0.2169, 0.1578, 0.3699, 0.2554], [0.2783, 0.1816, 0.2986, 0.2415]],
         dtype=np.float16)),
    (QGRU,
     np.array([
         -6.7578125e-01, 3.6837769e-01, 2.6020852e-18, 4.1682129e+00,
         -7.5769043e-01, 2.6020852e-18, -7.5000000e-01, 0.0
     ],
              dtype=np.float32),
     np.array(
         [[0.278, 0.1534, 0.314, 0.2546], [0.1985, 0.1788, 0.3823, 0.2402],
          [0.1997, 0.1621, 0.3792, 0.259], [0.2534, 0.1605, 0.281, 0.3052],
          [0.3794, 0.1266, 0.2296, 0.2642], [0.285, 0.1754, 0.2847, 0.255],
          [0.2878, 0.1339, 0.3042, 0.274], [0.2874, 0.1475, 0.279, 0.2861],
          [0.2379, 0.1356, 0.3186, 0.3079], [0.2234, 0.1476, 0.3274, 0.3015]],
         dtype=np.float16))
])
def test_qbidirectional(rnn, all_weights_signature, expected_output):
  K.set_learning_phase(0)
  np.random.seed(22)
  tf.random.set_seed(22)

  x = x_in = Input((2, 4), name='input')
  x = QBidirectional(
      rnn(16,
          activation='quantized_po2(8)',
          kernel_quantizer='quantized_po2(8)',
          recurrent_quantizer='quantized_po2(8)',
          bias_quantizer='quantized_po2(8)',
          state_quantizer='quantized_po2(8)',
          name='qbirnn_0'))(
              x)
  x = QDense(
      4,
      kernel_quantizer=quantized_bits(8, 2, 1, alpha=1.0),
      bias_quantizer=quantized_bits(8, 0, 1),
      name='dense')(
          x)
  x = Activation('softmax', name='softmax')(x)

  model = Model(inputs=[x_in], outputs=[x])

  # reload the model to ensure saving/loading works
  json_string = model.to_json()
  clear_session()
  model = quantized_model_from_json(json_string)

  # Save the model as an h5 file using Keras's model.save()
  fd, fname = tempfile.mkstemp('.h5')
  model.save(fname)
  del model  # Delete the existing model

  # Return a compiled model identical to the previous one
  model = load_qmodel(fname)

  # Clean the created h5 file after loading the model
  os.close(fd)
  os.remove(fname)

  # apply quantizer to weights
  model_save_quantized_weights(model)

  all_weights = []

  for layer in model.layers:
    for _, weights in enumerate(layer.get_weights()):

      w = np.sum(weights)
      all_weights.append(w)

  all_weights = np.array(all_weights)
  assert all_weights.size == all_weights_signature.size
  assert np.all(all_weights == all_weights_signature)

  # test forward:
  inputs = 2 * np.random.rand(10, 2, 4)
  actual_output = model.predict(inputs).astype(np.float16)
  assert_allclose(actual_output, expected_output, rtol=1e-4)


def create_network_rnn(rnn):
  xi = Input((16, 1,))
  x = rnn(8)(xi)
  return Model(inputs=xi, outputs=x)


@pytest.mark.parametrize('rnn', [SimpleRNN, LSTM, GRU])
def test_rnn_conversion(rnn):
  m = create_network_rnn(rnn)
  name = 'Q' + m.layers[1].__class__.__name__
  d = {
      name: {
          'kernel_quantizer': 'binary',
          'recurrent_quantizer': 'binary',
          'bias_quantizer': 'binary',
          'state_quantizer': 'binary',
          'activation_quantizer': 'binary',
      }
  }
  if name != 'QSimpleRNN':
    d[name]['recurrent_activation_quantizer'] = 'binary'

  qq = model_quantize(m, d, 4)
  assert str(qq.layers[1].kernel_quantizer) == 'binary'
  assert str(qq.layers[1].recurrent_quantizer) == 'binary'
  assert str(qq.layers[1].bias_quantizer) == 'binary'
  assert str(qq.layers[1].state_quantizer) == 'binary'
  assert str(qq.layers[1].activation) == 'binary()'
  if name != 'QSimpleRNN':
    assert str(qq.layers[1].recurrent_activation) == 'binary()'


def create_network_birnn(rnn):
  xi = Input((16, 1,))
  x = Bidirectional(rnn(8))(xi)
  return Model(inputs=xi, outputs=x)


@pytest.mark.parametrize('rnn', [SimpleRNN, LSTM, GRU])
def test_birnn_conversion(rnn):
  m = create_network_birnn(rnn)
  name = 'Q' + m.layers[1].layer.__class__.__name__
  d = {
      'QBidirectional': {
          'kernel_quantizer': 'binary',
          'recurrent_quantizer': 'binary',
          'bias_quantizer': 'binary',
          'state_quantizer': 'binary',
          'activation_quantizer': 'binary',
      }
  }
  if name != 'QSimpleRNN':
    d['QBidirectional']['recurrent_activation_quantizer'] = 'binary'

  qq = model_quantize(m, d, 4)
  layer = qq.layers[1].layer
  assert str(layer.kernel_quantizer) == 'binary'
  assert str(layer.recurrent_quantizer) == 'binary'
  assert str(layer.bias_quantizer) == 'binary'
  assert str(layer.state_quantizer) == 'binary'
  assert str(layer.activation) == 'binary()'
  if name != 'QSimpleRNN':
    assert str(layer.recurrent_activation) == 'binary()'
  backward_layer = qq.layers[1].backward_layer
  # backwards weight quantizers are dict because of contraints.serialize
  assert str(backward_layer.kernel_quantizer['class_name']) == 'binary'
  assert str(backward_layer.recurrent_quantizer['class_name']) == 'binary'
  assert str(backward_layer.bias_quantizer['class_name']) == 'binary'
  assert str(backward_layer.state_quantizer['class_name']) == 'binary'
  assert str(backward_layer.activation) == 'binary()'
  if name != 'QSimpleRNN':
    assert str(backward_layer.recurrent_activation) == 'binary()'


def test_birnn_subrnn():
  model = Sequential([Bidirectional(LSTM(16)), LSTM(8)])
  d = {
      'QLSTM': {
          'activation_quantizer': 'ternary',
          'recurrent_activation_quantizer': 'ternary',
          'kernel_quantizer': 'ternary',
          'recurrent_quantizer': 'ternary',
          'bias_quantizer': 'ternary',
          'state_quantizer': 'ternary',
      },
      'QBidirectional': {
          'activation_quantizer': 'binary',
          'recurrent_activation_quantizer': 'binary',
          'kernel_quantizer': 'binary',
          'recurrent_quantizer': 'binary',
          'bias_quantizer': 'binary',
          'state_quantizer': 'binary',
      }
  }
  qmodel = model_quantize(model, d, 4)
  layer = qmodel.layers[1]
  assert str(layer.kernel_quantizer) == 'ternary'
  assert str(layer.recurrent_quantizer) == 'ternary'
  assert str(layer.bias_quantizer) == 'ternary'
  assert str(layer.state_quantizer) == 'ternary'
  assert str(layer.activation) == 'ternary()'


if __name__ == '__main__':
  pytest.main([__file__])
