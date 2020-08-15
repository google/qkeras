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
"""Tests for qrecurrent.py"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import numpy as np
import tensorflow as tf
from numpy.testing import assert_allclose
import pytest
import tempfile
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.backend import clear_session
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.models import Sequential

from qkeras import QActivation
from qkeras import QSimpleRNN
from qkeras import QLSTM
from qkeras import QGRU
from qkeras import QBidirectional
from qkeras import QDense
from qkeras import quantized_bits
from qkeras import quantized_tanh
from qkeras.utils import model_save_quantized_weights
from qkeras.utils import quantized_model_from_json
from qkeras.utils import load_qmodel
from qkeras.utils import model_quantize


@pytest.mark.parametrize(
    'rnn, all_weights_signature, expected_output',
    [
      (
        QSimpleRNN,
        np.array([5.109375, -1.8828125, 0.0, -0.5, 0.0],
            dtype=np.float32),
        np.array(
           [[0.2812  , 0.4949  , 0.10254 , 0.1215  ],
            [0.1874  , 0.6055  , 0.09    , 0.1173  ],
            [0.3965  , 0.4778  , 0.02974 , 0.0962  ],
            [0.4158  , 0.5005  , 0.0172  , 0.06665 ],
            [0.3367  , 0.537   , 0.02444 , 0.1018  ],
            [0.2125  , 0.584   , 0.03937 , 0.164   ],
            [0.2368  , 0.639   , 0.04245 , 0.0815  ],
            [0.4468  , 0.4436  , 0.01942 , 0.0902  ],
            [0.622   , 0.257   , 0.03293 , 0.0878  ],
            [0.4814  , 0.3923  , 0.011215, 0.11505 ]], dtype=np.float16)
      ),
      (
        QLSTM,
        np.array([3.7421875, 2.1328125, 15.875, -0.5,  0.0],
            dtype=np.float32),
        np.array(
           [[0.265 , 0.1775, 0.319 , 0.2384],
            [0.2896, 0.2417, 0.2563, 0.2124],
            [0.309 , 0.193 , 0.2734, 0.2246],
            [0.322 , 0.17  , 0.2668, 0.2412],
            [0.267 , 0.174 , 0.301 , 0.2578],
            [0.311 , 0.1774, 0.2566, 0.255 ],
            [0.2854, 0.174 , 0.2927, 0.248 ],
            [0.2668, 0.2268, 0.2585, 0.2479],
            [0.2795, 0.2113, 0.2659, 0.2434],
            [0.275 , 0.2333, 0.2505, 0.2415]], dtype=np.float16)
      ),
      (
        QGRU,
        np.array([4.6875, 4.3984375, 0.0, -0.5, 0.0],
            dtype=np.float32),
        np.array(
           [[0.203 , 0.3547, 0.2854, 0.1567],
            [0.294 , 0.334 , 0.1985, 0.1736],
            [0.2096, 0.4392, 0.1812, 0.1702],
            [0.1974, 0.4927, 0.1506, 0.1593],
            [0.1582, 0.4788, 0.1968, 0.1661],
            [0.2028, 0.4421, 0.1678, 0.1871],
            [0.1583, 0.5464, 0.1705, 0.125 ],
            [0.1956, 0.4407, 0.1703, 0.1935],
            [0.1638, 0.511 , 0.1725, 0.1527],
            [0.2074, 0.3862, 0.208 , 0.1982]], dtype=np.float16)
      )
    ])
def test_qrnn(rnn, all_weights_signature, expected_output):
  K.set_learning_phase(0)
  np.random.seed(22)
  tf.random.set_seed(22)

  x = x_in = Input((2, 4), name='input')
  x = rnn(
    16,
    activation=quantized_tanh(bits=8),
    kernel_quantizer=quantized_bits(8, 0, 1, alpha=1.0),
    recurrent_quantizer=quantized_bits(8, 0, 1, alpha=1.0),
    bias_quantizer=quantized_bits(8, 0, 1, alpha=1.0),
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


@pytest.mark.parametrize(
  'rnn, all_weights_signature, expected_output',
  [
    (
      QSimpleRNN,
      np.array([
        -2.6562500e+00, -4.3466797e+00,  8.6736174e-19,  6.2548828e-01,
        -6.0751953e+00,  8.6736174e-19, -7.5000000e-01,  0.0], dtype=np.float32),
      np.array([
        [0.0851 , 0.1288 , 0.586  , 0.2002 ],
        [0.1044 , 0.1643 , 0.7217 , 0.00978],
        [0.04135, 0.0537 , 0.8706 , 0.03455],
        [0.03354, 0.0489 , 0.889  , 0.02852],
        [0.04358, 0.05246, 0.7563 , 0.1478 ],
        [0.03403, 0.0743 , 0.4177 , 0.4739 ],
        [0.0859 , 0.1567 , 0.3972 , 0.36   ],
        [0.27   , 0.1945 , 0.4841 , 0.05124],
        [0.12115, 0.05722, 0.728  , 0.0938 ],
        [0.2864 , 0.1262 , 0.339  , 0.2484 ]], dtype=np.float16)
    ),
    (
      QLSTM,
      np.array([
        -4.1406555,  3.2921143, 16.       ,  7.0236816,  4.1237793,
        16.       , -0.75     ,  0.       ], dtype=np.float32),
      np.array([
        [0.3066, 0.2026, 0.2335, 0.2573],
        [0.1796, 0.283 , 0.27  , 0.2673],
        [0.1702, 0.2144, 0.308 , 0.3074],
        [0.2216, 0.2153, 0.286 , 0.277 ],
        [0.3533, 0.1725, 0.2322, 0.2421],
        [0.2764, 0.2153, 0.227 , 0.2812],
        [0.2786, 0.1711, 0.2861, 0.2642],
        [0.2493, 0.1882, 0.3098, 0.2527],
        [0.1926, 0.1779, 0.3137, 0.316 ],
        [0.263 , 0.1783, 0.3086, 0.2502]], dtype=np.float16)
    ),
    (
      QGRU,
      np.array([
        -6.7578125e-01,  3.6837769e-01,  2.6020852e-18,  4.1682129e+00,
        -7.5769043e-01,  2.6020852e-18, -7.5000000e-01,  0.0], dtype=np.float32),
      np.array([
        [0.2764, 0.1531, 0.3047, 0.2659],
        [0.2012, 0.1885, 0.3638, 0.2466],
        [0.2024, 0.1703, 0.3718, 0.2554],
        [0.2451, 0.1581, 0.294 , 0.3027],
        [0.3987, 0.117 , 0.2343, 0.25  ],
        [0.2834, 0.1829, 0.2734, 0.2603],
        [0.2905, 0.1345, 0.3003, 0.2747],
        [0.2954, 0.1481, 0.2744, 0.2822],
        [0.2336, 0.1282, 0.334 , 0.3042],
        [0.2396, 0.1595, 0.3093, 0.2915]], dtype=np.float16)
    )
  ])
def test_qbidirectional(rnn, all_weights_signature, expected_output):
  K.set_learning_phase(0)
  np.random.seed(22)
  tf.random.set_seed(22)

  x = x_in = Input((2,4), name='input')
  x = QBidirectional(rnn(
    16,
    activation="quantized_po2(8)",
    kernel_quantizer="quantized_po2(8)",
    recurrent_quantizer="quantized_po2(8)",
    bias_quantizer="quantized_po2(8)",
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


def create_network_rnn(rnn):
  xi = Input((16, 1,))
  x = rnn(8)(xi)
  return Model(inputs=xi, outputs=x)


@pytest.mark.parametrize(
  'rnn',
  [
    SimpleRNN,
    LSTM,
    GRU
  ]
)
def test_rnn_conversion(rnn):
  m = create_network_rnn(rnn)
  name = 'Q' + m.layers[1].__class__.__name__
  d = {
    name : {
      'kernel_quantizer' : 'binary',
      'recurrent_quantizer' : 'binary',
      'bias_quantizer' : 'binary',
      'activation_quantizer' : 'binary',
    }
  }
  if name != 'QSimpleRNN':
    d[name]['recurrent_activation_quantizer'] = 'binary'

  qq = model_quantize(m, d, 4)
  assert str(qq.layers[1].kernel_quantizer) == 'binary'
  assert str(qq.layers[1].recurrent_quantizer) == 'binary'
  assert str(qq.layers[1].bias_quantizer) == 'binary'
  assert str(qq.layers[1].activation) == 'binary()'
  if name != 'QSimpleRNN':
    assert str(qq.layers[1].recurrent_activation) == 'binary()'


def create_network_birnn(rnn):
  xi = Input((16, 1,))
  x = Bidirectional(rnn(8))(xi)
  return Model(inputs=xi, outputs=x)


@pytest.mark.parametrize(
  'rnn',
  [
    SimpleRNN,
    LSTM,
    GRU
  ]
)
def test_birnn_conversion(rnn):
  m = create_network_birnn(rnn)
  name = 'Q' + m.layers[1].layer.__class__.__name__
  d = {
    'QBidirectional' : {
      'kernel_quantizer' : 'binary',
      'recurrent_quantizer' : 'binary',
      'bias_quantizer' : 'binary',
      'activation_quantizer' : 'binary',
    }
  }
  if name != 'QSimpleRNN':
    d['QBidirectional']['recurrent_activation_quantizer'] = 'binary'

  qq = model_quantize(m, d, 4)
  layer = qq.layers[1].layer
  assert str(layer.kernel_quantizer) == 'binary'
  assert str(layer.recurrent_quantizer) == 'binary'
  assert str(layer.bias_quantizer) == 'binary'
  assert str(layer.activation) == 'binary()'
  if name != 'QSimpleRNN':
    assert str(layer.recurrent_activation) == 'binary()'
  backward_layer = qq.layers[1].backward_layer
  # backwards weight quantizers are dict because of contraints.serialize
  assert str(backward_layer.kernel_quantizer['class_name']) == 'binary'
  assert str(backward_layer.recurrent_quantizer['class_name']) == 'binary'
  assert str(backward_layer.bias_quantizer['class_name']) == 'binary'
  assert str(backward_layer.activation) == 'binary()'
  if name != 'QSimpleRNN':
    assert str(backward_layer.recurrent_activation) == 'binary()'


def test_birnn_subrnn():
  model = Sequential([Bidirectional(LSTM(16)), LSTM(8)])
  d = {
    'QLSTM' : {
      'activation_quantizer' : 'ternary',
      'recurrent_activation_quantizer' : 'ternary',
      'kernel_quantizer' : 'ternary',
      'recurrent_quantizer' : 'ternary',
      'bias_quantizer' : 'ternary',
    },
    "QBidirectional": {
        'activation_quantizer' : 'binary',
        'recurrent_activation_quantizer' : 'binary',
        'kernel_quantizer' : 'binary',
        'recurrent_quantizer' : 'binary',
        'bias_quantizer' : 'binary',
    }
  }
  qmodel = model_quantize(model, d, 4)
  layer = qmodel.layers[1]
  assert str(layer.kernel_quantizer) == 'ternary'
  assert str(layer.recurrent_quantizer) == 'ternary'
  assert str(layer.bias_quantizer) == 'ternary'
  assert str(layer.activation) == 'ternary()'


if __name__ == '__main__':
  pytest.main([__file__])
