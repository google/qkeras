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

from qkeras import QActivation
from qkeras import QSimpleRNN
from qkeras import QLSTM
from qkeras import QGRU
from qkeras import QDense
from qkeras import quantized_bits
from qkeras import quantized_tanh
from qkeras.utils import model_save_quantized_weights
from qkeras.utils import quantized_model_from_json
from qkeras.utils import load_qmodel

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

  x = x_in = Input((2,4), name='input')
  x = QSimpleRNN(
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

  # test_qnetwork_weight_quantization: TODO
  # assert all_weights.size == all_weights_signature.size
  # assert np.all(all_weights == all_weights_signature)

  # test_qnetwork_forward:  
  # inputs = 2 * np.random.rand(10, 2, 4)
  # actual_output = model.predict(inputs).astype(np.float16)
  # assert_allclose(actual_output, expected_output, rtol=1e-4)

    

if __name__ == '__main__':
  pytest.main([__file__])
