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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import tempfile

import numpy as np
from numpy.testing import assert_allclose
from numpy.testing import assert_equal
import pytest
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from qkeras import QScaleShift
from qkeras.utils import load_qmodel


def create_qmac_model(layer_cls,
                      kwargs=None,
                      input_data=None,
                      weight_data=None):
  """Create a QMAC model for test purpose."""
  layer = layer_cls(**kwargs)
  x = Input(shape=input_data.shape[1:], dtype=input_data.dtype)
  y = layer(x)
  layer.set_weights(weight_data)

  return Model(x, y)


@pytest.mark.parametrize(
    'layer_kwargs, input_data, weight_data, bias_data, expected_output',
    [
        (
            {
                'weight_quantizer': 'quantized_bits(8,2,alpha=1.0)',
                'bias_quantizer': 'quantized_bits(8,2,alpha=1.0)',
                'activation': 'quantized_bits(8,4,alpha=1.0)'
            },
            np.array([[1, 1], [2, 2]], dtype=K.floatx()),
            np.array([[1.0]]),
            np.array([[4.0]]),
            np.array([[5, 5], [6, 6]], dtype=K.floatx())),
    ])
def test_qmac(layer_kwargs, input_data, weight_data, bias_data,
              expected_output):
  model = create_qmac_model(
      layer_cls=QScaleShift,
      kwargs=layer_kwargs,
      input_data=input_data,
      weight_data=[weight_data, bias_data])

  actual_output = model.predict(input_data)
  assert_allclose(actual_output, expected_output, rtol=1e-4)

  # Test model loading and saving.
  fd, fname = tempfile.mkstemp('.h5')
  model.save(fname)

  # Load the model.
  loaded_model = load_qmodel(fname)

  # Clean the h5 file after loading the model
  os.close(fd)
  os.remove(fname)

  # Compare weights of original and loaded models.
  model_weights = model.weights
  loaded_model_weights = loaded_model.weights

  assert_equal(len(model_weights), len(loaded_model_weights))
  for i, model_weight in enumerate(model_weights):
    assert_equal(model_weight.numpy(), loaded_model_weights[i].numpy())

  # Compare if loaded models have the same prediction as original models.
  loaded_model_output = loaded_model.predict(input_data)
  assert_equal(actual_output, loaded_model_output)


if __name__ == '__main__':
  pytest.main([__file__])
