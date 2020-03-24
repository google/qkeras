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
import numpy as np
from numpy.testing import assert_allclose
import pytest
import logging
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.backend import clear_session

from qkeras import QActivation
from qkeras import QDense
from qkeras import quantized_bits
from qkeras.utils import model_save_quantized_weights
from qkeras.utils import quantized_model_from_json


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
    'layer_kwargs, input_data, weight_data, bias_data, expected_output',
    [
        (
            {
                'units': 2,
                'use_bias': True,
                'kernel_initializer': 'glorot_uniform',
                'bias_initializer': 'zeros'
            },
            np.array([[1, 1, 1, 1]], dtype=K.floatx()),
            np.array([[10, 20], [10, 20], [10, 20], [10, 20]],
                     dtype=K.floatx()),  # weight_data
            np.array([0, 0], dtype=K.floatx()),  # bias
            np.array([[40, 80]], dtype=K.floatx())),  # expected_output
        (
            {
                'units': 2,
                'use_bias': True,
                'kernel_initializer': 'glorot_uniform',
                'bias_initializer': 'zeros',
                'kernel_quantizer': 'quantized_bits(2,0,alpha=1.0)',
                'bias_quantizer': 'quantized_bits(2,0)',
            },
            np.array([[1, 1, 1, 1]], dtype=K.floatx()),
            np.array([[10, 20], [10, 20], [10, 20], [10, 20]],
                     dtype=K.floatx()),  # weight_data
            np.array([0, 0], dtype=K.floatx()),  # bias
            np.array([[2, 2]], dtype=K.floatx())),  #expected_output
    ])
def test_qdense(layer_kwargs, input_data, weight_data, bias_data,
                expected_output):
  qdense_util(
      layer_cls=QDense,
      kwargs=layer_kwargs,
      input_data=input_data,
      weight_data=[weight_data, bias_data],
      expected_output=expected_output)


if __name__ == '__main__':
  pytest.main([__file__])
