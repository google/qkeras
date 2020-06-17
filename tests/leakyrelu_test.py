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
"""Test activation from qlayers.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from numpy.testing import assert_allclose

import pytest
from tensorflow.keras import backend as K

from qkeras import quantized_relu


@pytest.mark.parametrize(
    'bits, integer, use_sigmoid, negative_slope, test_values, expected_values',
    [
        (6, 2, 0, 0.25,
         np.array(
             [[-3.0, -2.0, -1.0, 0.0, 2.5625, 3.3671875, 1.5625, 1.046875,
               0.054688, 6.0]],
             dtype=K.floatx()),
         np.array([[-0.75, -0.5, -0.25, 0.0, 2.5, 3.375, 1.5, 1.0, 0.0, 3.875]],
             dtype=K.floatx()),
        ),
        (6, 2, 1, 0.125,
         np.array([[
             0.458069, 0.573227, 0.194336, 1.539047, 0.045883, 4.009995,
             3.962494, 3.937500, 0.363266, 0.875198, 0.710938, 4.000000,
             7.000000, 3.937500, 3.937592, 0.199326, 0.458008, 0.625977,
             0.544922, 1.046875, 0.586899, 3.367188, 3.804688, 0.312500,
             0.062500, 0.562500, 0.375000, 3.367188, 1.046875, 2.796875,
             0.054688, 1.562500, 2.562500
         ]], dtype=K.floatx()),
         np.array([[
             0.5  , 0.5  , 0.25 , 1.5  , 0.   , 3.875, 3.875, 3.875, 0.25 ,
             1.   , 0.75 , 3.875, 3.875, 3.875, 3.875, 0.25 , 0.5  , 0.75 ,
             0.5  , 1.   , 0.5  , 3.25 , 3.75 , 0.25 , 0.   , 0.5  , 0.5  ,
             3.25 , 1.   , 2.75 , 0.   , 1.5  , 2.5
         ]], dtype=K.floatx())),
        (6, 2, 1, 0.125,
         np.array([[
             -0.458069, -0.573227, -0.194336, -1.539047, -0.045883, -4.009995,
             -3.962494, -3.937500, -0.363266, -0.875198, -0.710938, -4.000000,
             -7.000000, -3.937500, -3.937592, -0.199326, -0.458008, -0.625977,
             -0.544922, -1.046875, -0.586899, -3.367188, -3.804688, -0.312500,
             -0.062500, -0.562500, -0.375000, -3.367188, -1.046875, -2.796875,
             -0.054688, -1.562500, -2.562500
         ]], dtype=K.floatx()),
         np.array([[
              0.0,       0.0,       0.0,      -0.25,      0.0,      -0.5,
             -0.5,      -0.5,       0.0,       0.0,       0.0,      -0.5,
             -0.5,      -0.5,      -0.5,       0.0,       0.0,       0.0,
              0.0,      -0.25,      0.0,      -0.5,      -0.5,       0.0,
              0.0,       0.0,       0.0,      -0.5,      -0.25,     -0.25,
              0.0,      -0.25,     -0.25
         ]], dtype=K.floatx())),
    ])
def test_quantized_relu(bits, integer, use_sigmoid, negative_slope, test_values,
                        expected_values):
  """Test quantized_relu function."""
  x = K.placeholder(ndim=2)
  f = K.function([x], [quantized_relu(bits, integer, use_sigmoid,
                                      negative_slope)(x)])
  result = f([test_values])[0]
  assert_allclose(result, expected_values, rtol=1e-05)


if __name__ == '__main__':
  pytest.main([__file__])
