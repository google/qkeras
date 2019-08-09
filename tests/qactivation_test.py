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
import numpy as np
from numpy.testing import assert_allclose

import pytest
from keras import backend as K

from qkeras import binary
from qkeras import hard_sigmoid
from qkeras import smooth_sigmoid
from qkeras import quantized_bits
from qkeras import quantized_relu
from qkeras import ternary


def test_smooth_sigmoid():
  """Test smooth_sigmoid function."""
  test_values = np.array(
      [[-3.0, -2.0, -1.0, -0.5, 0.005, 0.0, 0.005, 0.5, 1, 4, 10]],
      dtype=K.floatx())

  def ref_smooth_sigmoid(y):
    x = 0.1875 * y + 0.5
    z = 0.0 if x <= 0.0 else (1.0 if x >= 1.0 else x)
    return z

  sigmoid = np.vectorize(ref_smooth_sigmoid)
  x = K.placeholder(ndim=2)
  f = K.function([x], [smooth_sigmoid(x)])
  result = f([test_values])[0]
  expected = sigmoid(test_values)
  assert_allclose(result, expected, rtol=1e-05)


def test_hard_sigmoid():
  """Test hard_sigmoid function."""
  test_values = np.array(
      [[-3.0, -2.0, -1.0, -0.5, 0.005, 0.0, 0.005, 0.5, 1, 4, 10]],
      dtype=K.floatx())

  def ref_hard_sigmoid(y):
    x = 0.5 * y + 0.5
    z = 0.0 if x <= 0.0 else (1.0 if x >= 1.0 else x)
    return z

  sigmoid = np.vectorize(ref_hard_sigmoid)

  x = K.placeholder(ndim=2)
  f = K.function([x], [hard_sigmoid(x)])
  result = f([test_values])[0]
  expected = sigmoid(test_values)
  assert_allclose(result, expected, rtol=1e-05)


@pytest.mark.parametrize(
    'bits, integer, use_sigmoid, test_values, expected_values', [
        (
            6,
            2,
            0,
            np.array(
                [[-3.0, 0.0, 2.5625, 3.3671875, 1.5625, 1.046875, 0.054688]],
                dtype=K.floatx()),
            np.array([[0.0, 0.0, 2.5625, 3.375, 1.5625, 1.0625, 0.0625]],
                     dtype=K.floatx()),
        ),
        (6, 2, 1,
         np.array([[
             0.458069, 0.573227, 0.194336, 1.539047, 0.045883, 4.009995,
             3.962494, 3.937500, 0.363266, 0.875198, 0.710938, 4.000000,
             7.000000, 3.937500, 3.937592, 0.199326, 0.458008, 0.625977,
             0.544922, 1.046875, 0.586899, 3.367188, 3.804688, 0.312500,
             0.062500, 0.562500, 0.375000, 3.367188, 1.046875, 2.796875,
             0.054688, 1.562500, 2.562500
         ]],
                  dtype=K.floatx()),
         np.array([[
             0.500000, 0.625000, 0.250000, 1.500000, 0.000000, 3.937500,
             3.937500, 3.937500, 0.375000, 0.875000, 0.750000, 3.937500,
             3.937500, 3.937500, 3.937500, 0.250000, 0.500000, 0.625000,
             0.500000, 1.000000, 0.625000, 3.375000, 3.750000, 0.250000,
             0.000000, 0.500000, 0.375000, 3.375000, 1.000000, 2.750000,
             0.000000, 1.500000, 2.500000
         ]],
                  dtype=K.floatx())),
    ])
def test_quantized_relu(bits, integer, use_sigmoid, test_values, expected_values):
  """Test quantized_relu function."""
  x = K.placeholder(ndim=2)
  f = K.function([x], [quantized_relu(bits, integer, use_sigmoid)(x)])
  result = f([test_values])[0]
  assert_allclose(result, expected_values, rtol=1e-05)


@pytest.mark.parametrize(
    'bits, integer, symmetric, keep_negative, test_values, expected_values', [
        (
            6,
            2,
            0,
            1,
            np.array([[-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1, 4, 10]],
                     dtype=K.floatx()),
            np.array([[-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1, 3.875, 3.875]],
                     dtype=K.floatx()),
        ),
        (
            6,
            2,
            0,
            0,
            np.array([[-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1, 4, 10]],
                     dtype=K.floatx()),
            np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1, 3.9375, 3.9375]],
                     dtype=K.floatx()),
        ),
        (
            6,
            2,
            1,
            1,
            np.array([[-10, -4, -1.0, -0.5, 0.0, 0.5, 1, 4, 10]],
                     dtype=K.floatx()),
            np.array([[-3.875, -3.875, -1.0, -0.5, 0.0, 0.5, 1, 3.875, 3.875]],
                     dtype=K.floatx()),
        )
    ])
def test_quantized_bits(bits, integer, symmetric, keep_negative, test_values,
                        expected_values):
  x = K.placeholder(ndim=2)
  f = K.function([x],
                 [quantized_bits(bits, integer, symmetric, keep_negative)(x)])
  result = f([test_values])[0]
  assert_allclose(result, expected_values, rtol=1e-05)


@pytest.mark.parametrize(
    'alpha, threshold, test_values, expected_values', [
        (1.0, 0.33,
         np.array([[-3.0, -2.0, -1.0, -0.2, 0.0, 0.3, 1, 4, 10]],
                   dtype=K.floatx()),
         np.array([[-1.0, -1.0, -1.0, 0, 0.0, 0.0, 1, 1, 1]],
                   dtype=K.floatx())),
         (10.0, 5.0,
         np.array([[-11.0, -7.0, -4.0, -0.2, 0.0, 0.3, 1, 4, 10]],
                   dtype=K.floatx()),
         np.array([[-10.0, -10.0, 0.0, 0, 0.0, 0.0, 0, 0, 10]],
                   dtype=K.floatx())),

    ]
)
def test_ternary(alpha, threshold, test_values, expected_values):
  x = K.placeholder(ndim=2)
  f = K.function([x],
                 [ternary(alpha, threshold)(x)])
  result = f([test_values])[0]
  assert_allclose(result, expected_values, rtol=1e-05)


@pytest.mark.parametrize('use_01, alpha, test_values, expected_values', [
    (False, 1.0,
     np.array([[-3.0, -2.0, -1.0, -0.2, 0.0, 0.3, 1, 4, 10]], dtype=K.floatx()),
     np.array([[-1.0, -1.0, -1.0, -1.0, 1, 1, 1, 1, 1]], dtype=K.floatx())),
    (False, 5.0,
     np.array([[-11.0, -7.0, -4.0, -0.2, 0.0, 0.3, 1, 4, 10]],
              dtype=K.floatx()),
     np.array([[-5.0, -5.0, -5.0, -5, 5.0, 5.0, 5, 5, 5]], dtype=K.floatx())),
    (True, 5.0,
     np.array([[-11.0, -7.0, -4.0, -0.2, 0.0, 0.3, 1, 4, 10]],
              dtype=K.floatx()),
     np.array([[0, 0, 0, 0, 5, 5, 5, 5, 5]], dtype=K.floatx())),
])
def test_binary(use_01, alpha, test_values, expected_values):
  x = K.placeholder(ndim=2)
  f = K.function([x], [binary(use_01, alpha)(x)])
  result = f([test_values])[0]
  assert_allclose(result, expected_values, rtol=1e-05)


if __name__ == '__main__':
  pytest.main([__file__])
