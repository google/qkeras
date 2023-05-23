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
"""Test range values that are used for codebook computation"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from numpy.testing import assert_allclose

import pytest
from tensorflow.keras import backend as K
import tensorflow as tf

from qkeras import quantized_relu
from qkeras import quantized_bits
from qkeras import quantized_linear


@pytest.mark.parametrize(
    'bits, integer, expected_values',
    [
        (3, 0, np.array([0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])),
        (3, 1, np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75])),
        (3, 2, np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])),
        (3, 3, np.array([0, 1, 2, 3, 4, 5, 6, 7])),
        (6, 1, np.array(
            [0.0, 0.03125, 0.0625, 0.09375, 0.125, 0.15625, 0.1875, 0.21875,
            0.25, 0.28125, 0.3125, 0.34375, 0.375, 0.40625, 0.4375, 0.46875,
            0.5, 0.53125, 0.5625, 0.59375, 0.625, 0.65625, 0.6875, 0.71875,
            0.75, 0.78125, 0.8125, 0.84375, 0.875, 0.90625, 0.9375, 0.96875,
            1.0, 1.03125, 1.0625, 1.09375, 1.125, 1.15625, 1.1875, 1.21875,
            1.25, 1.28125, 1.3125, 1.34375, 1.375, 1.40625, 1.4375, 1.46875,
            1.5, 1.53125, 1.5625, 1.59375, 1.625, 1.65625, 1.6875, 1.71875,
            1.75, 1.78125, 1.8125, 1.84375, 1.875, 1.90625, 1.9375, 1.96875]))
    ])
def test_quantized_relu_range(bits, integer, expected_values):
  """Test quantized_relu range function."""
  q = quantized_relu(bits, integer)
  result = q.range()
  assert_allclose(result, expected_values, rtol=1e-05)


@pytest.mark.parametrize(
    'bits, integer, expected_values',
    [
        (3, 0, np.array([0.0, 0.25, 0.5, 0.75, -1.0, -0.75, -0.5, -0.25])),
        (3, 1, np.array([0.0, 0.5, 1.0, 1.5, -2.0, -1.5, -1.0, -0.5])),
        (3, 2, np.array([0.0, 1.0, 2.0, 3.0, -4.0, -3.0, -2.0, -1.0])),
        (3, 3, np.array([0.0, 2.0, 4.0, 6.0, -8.0, -6.0, -4.0, -2.0])),
        (6, 1, np.array(
            [0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625,
             0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0, 1.0625, 1.125, 1.1875,
             1.25, 1.3125, 1.375, 1.4375, 1.5, 1.5625, 1.625, 1.6875, 1.75, 1.8125,
             1.875, 1.9375, -2.0, -1.9375, -1.875, -1.8125, -1.75, -1.6875, -1.625,
             -1.5625, -1.5, -1.4375, -1.375, -1.3125, -1.25, -1.1875, -1.125, -1.0625,
             -1.0, -0.9375, -0.875, -0.8125, -0.75, -0.6875, -0.625, -0.5625, -0.5,
             -0.4375, -0.375, -0.3125, -0.25, -0.1875, -0.125, -0.0625]))
    ])
def test_quantized_bits_range(bits, integer, expected_values):
  """Test quantized_bits range function."""
  q = quantized_bits(bits, integer)
  result = q.range()
  assert_allclose(result, expected_values, rtol=1e-05)

@pytest.mark.parametrize('alpha', [None, 2.0])
@pytest.mark.parametrize('symmetric,keep_negative', 
                         [(True, True), (False, True), (False, False)])
@pytest.mark.parametrize('bits', [1, 8])
def test_quantized_linear_range(bits, symmetric, keep_negative, alpha):
  """Test quantized_linear range function."""
  q = quantized_linear(bits, 0, symmetric=symmetric, keep_negative=keep_negative,
                       alpha=alpha)
  # compute output on array of inputs, and compare to q.range()
  x = np.linspace(-10.0, 10.0, 10 * 2**(bits + 1) + 1)
  y = q(x)
  q_range = q.range()
  # assert that y and q_range have the same set of values
  _assert_same_unique_values(q_range, y)
  # assert that values ordered on binary range asending
  _assert_binary_range_ordering(q_range)


def _assert_same_unique_values(x, y):
    """Check if two TensorFlow tensors have the same unique set of values."""
    # Get the unique values of each tensor
    unique_x = tf.unique(x)[0].numpy()
    unique_y = tf.unique(y)[0].numpy()

    # sort the unique values
    unique_x.sort()
    unique_y.sort()

    assert unique_x.shape == unique_y.shape
    assert np.allclose(unique_x, unique_y)


def _assert_binary_range_ordering(x):
  """Assert that x is ordered by binary representation ascending"""

  x = np.array(x)
  # get positive values in x
  x_pos = x[x >= 0]
  # get negative values in x
  x_neg = x[x < 0]
  # assert that positive values are ordered ascending
  assert np.all(np.diff(x_pos) >= 0)
  # assert that negative values are ordered ascending
  assert np.all(np.diff(x_neg) >= 0)
  # assert that all positive values come before negative values
  assert np.all(x == np.concatenate([x_pos, x_neg]))


if __name__ == "__main__":
  pytest.main([__file__])
