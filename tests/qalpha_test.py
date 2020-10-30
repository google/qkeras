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
"""Test get_weight_scale function with auto and auto_po2 modes of quantizers.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import logging
from numpy.testing import assert_allclose
from numpy.testing import assert_equal
import pytest
from tensorflow.keras import backend as K
from qkeras import binary
from qkeras import get_weight_scale
from qkeras import ternary
from qkeras.quantizers import _get_integer_bits


# expected value if input is uniform distribution is:
#   - alpha = m/2.0 for binary
#   - alpha = (m+d)/2.0 for ternary


def test_binary_auto():
  """Test binary auto scale quantizer."""

  np.random.seed(42)
  N = 1000000
  m_list = [1.0, 0.1, 0.01, 0.001]

  for m in m_list:
    x = np.random.uniform(-m, m, (N, 10)).astype(K.floatx())
    x = K.constant(x)

    quantizer = binary(alpha="auto")
    q = K.eval(quantizer(x))

    result = get_weight_scale(quantizer, q)
    expected = m / 2.0
    logging.info("expect %s", expected)
    logging.info("result %s", result)
    assert_allclose(result, expected, rtol=0.02)


def test_binary_auto_po2():
  """Test binary auto_po2 scale quantizer."""

  np.random.seed(42)
  N = 1000000
  m_list = [1.0, 0.1, 0.01, 0.001]

  for m in m_list:
    x = np.random.uniform(-m, m, (N, 10)).astype(K.floatx())
    x = K.constant(x)

    quantizer_ref = binary(alpha="auto")
    quantizer = binary(alpha="auto_po2")

    q_ref = K.eval(quantizer_ref(x))
    q = K.eval(quantizer(x))

    ref = get_weight_scale(quantizer_ref, q_ref)

    expected = np.power(2.0, np.round(np.log2(ref)))
    result = get_weight_scale(quantizer, q)

    assert_allclose(result, expected, rtol=0.0001)


def test_ternary_auto():
  """Test ternary auto scale quantizer."""

  np.random.seed(42)
  N = 1000000
  m_list = [1.0, 0.1, 0.01, 0.001]

  for m in m_list:
    x = np.random.uniform(-m, m, (N, 10)).astype(K.floatx())
    x = K.constant(x)

    quantizer = ternary(alpha="auto")
    q = K.eval(quantizer(x))

    d = m/3.0
    result = np.mean(get_weight_scale(quantizer, q))
    expected = (m + d) / 2.0
    assert_allclose(result, expected, rtol=0.02)


def test_ternary_auto_po2():
  """Test ternary auto_po2 scale quantizer."""

  np.random.seed(42)
  N = 1000000
  m_list = [1.0, 0.1, 0.01, 0.001]

  for m in m_list:
    x = np.random.uniform(-m, m, (N, 10)).astype(K.floatx())
    x = K.constant(x)

    quantizer_ref = ternary(alpha="auto")
    quantizer = ternary(alpha="auto_po2")

    q_ref = K.eval(quantizer_ref(x))
    q = K.eval(quantizer(x))

    ref = get_weight_scale(quantizer_ref, q_ref)

    expected = np.power(2.0, np.round(np.log2(ref)))
    result = get_weight_scale(quantizer, q)

    assert_allclose(result, expected, rtol=0.0001)


def test_get_integer_bits():
  """Test automated integer bit (po2 scale) estimator."""

  bits = 4
  min_value = np.array([
      -4.0, -4.0, -4.0, -4.0, 1.0, -3.0, -10.0, -16, -25, 0, 0, 0, 0.1, 0.0,
      -1.0, 0.0, 0.0, 0.0, 0, 0, 0
  ])
  max_value = np.array([
      3.5, 3.51, 3.75, 3.751, 2.0, 4.0, 5.0, 8, 0, 0, 0.1, 0.999, 0.5, 0.8751,
      0.9375, 0.93751, 1.875, 1.8751, 9, 11, 12
  ])

  # unsigned number (keep_negative=False) without clippling.
  symmetric = False  # symmetric is irrelevant.
  keep_negative = False
  is_clipping = False
  integer_bits = _get_integer_bits(
      min_value=min_value,
      max_value=max_value,
      bits=bits,
      symmetric=symmetric,
      keep_negative=keep_negative,
      is_clipping=is_clipping)
  assert_equal(
      integer_bits,
      np.array([2, 2, 2, 3, 2, 3, 3, 4, 0, 0, 0, 1, 0, 0, 0, 1, 1, 2, 4, 4, 4]))

  # unsigned number (keep_negative=False) with clippling.
  symmetric = False  # symmetric is irrelevant.
  keep_negative = False
  is_clipping = True
  integer_bits = _get_integer_bits(
      min_value=min_value,
      max_value=max_value,
      bits=bits,
      symmetric=symmetric,
      keep_negative=keep_negative,
      is_clipping=is_clipping)
  assert_equal(
      integer_bits,
      np.array([2, 2, 2, 2, 1, 2, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 3, 3, 4]))

  # signed number (keep_negative=True) non-symmetric without clippling
  symmetric = False
  keep_negative = True
  is_clipping = False
  integer_bits = _get_integer_bits(
      min_value=min_value,
      max_value=max_value,
      bits=bits,
      symmetric=symmetric,
      keep_negative=keep_negative,
      is_clipping=is_clipping)
  assert_equal(
      integer_bits,
      np.array([2, 3, 3, 3, 2, 3, 3, 3, 3, 0, 0, 1, 0, 1, 1, 1, 2, 2, 3, 3, 3]))

  # signed number (keep_negative=True) non-symmetric with clippling
  symmetric = False
  keep_negative = True
  is_clipping = True
  integer_bits = _get_integer_bits(
      min_value=min_value,
      max_value=max_value,
      bits=bits,
      symmetric=symmetric,
      keep_negative=keep_negative,
      is_clipping=is_clipping)
  assert_equal(
      integer_bits,
      np.array([2, 2, 2, 2, 1, 2, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 1, 1, 3, 3, 3]))

  # signed number (keep_negative=True) symmetric without clippling
  symmetric = True
  keep_negative = True
  is_clipping = False
  integer_bits = _get_integer_bits(
      min_value=min_value,
      max_value=max_value,
      bits=bits,
      symmetric=symmetric,
      keep_negative=keep_negative,
      is_clipping=is_clipping)
  assert_equal(
      integer_bits,
      np.array([3, 3, 3, 3, 2, 3, 3, 3, 3, 0, 0, 1, 0, 1, 1, 1, 2, 2, 3, 3, 3]))

  # signed number (keep_negative=True) symmetric with clippling
  symmetric = True
  keep_negative = True
  is_clipping = True
  integer_bits = _get_integer_bits(
      min_value=min_value,
      max_value=max_value,
      bits=bits,
      symmetric=symmetric,
      keep_negative=keep_negative,
      is_clipping=is_clipping)
  assert_equal(
      integer_bits,
      np.array([2, 2, 2, 2, 1, 2, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 1, 1, 3, 3, 3]))


if __name__ == "__main__":
  pytest.main([__file__])
