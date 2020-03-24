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
import pytest
from tensorflow.keras import backend as K
from qkeras import binary
from qkeras import get_weight_scale
from qkeras import ternary


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


if __name__ == "__main__":
  pytest.main([__file__])
