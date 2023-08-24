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
"""Tests min/max values that are used for autorange."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
from qkeras import *
from tensorflow.keras import backend as K


def test_binary():
  q = binary(alpha=1.0)
  assert q.min() == -1.0
  assert q.max() == 1.0

  q = stochastic_binary(alpha=1.0)
  assert q.min() == -1.0
  assert q.max() == 1.0


def test_ternary():
  q = ternary(alpha=1.0)
  assert q.min() == -1.0
  assert q.max() == 1.0

  q = stochastic_ternary(alpha=1.0)
  assert q.min() == -1.0
  assert q.max() == 1.0


def test_quantized_bits():
  results = {
      (1,0): [-1.0, 1.0],
      (2,0): [-1.0, 1.0],
      (3,0): [-1.0, 1.0],
      (4,0): [-1.0, 1.0],
      (5,0): [-1.0, 1.0],
      (6,0): [-1.0, 1.0],
      (7,0): [-1.0, 1.0],
      (8,0): [-1.0, 1.0],
      (1,1): [-1.0, 1.0],
      (2,1): [-2.0, 2.0],
      (3,1): [-2.0, 2.0],
      (4,1): [-2.0, 2.0],
      (5,1): [-2.0, 2.0],
      (6,1): [-2.0, 2.0],
      (7,1): [-2.0, 2.0],
      (8,1): [-2.0, 2.0],
      (3,2): [-4.0, 4.0],
      (4,2): [-4.0, 4.0],
      (5,2): [-4.0, 4.0],
      (6,2): [-4.0, 4.0],
      (7,2): [-4.0, 4.0],
      (8,2): [-4.0, 4.0],
  }

  for i in range(3):
    for b in range(1,9):
      if b <= i: continue
      q = quantized_bits(b,i,1)
      expected = results[(b,i)]
      assert expected[0] == q.min()
      assert expected[1] == q.max()


@pytest.mark.parametrize('alpha', [None, 2.0])
@pytest.mark.parametrize('symmetric,keep_negative', 
                         [(True, True), (False, True), (False, False)])
@pytest.mark.parametrize('bits', [1, 8])
def test_quantized_linear(bits, symmetric, keep_negative, alpha):

  q = quantized_linear(bits=bits, 
                        symmetric=symmetric, 
                        keep_negative=keep_negative, 
                        alpha=alpha)
  assert q(-1000) == q.min()
  assert q(1000)== q.max()
  assert q(q.min()) == q.min()
  assert q(q.max()) == q.max()
  if bits != 1:
    middle_point = (q.max() + q.min()) / 2.0
    assert q(middle_point) != q.max()
    assert q(middle_point) != q.min()

def test_po2():
  po2 = {
    3: [-2, 2],
    4: [-8, 8],
    5: [-128, 128],
    6: [-32768, 32768]
  }

  po2_max_value = {
      (3,1): [-1.0, 1.0],
      (3,2): [-2, 2],
      (3,4): [-4, 4],
      (4,1): [-1.0, 1.0],
      (4,2): [-2, 2],
      (4,4): [-4, 4],
      (4,8): [-8, 8],
      (5,1): [-1.0, 1.0],
      (5,2): [-2, 2],
      (5,4): [-4, 4],
      (5,8): [-8, 8],
      (5,16): [-16, 16],
      (6,1): [-1.0, 1.0],
      (6,2): [-2, 2],
      (6,4): [-4, 4],
      (6,8): [-8, 8],
      (6,16): [-16, 16],
      (6,32): [-32, 32]
  }

  po2_quadratic = {
    4: [-4, 4],
    5: [-64, 64],
    6: [-16384, 16384]
  }

  relu_po2_quadratic = {
    4: [0.00390625, 64],
    5: [1.52587890625e-05, 16384],
    6: [2.3283064365386963e-10, 1073741824]
  }

  for b in range(3,7):
    q = quantized_po2(b)
    assert po2[b][0] == q.min()
    assert po2[b][1] == q.max()
    for i in range(0,b):
      q = quantized_po2(b,2**i)
      assert po2_max_value[(b,2**i)][0] == q.min()
      assert po2_max_value[(b,2**i)][1] == q.max()

  for b in range(4,7):
    q = quantized_po2(b,quadratic_approximation=True)
    assert po2_quadratic[b][0] == q.min()
    assert po2_quadratic[b][1] == q.max()
    q = quantized_relu_po2(b,quadratic_approximation=True)
    assert relu_po2_quadratic[b][0] == q.min()
    assert relu_po2_quadratic[b][1] == q.max()

if __name__ == "__main__":
  pytest.main([__file__])
