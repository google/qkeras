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

from qkeras import quantized_bits
from qkeras.codebook import weight_compression


@pytest.mark.parametrize(
  'bits, axis, quantizer, weights, expected_result',
  [
    (
      3, 3, quantized_bits(4, 0, 1, alpha='auto_po2'),
      np.array([
       [[ 0.14170583, -0.34360626,  0.29548156],
        [ 0.6517242,  0.06870092, -0.21646781],
        [ 0.12486842, -0.05406165, -0.23690471]],

       [[-0.07540564,  0.2123149 ,  0.2382695 ],
        [ 0.78434753,  0.36171672, -0.43612534],
        [ 0.3685556,  0.41328752, -0.48990643]],

      [[-0.04438099,  0.0590747 , -0.0644061 ],
        [ 0.15280165,  0.40714318, -0.04622072],
        [ 0.21560416, -0.22131851, -0.5365659 ]]], dtype=np.float32),
      np.array([
       [[ 0.125 , -0.375 ,  0.25  ],
        [ 0.75  ,  0.125 , -0.25  ],
        [ 0.125 ,  0.0   , -0.25  ]],

       [[ 0.0   ,  0.25  ,  0.25  ],
        [ 0.75  ,  0.375 , -0.375 ],
        [ 0.375 ,  0.375 , -0.5   ]],

       [[ 0.0   ,  0.0   ,  0.0   ],
        [ 0.125 ,  0.375 ,  0.0   ],
        [ 0.25  , -0.25  , -0.5   ]]], dtype=np.float32)
    )
  ]
)
def test_codebook_weights(bits, axis, quantizer, weights, expected_result):
  np.random.seed(22)
  weights = weights.reshape(weights.shape + (1,))
  expected_result = expected_result.reshape(expected_result.shape + (1,))
  index_table, codebook_table = weight_compression(weights,
                                                   bits,
                                                   axis,
                                                   quantizer)
  new_weights = np.zeros(weights.shape)
  for i in range(weights.shape[axis]):
    new_weights[:, :, :, i] = codebook_table[i][index_table[:, :, :, i]]

  assert_allclose(new_weights, expected_result, rtol=1e-4)


if __name__ == '__main__':
  pytest.main([__file__])
