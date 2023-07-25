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
"""Test gradual quantization noise injection with quantizers of quantizers.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import logging
from numpy.testing import assert_allclose
from numpy.testing import assert_equal
import pytest
from tensorflow.keras import backend as K
from qkeras.quantizers import quantized_bits
from qkeras.quantizers import quantized_linear
from qkeras.quantizers import quantized_relu


@pytest.mark.parametrize('quantizer', [quantized_bits, quantized_linear])
def test_qnoise_linear_quantizer(quantizer):
  """Tests for quantized_bits and quantized_linear."""
  
  # 1 sign bit, 1 integer bit, and 2 fractional bits.
  bits = 4
  integer = 1
  symmetric = True
  keep_negative = True
  alpha = 1
  use_stochastic_rounding = False

  q = quantizer(
      bits=bits,
      integer=integer,
      symmetric=symmetric,
      keep_negative=keep_negative,
      alpha=alpha,
      use_stochastic_rounding=use_stochastic_rounding,
      use_variables=True,
  )

  inputs = np.array([0.0, 0.5, -0.5, 0.6, -0.6, 2.0, -2.0], dtype=np.float32)
  x = np.array([0.0, 0.5, -0.5, 0.6, -0.6, 2.0, -2.0], dtype=np.float32)
  xq = np.array([0.0, 0.5, -0.5, 0.5, -0.5, 1.75, -1.75], dtype=np.float32)
  x_xq = 0.5 * (x + xq)

  # no quantization
  q.update_qnoise_factor(0.0)
  x_q_0 = q(inputs)
  assert_equal(x_q_0, x)

  # full quantization
  q.update_qnoise_factor(1.0)
  x_q_1 = q(inputs)
  assert_equal(x_q_1, xq)

  # mixing half and half of x and gxq
  q.update_qnoise_factor(0.5)
  x_q_05 = q(inputs)
  assert_equal(x_q_05, x_xq)


def test_qnoise_quantized_relu():
  # 0 sign bit, 1 integer bit, and 3 fractional bits.
  bits = 4
  integer = 1
  use_sigmoid = False
  negative_slope = 0
  use_stochastic_rounding = False

  # input to quantized relu
  inputs = np.array([0.0, 0.5, -0.5, 0.6, 2.0, 3.0], dtype=np.float32)
  # float relu
  x = np.array([0.0, 0.5, 0.0, 0.6, 2.0, 3.0], dtype=np.float32)
  # float relu with upper bound 1.5
  x_ub = np.array([0.0, 0.5, 0.0, 0.6, 1.5, 1.5], dtype=np.float32)
  # float relu with quantized clipping
  x_clipped = np.array([0.0, 0.5, 0.0, 0.6, 1.875, 1.875], dtype=np.float32)
  # quantized relu
  xq = np.array([0.0, 0.5, 0.0, 0.625, 1.875, 1.875], dtype=np.float32)

  # mixing half and half
  x_xq = 0.5 * (x + xq)
  x_clipped_xq = 0.5 * (x_clipped + xq)
  x_ub_xq = 0.5 * (x_ub + xq)

  #########################################
  # No relu upper bound
  # No quantized clip for float relu
  #########################################
  qr_qc_false = quantized_relu(
      bits=bits,
      integer=integer,
      use_sigmoid=use_sigmoid,
      negative_slope=negative_slope,
      use_stochastic_rounding=use_stochastic_rounding,
      relu_upper_bound=None,
      is_quantized_clip=False,
      use_variables=True)
  # no quantization
  qr_qc_false.update_qnoise_factor(qnoise_factor=0.0)
  x_q_0 = qr_qc_false(inputs)
  assert_equal(x_q_0, x)

  # full quantization
  qr_qc_false.update_qnoise_factor(qnoise_factor=1.0)
  x_q_1 = qr_qc_false(inputs)
  assert_equal(x_q_1, xq)

  # mixing half and half
  qr_qc_false.update_qnoise_factor(qnoise_factor=0.5)
  x_q_05 = qr_qc_false(inputs)
  assert_equal(x_q_05, x_xq)

  #########################################
  # No relu upper bound
  # Quantized clip for float relu
  #########################################
  qr_qc_true = quantized_relu(
      bits=bits,
      integer=integer,
      use_sigmoid=use_sigmoid,
      negative_slope=negative_slope,
      use_stochastic_rounding=use_stochastic_rounding,
      relu_upper_bound=None,
      is_quantized_clip=True,
      use_variables=True)
  # no quantization
  qr_qc_true.update_qnoise_factor(qnoise_factor=0.0)
  x_q_0 = qr_qc_true(inputs)
  assert_equal(x_q_0, x_clipped)

  # full quantization
  qr_qc_true.update_qnoise_factor(qnoise_factor=1.0)
  x_q_1 = qr_qc_true(inputs)
  assert_equal(x_q_1, xq)

  # mixing half and half
  qr_qc_true.update_qnoise_factor(qnoise_factor=0.5)
  x_q_05 = qr_qc_true(inputs)
  assert_equal(x_q_05, x_clipped_xq)

  #########################################
  # Relu upper bound
  # No quantized clip for float relu
  #########################################
  qr_ub_qc_false = quantized_relu(
      bits=bits,
      integer=integer,
      use_sigmoid=use_sigmoid,
      negative_slope=negative_slope,
      use_stochastic_rounding=use_stochastic_rounding,
      relu_upper_bound=1.5,
      is_quantized_clip=False,
      use_variables=True)
  # no quantization
  qr_ub_qc_false.update_qnoise_factor(qnoise_factor=0.0)
  x_q_0 = qr_ub_qc_false(inputs)
  assert_equal(x_q_0, np.clip(x_ub, a_min=None, a_max=1.5))

  # full quantization
  qr_ub_qc_false.update_qnoise_factor(qnoise_factor=1.0)
  x_q_1 = qr_ub_qc_false(inputs)
  assert_equal(x_q_1, np.clip(xq, a_min=None, a_max=1.5))

  # mixing half and half
  qr_ub_qc_false.update_qnoise_factor(qnoise_factor=0.5)
  x_q_05 = qr_ub_qc_false(inputs)
  assert_equal(x_q_05, np.clip(x_ub_xq, a_min=None, a_max=1.5))

  #########################################
  # Relu upper bound
  # Quantized clip for float relu
  # (The quantized clip has precedence over the relu upper bound.)
  #########################################
  qr_ub_qc_true = quantized_relu(
      bits=bits,
      integer=integer,
      use_sigmoid=use_sigmoid,
      negative_slope=negative_slope,
      use_stochastic_rounding=use_stochastic_rounding,
      relu_upper_bound=1.5,
      is_quantized_clip=True,
      use_variables=True)
  # no quantization
  qr_ub_qc_true.update_qnoise_factor(qnoise_factor=0.0)
  x_q_0 = qr_ub_qc_true(inputs)
  assert_equal(x_q_0, x_clipped)

  # full quantization
  qr_ub_qc_true.update_qnoise_factor(qnoise_factor=1.0)
  x_q_1 = qr_ub_qc_true(inputs)
  assert_equal(x_q_1, xq)

  # mixing half and half
  qr_ub_qc_true.update_qnoise_factor(qnoise_factor=0.5)
  x_q_05 = qr_ub_qc_true(inputs)
  assert_equal(x_q_05, x_clipped_xq)


if __name__ == "__main__":
  pytest.main([__file__])
