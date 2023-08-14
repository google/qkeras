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
"""Tests for methods in quantizer_impl.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from qkeras import *
from qkeras.qtools.quantized_operators import quantizer_impl
from qkeras import quantizers
from numpy.testing import assert_equal


# pylint: disable=invalid-name
def test_QuantizedBits():
  qkeras_quantizer = quantizers.quantized_bits()
  qtools_quantizer = quantizer_impl.QuantizedBits()
  qtools_quantizer.convert_qkeras_quantizer(qkeras_quantizer)
  new_quantizer = qtools_quantizer.convert_to_qkeras_quantizer(
      symmetric=qkeras_quantizer.symmetric, alpha=qkeras_quantizer.alpha,
      use_stochastic_rounding=qkeras_quantizer.use_stochastic_rounding,
      scale_axis=qkeras_quantizer.scale_axis,
      qnoise_factor=qkeras_quantizer.qnoise_factor)

  result = new_quantizer.__dict__
  for (key, val) in result.items():
    assert_equal(val, qkeras_quantizer.__dict__[key])


def test_QuantizedTanh():
  qkeras_quantizer = quantizers.quantized_tanh()
  qtools_quantizer = quantizer_impl.QuantizedTanh()
  qtools_quantizer.convert_qkeras_quantizer(qkeras_quantizer)
  new_quantizer = qtools_quantizer.convert_to_qkeras_quantizer(
      use_stochastic_rounding=qkeras_quantizer.use_stochastic_rounding,
      symmetric=qkeras_quantizer.symmetric)

  result = new_quantizer.__dict__
  for (key, val) in result.items():
    assert_equal(val, qkeras_quantizer.__dict__[key])


def test_QuantizedUlaw():
  qkeras_quantizer = quantizers.quantized_ulaw()
  qtools_quantizer = quantizer_impl.QuantizedUlaw()
  qtools_quantizer.convert_qkeras_quantizer(qkeras_quantizer)
  new_quantizer = qtools_quantizer.convert_to_qkeras_quantizer(
      symmetric=qkeras_quantizer.symmetric,
      u=qkeras_quantizer.u)

  result = new_quantizer.__dict__
  for (key, val) in result.items():
    assert_equal(val, qkeras_quantizer.__dict__[key])


def test_Binary():
  qkeras_quantizer = quantizers.binary()
  qtools_quantizer = quantizer_impl.Binary()
  qtools_quantizer.convert_qkeras_quantizer(qkeras_quantizer)
  new_quantizer = qtools_quantizer.convert_to_qkeras_quantizer(
      alpha=qkeras_quantizer.alpha,
      use_stochastic_rounding=qkeras_quantizer.use_stochastic_rounding)

  result = new_quantizer.__dict__
  for (key, val) in result.items():
    assert_equal(val, qkeras_quantizer.__dict__[key])


def test_StochasticBinary():
  qkeras_quantizer = quantizers.stochastic_binary()
  qtools_quantizer = quantizer_impl.StochasticBinary()
  qtools_quantizer.convert_qkeras_quantizer(qkeras_quantizer)
  new_quantizer = qtools_quantizer.convert_to_qkeras_quantizer(
      alpha=qkeras_quantizer.alpha,
      temperature=qkeras_quantizer.temperature,
      use_real_sigmoid=qkeras_quantizer.use_real_sigmoid)

  result = new_quantizer.__dict__
  for (key, val) in result.items():
    assert_equal(val, qkeras_quantizer.__dict__[key])


def test_Bernoulli():
  qkeras_quantizer = quantizers.bernoulli()
  qtools_quantizer = quantizer_impl.Bernoulli()
  qtools_quantizer.convert_qkeras_quantizer(qkeras_quantizer)
  new_quantizer = qtools_quantizer.convert_to_qkeras_quantizer(
      alpha=qkeras_quantizer.alpha, temperature=qkeras_quantizer.temperature,
      use_real_sigmoid=qkeras_quantizer.use_real_sigmoid)

  result = new_quantizer.__dict__
  for (key, val) in result.items():
    assert_equal(val, qkeras_quantizer.__dict__[key])


def test_QuantizedRelu():
  qkeras_quantizer = quantizers.quantized_relu()
  qtools_quantizer = quantizer_impl.QuantizedRelu()
  qtools_quantizer.convert_qkeras_quantizer(qkeras_quantizer)
  new_quantizer = qtools_quantizer.convert_to_qkeras_quantizer(
      use_sigmoid=qkeras_quantizer.use_sigmoid,
      negative_slope=qkeras_quantizer.negative_slope,
      use_stochastic_rounding=qkeras_quantizer.use_stochastic_rounding,
      relu_upper_bound=qkeras_quantizer.relu_upper_bound,
      is_quantized_clip=qkeras_quantizer.is_quantized_clip,
      qnoise_factor=qkeras_quantizer.qnoise_factor)

  result = new_quantizer.__dict__
  for (key, val) in result.items():
    assert_equal(val, qkeras_quantizer.__dict__[key])


def test_Ternary():

  qkeras_quantizer = quantizers.ternary()
  qtools_quantizer = quantizer_impl.Ternary()
  qtools_quantizer.convert_qkeras_quantizer(qkeras_quantizer)
  new_quantizer = qtools_quantizer.convert_to_qkeras_quantizer(
      alpha=qkeras_quantizer.alpha, threshold=qkeras_quantizer.threshold,
      use_stochastic_rounding=qkeras_quantizer.use_stochastic_rounding,
      number_of_unrolls=qkeras_quantizer.number_of_unrolls)

  result = new_quantizer.__dict__
  for (key, val) in result.items():
    assert_equal(val, qkeras_quantizer.__dict__[key])


def test_StochasticTernary():
  qkeras_quantizer = quantizers.stochastic_ternary()
  qtools_quantizer = quantizer_impl.StochasticTernary()
  qtools_quantizer.convert_qkeras_quantizer(qkeras_quantizer)
  new_quantizer = qtools_quantizer.convert_to_qkeras_quantizer(
      alpha=qkeras_quantizer.alpha, threshold=qkeras_quantizer.threshold,
      temperature=qkeras_quantizer.temperature,
      use_real_sigmoid=qkeras_quantizer.use_real_sigmoid,
      number_of_unrolls=qkeras_quantizer.number_of_unrolls)

  result = new_quantizer.__dict__
  for (key, val) in result.items():
    assert_equal(val, qkeras_quantizer.__dict__[key])


def test_PowerOfTwo():
  qkeras_quantizer = quantizers.quantized_po2()
  qtools_quantizer = quantizer_impl.PowerOfTwo(is_signed=True)
  qtools_quantizer.convert_qkeras_quantizer(qkeras_quantizer)
  new_quantizer = qtools_quantizer.convert_to_qkeras_quantizer(
      negative_slope=None,
      use_stochastic_rounding=qkeras_quantizer.use_stochastic_rounding,
      quadratic_approximation=qkeras_quantizer.quadratic_approximation)

  result = new_quantizer.__dict__
  for (key, val) in result.items():
    assert_equal(val, qkeras_quantizer.__dict__[key])


def test_ReluPowerOfTwo():
  qkeras_quantizer = quantizers.quantized_relu_po2()
  qtools_quantizer = quantizer_impl.ReluPowerOfTwo()
  qtools_quantizer.convert_qkeras_quantizer(qkeras_quantizer)
  new_quantizer = qtools_quantizer.convert_to_qkeras_quantizer(
      negative_slope=qkeras_quantizer.negative_slope,
      use_stochastic_rounding=qkeras_quantizer.use_stochastic_rounding,
      quadratic_approximation=qkeras_quantizer.quadratic_approximation)

  result = new_quantizer.__dict__
  for (key, val) in result.items():
    assert_equal(val, qkeras_quantizer.__dict__[key])


def test_GetScale_PerChannelScale():
  # Rank1 tensors
  x_r1 = tf.ones([4])
  q_r1 = tf.ones([4])
  scale_r1_pcs_true = quantizers._get_least_squares_scale(
      "auto", x_r1, q_r1, scale_axis=None, per_channel_scale=True)
  scale_r1_pcs_false = quantizers._get_least_squares_scale(
      "auto", x_r1, q_r1, scale_axis=None, per_channel_scale=False)
  assert_equal(tf.shape(scale_r1_pcs_true).numpy(), [4])
  assert_equal(tf.shape(scale_r1_pcs_false).numpy(), [1])

  # Rank2 tensors
  x_r2 = tf.ones([2, 4])
  q_r2 = tf.ones([2, 4])
  scale_r2_pcs_true = quantizers._get_least_squares_scale(
      "auto", x_r2, q_r2, scale_axis=None, per_channel_scale=True)
  scale_r2_pcs_false = quantizers._get_least_squares_scale(
      "auto", x_r2, q_r2, scale_axis=None, per_channel_scale=False)
  assert_equal(tf.shape(scale_r2_pcs_true).numpy(), [1, 4])
  assert_equal(tf.shape(scale_r2_pcs_false).numpy(), [1, 1])

  # Rank3 tensors
  x_r3 = tf.ones([3, 3, 4])
  q_r3 = tf.ones([3, 3, 4])
  scale_r3_pcs_true = quantizers._get_least_squares_scale(
      "auto", x_r3, q_r3, scale_axis=None, per_channel_scale=True)
  scale_r3_pcs_false = quantizers._get_least_squares_scale(
      "auto", x_r3, q_r3, scale_axis=None, per_channel_scale=False)
  assert_equal(tf.shape(scale_r3_pcs_true).numpy(), [1, 1, 4])
  assert_equal(tf.shape(scale_r3_pcs_false).numpy(), [1, 1, 1])

  # Rank4 tensors
  x_r4 = tf.ones([1, 1, 3, 4])
  q_r4 = tf.ones([1, 1, 3, 4])
  scale_r4_pcs_true = quantizers._get_least_squares_scale(
      "auto", x_r4, q_r4, scale_axis=None, per_channel_scale=True)
  scale_r4_pcs_false = quantizers._get_least_squares_scale(
      "auto", x_r4, q_r4, scale_axis=None, per_channel_scale=False)
  assert_equal(tf.shape(scale_r4_pcs_true).numpy(), [1, 1, 1, 4])
  assert_equal(tf.shape(scale_r4_pcs_false).numpy(), [1, 1, 1, 1])


def _get_num_unique_elements(input_tensor):
  return len(np.unique(input_tensor.numpy()))


def test_GetScale_ElementsPerScale_Scalar_ScaleAxis_EPS():
  # Test get_scale function when elements_per_scale and scale_axis have scalar
  # values and the input x and q tensors have rank 2
  x_r2 = tf.random.uniform([4, 8])
  q_r2 = tf.random.uniform([4, 8])
  scale_r2_eps_none_ua_none = quantizers._get_least_squares_scale(
      "auto", x_r2, q_r2, elements_per_scale=None, scale_axis=None)
  scale_r2_eps_2_ua_0 = quantizers._get_least_squares_scale(
      "auto", x_r2, q_r2, elements_per_scale=2, scale_axis=0)
  scale_r2_eps_2_ua_1 = quantizers._get_least_squares_scale(
      "auto", x_r2, q_r2, elements_per_scale=2, scale_axis=1)

  assert_equal(tf.shape(scale_r2_eps_none_ua_none).numpy(), [1, 8])
  assert_equal(_get_num_unique_elements(scale_r2_eps_none_ua_none), 8)

  assert_equal(tf.shape(scale_r2_eps_2_ua_0).numpy(), [4, 1])
  assert_equal(_get_num_unique_elements(scale_r2_eps_2_ua_0), 2)

  assert_equal(tf.shape(scale_r2_eps_2_ua_1).numpy(), [1, 8])
  assert_equal(_get_num_unique_elements(scale_r2_eps_2_ua_1), 4)

  # Test get_scale function when elements_per_scale and scale_axis have scalar
  # values and the input x and q tensors have rank 3
  x_r3 = tf.random.uniform([2, 4, 8])
  q_r3 = tf.random.uniform([2, 4, 8])
  scale_r3_eps_none_ua_none = quantizers._get_least_squares_scale(
      "auto", x_r3, q_r3, elements_per_scale=None, scale_axis=None)
  scale_r3_eps_2_ua_0 = quantizers._get_least_squares_scale(
      "auto", x_r3, q_r3, elements_per_scale=2, scale_axis=0)
  scale_r3_eps_2_ua_1 = quantizers._get_least_squares_scale(
      "auto", x_r3, q_r3, elements_per_scale=2, scale_axis=1)
  scale_r3_eps_2_ua_2 = quantizers._get_least_squares_scale(
      "auto", x_r3, q_r3, elements_per_scale=2, scale_axis=2)

  assert_equal(tf.shape(scale_r3_eps_none_ua_none).numpy(), [1, 1, 8])
  assert_equal(_get_num_unique_elements(scale_r3_eps_none_ua_none), 8)

  assert_equal(tf.shape(scale_r3_eps_2_ua_0).numpy(), [2, 1, 1])
  assert_equal(_get_num_unique_elements(scale_r3_eps_2_ua_0), 1)

  assert_equal(tf.shape(scale_r3_eps_2_ua_1).numpy(), [1, 4, 1])
  assert_equal(_get_num_unique_elements(scale_r3_eps_2_ua_1), 2)

  assert_equal(tf.shape(scale_r3_eps_2_ua_2).numpy(), [1, 1, 8])
  assert_equal(_get_num_unique_elements(scale_r3_eps_2_ua_2), 4)

  # Test get_scale function when elements_per_scale and scale_axis have scalar
  # values and the input x and q tensors have rank 4
  x_r4 = tf.random.uniform([2, 4, 8, 16])
  q_r4 = tf.random.uniform([2, 4, 8, 16])
  scale_r4_eps_none_ua_none = quantizers._get_least_squares_scale(
      "auto", x_r4, q_r4, elements_per_scale=None, scale_axis=None)
  scale_r4_eps_2_ua_0 = quantizers._get_least_squares_scale(
      "auto", x_r4, q_r4, elements_per_scale=2, scale_axis=0)
  scale_r4_eps_2_ua_1 = quantizers._get_least_squares_scale(
      "auto", x_r4, q_r4, elements_per_scale=2, scale_axis=1)
  scale_r4_eps_2_ua_2 = quantizers._get_least_squares_scale(
      "auto", x_r4, q_r4, elements_per_scale=2, scale_axis=2)
  scale_r4_eps_2_ua_3 = quantizers._get_least_squares_scale(
      "auto", x_r4, q_r4, elements_per_scale=2, scale_axis=3)

  assert_equal(tf.shape(scale_r4_eps_none_ua_none).numpy(), [1, 1, 1, 16])
  assert_equal(_get_num_unique_elements(scale_r4_eps_none_ua_none), 16)

  assert_equal(tf.shape(scale_r4_eps_2_ua_0).numpy(), [2, 1, 1, 1])
  assert_equal(_get_num_unique_elements(scale_r4_eps_2_ua_0), 1)

  assert_equal(tf.shape(scale_r4_eps_2_ua_1).numpy(), [1, 4, 1, 1])
  assert_equal(_get_num_unique_elements(scale_r4_eps_2_ua_1), 2)

  assert_equal(tf.shape(scale_r4_eps_2_ua_2).numpy(), [1, 1, 8, 1])
  assert_equal(_get_num_unique_elements(scale_r4_eps_2_ua_2), 4)

  assert_equal(tf.shape(scale_r4_eps_2_ua_3).numpy(), [1, 1, 1, 16])
  assert_equal(_get_num_unique_elements(scale_r4_eps_2_ua_3), 8)


def test_GetScale_ElementsPerScale_List_ScaleAxis_EPS():
  # Test get_scale function when elements_per_scale and scale_axis are lists of
  # rank 1 and the input x and q tensors have rank 3
  x_r3 = tf.random.uniform([2, 4, 8])
  q_r3 = tf.random.uniform([2, 4, 8])

  scale_r3_eps_none_ua_0 = quantizers._get_least_squares_scale(
      "auto", x_r3, q_r3, elements_per_scale=None, scale_axis=[0])
  scale_r3_eps_2_ua_0 = quantizers._get_least_squares_scale(
      "auto", x_r3, q_r3, elements_per_scale=[2], scale_axis=[0])
  scale_r3_eps_2_ua_1 = quantizers._get_least_squares_scale(
      "auto", x_r3, q_r3, elements_per_scale=[2], scale_axis=[1])
  scale_r3_eps_2_ua_2 = quantizers._get_least_squares_scale(
      "auto", x_r3, q_r3, elements_per_scale=[2], scale_axis=[2])

  assert_equal(tf.shape(scale_r3_eps_none_ua_0).numpy(), [2, 1, 1])
  assert_equal(_get_num_unique_elements(scale_r3_eps_none_ua_0), 2)

  assert_equal(tf.shape(scale_r3_eps_2_ua_0).numpy(), [2, 1, 1])
  assert_equal(_get_num_unique_elements(scale_r3_eps_2_ua_0), 1)

  assert_equal(tf.shape(scale_r3_eps_2_ua_1).numpy(), [1, 4, 1])
  assert_equal(_get_num_unique_elements(scale_r3_eps_2_ua_1), 2)

  assert_equal(tf.shape(scale_r3_eps_2_ua_2).numpy(), [1, 1, 8])
  assert_equal(_get_num_unique_elements(scale_r3_eps_2_ua_2), 4)

  # Test get_scale function when elements_per_scale and scale_axis are lists of
  # rank 2 and the input x and q tensors have rank 3
  x_r3 = tf.random.uniform([2, 4, 8])
  q_r3 = tf.random.uniform([2, 4, 8])

  scale_r3_eps_none_ua_01 = quantizers._get_least_squares_scale(
      "auto", x_r3, q_r3, elements_per_scale=None, scale_axis=[0, 1])
  scale_r3_eps_22_ua_01 = quantizers._get_least_squares_scale(
      "auto", x_r3, q_r3, elements_per_scale=[2, 2], scale_axis=[0, 1])
  scale_r3_eps_11_ua_12 = quantizers._get_least_squares_scale(
      "auto", x_r3, q_r3, elements_per_scale=[2, 2], scale_axis=[1, 2])
  scale_r3_eps_11_ua_02 = quantizers._get_least_squares_scale(
      "auto", x_r3, q_r3, elements_per_scale=[1, 1], scale_axis=[0, 2])

  assert_equal(tf.shape(scale_r3_eps_none_ua_01).numpy(), [2, 4, 1])
  assert_equal(_get_num_unique_elements(scale_r3_eps_none_ua_01), 8)

  assert_equal(tf.shape(scale_r3_eps_22_ua_01).numpy(), [2, 4, 1])
  assert_equal(_get_num_unique_elements(scale_r3_eps_22_ua_01), 2)

  assert_equal(tf.shape(scale_r3_eps_11_ua_12).numpy(), [1, 4, 8])
  assert_equal(_get_num_unique_elements(scale_r3_eps_11_ua_12), 8)

  assert_equal(tf.shape(scale_r3_eps_11_ua_02).numpy(), [2, 1, 8])
  assert_equal(_get_num_unique_elements(scale_r3_eps_11_ua_02), 16)

  # Test get_scale function when elements_per_scale and scale_axis are lists of
  # rank 3 and the input x and q tensors have rank 4
  x_r4 = tf.random.uniform([2, 4, 8, 16])
  q_r4 = tf.random.uniform([2, 4, 8, 16])

  scale_r4_eps_none_ua_012 = quantizers._get_least_squares_scale(
      "auto", x_r4, q_r4, elements_per_scale=None, scale_axis=[0, 1, 2])
  scale_r4_eps_221_ua_012 = quantizers._get_least_squares_scale(
      "auto", x_r4, q_r4, elements_per_scale=[2, 2, 1], scale_axis=[0, 1, 2])
  scale_r4_eps_221_ua_123 = quantizers._get_least_squares_scale(
      "auto", x_r4, q_r4, elements_per_scale=[2, 2, 1], scale_axis=[1, 2, 3])
  scale_r4_eps_221_ua_013 = quantizers._get_least_squares_scale(
      "auto", x_r4, q_r4, elements_per_scale=[2, 2, 1], scale_axis=[0, 1, 3])

  assert_equal(tf.shape(scale_r4_eps_none_ua_012).numpy(), [2, 4, 8, 1])
  assert_equal(_get_num_unique_elements(scale_r4_eps_none_ua_012), 64)

  assert_equal(tf.shape(scale_r4_eps_221_ua_012).numpy(), [2, 4, 8, 1])
  assert_equal(_get_num_unique_elements(scale_r4_eps_221_ua_012), 16)

  assert_equal(tf.shape(scale_r4_eps_221_ua_123).numpy(), [1, 4, 8, 16])
  assert_equal(_get_num_unique_elements(scale_r4_eps_221_ua_123), 128)

  assert_equal(tf.shape(scale_r4_eps_221_ua_013).numpy(), [2, 4, 1, 16])
  assert_equal(_get_num_unique_elements(scale_r4_eps_221_ua_013), 32)


def test_GetScale_MinPO2Exponent_MaxPO2Exponent():
  """Verify get_scale function with min and max po2_exponent clipping."""

  def _get_min_max_po2_exponent(x):
    """Get min and max po2 exponent of x."""
    po2_x = K.log(x)/np.log(2.0)
    return (tf.math.reduce_min(po2_x).numpy(),
            tf.math.reduce_max(po2_x).numpy())

  # generate small decimal numbers to verify that po2 clipping works properly
  x = 2**tf.random.uniform(shape=[2, 4, 8], minval=-50, maxval=0)
  q = 2**tf.random.uniform(shape=[2, 4, 8], minval=-50, maxval=0)

  # set various min and max po2 exponents for the scale
  scale_min_neg3_max_1 = quantizers._get_least_squares_scale(
      "auto_po2", x, q, elements_per_scale=4, scale_axis=2, min_po2_exponent=-3,
      max_po2_exponent=1)

  scale_min_neg8_max_0 = quantizers._get_least_squares_scale(
      "auto_po2", x, q, elements_per_scale=4, scale_axis=2, min_po2_exponent=-8,
      max_po2_exponent=0)

  scale_min_neg10_max_1 = quantizers._get_least_squares_scale(
      "auto_po2", x, q, elements_per_scale=4, scale_axis=2,
      min_po2_exponent=-10, max_po2_exponent=1)

  # verify that the output scales have the correct min and max ranges
  assert_equal(tf.shape(scale_min_neg3_max_1).numpy(), [1, 1, 8])
  min_po2_exp, max_po2_exp = _get_min_max_po2_exponent(scale_min_neg3_max_1)
  assert min_po2_exp >= -3
  assert max_po2_exp <= 1

  assert_equal(tf.shape(scale_min_neg8_max_0).numpy(), [1, 1, 8])
  min_po2_exp, max_po2_exp = _get_min_max_po2_exponent(scale_min_neg8_max_0)
  assert min_po2_exp >= -8
  assert max_po2_exp <= 0

  assert_equal(tf.shape(scale_min_neg10_max_1).numpy(), [1, 1, 8])
  min_po2_exp, max_po2_exp = _get_min_max_po2_exponent(scale_min_neg10_max_1)
  assert min_po2_exp >= -10
  assert max_po2_exp <= 1


def test_GetUnrolledShape_GetRolledBackShape():
  x_r4 = [4, 4, 8, 16]

  # Scalar unroll_factor and unroll_axis - Test _get_unrolled_shape
  unrolled_x_r4_uf_2_ua_0 = quantizers._get_unrolled_shape(
      x_r4, unroll_factor=2, unroll_axis=0)
  unrolled_x_r4_uf_2_ua_1 = quantizers._get_unrolled_shape(
      x_r4, unroll_factor=2, unroll_axis=1)
  unrolled_x_r4_uf_2_ua_2 = quantizers._get_unrolled_shape(
      x_r4, unroll_factor=2, unroll_axis=2)
  unrolled_x_r4_uf_2_ua_3 = quantizers._get_unrolled_shape(
      x_r4, unroll_factor=2, unroll_axis=3)

  assert_equal(unrolled_x_r4_uf_2_ua_0, ([2, 2, 4, 8, 16], 0))
  assert_equal(unrolled_x_r4_uf_2_ua_1, ([4, 2, 2, 8, 16], 1))
  assert_equal(unrolled_x_r4_uf_2_ua_2, ([4, 4, 4, 2, 16], 2))
  assert_equal(unrolled_x_r4_uf_2_ua_3, ([4, 4, 8, 8, 2], 3))

  # Scalar unroll_factor and unroll_axis - Test _get_rolled_back_shape
  rolled_back_x_r4_uf_2_ua_0 = quantizers._get_rolled_back_shape(
      unrolled_x_r4_uf_2_ua_0[0], roll_axis=unrolled_x_r4_uf_2_ua_0[1])
  rolled_back_x_r4_uf_2_ua_1 = quantizers._get_rolled_back_shape(
      unrolled_x_r4_uf_2_ua_1[0], roll_axis=unrolled_x_r4_uf_2_ua_1[1])
  rolled_back_x_r4_uf_2_ua_2 = quantizers._get_rolled_back_shape(
      unrolled_x_r4_uf_2_ua_2[0], roll_axis=unrolled_x_r4_uf_2_ua_2[1])
  rolled_back_x_r4_uf_2_ua_3 = quantizers._get_rolled_back_shape(
      unrolled_x_r4_uf_2_ua_3[0], roll_axis=unrolled_x_r4_uf_2_ua_3[1])

  assert_equal(x_r4, rolled_back_x_r4_uf_2_ua_0)
  assert_equal(x_r4, rolled_back_x_r4_uf_2_ua_1)
  assert_equal(x_r4, rolled_back_x_r4_uf_2_ua_2)
  assert_equal(x_r4, rolled_back_x_r4_uf_2_ua_3)

  # List[2] unroll_factor and unroll_axis - Test _get_unrolled_shape
  unrolled_x_r4_uf_24_ua_01 = quantizers._get_unrolled_shape(
      x_r4, unroll_factor=[2, 4], unroll_axis=[0, 1])
  unrolled_x_r4_uf_24_ua_12 = quantizers._get_unrolled_shape(
      x_r4, unroll_factor=[2, 4], unroll_axis=[1, 2])
  unrolled_x_r4_uf_24_ua_13 = quantizers._get_unrolled_shape(
      x_r4, unroll_factor=[2, 4], unroll_axis=[1, 3])
  unrolled_x_r4_uf_24_ua_34 = quantizers._get_unrolled_shape(
      x_r4, unroll_factor=[2, 4], unroll_axis=[2, 3])

  assert_equal(unrolled_x_r4_uf_24_ua_01, ([2, 2, 1, 4, 8, 16], [0, 2]))
  assert_equal(unrolled_x_r4_uf_24_ua_12, ([4, 2, 2, 2, 4, 16], [1, 3]))
  assert_equal(unrolled_x_r4_uf_24_ua_13, ([4, 2, 2, 8, 4, 4], [1, 4]))
  assert_equal(unrolled_x_r4_uf_24_ua_34, ([4, 4, 4, 2, 4, 4], [2, 4]))

  # List[2] unroll_factor and unroll_axis - Test _get_rolled_back_shape
  rolled_back_x_r4_uf_24_ua_01 = quantizers._get_rolled_back_shape(
      unrolled_x_r4_uf_24_ua_01[0], roll_axis=unrolled_x_r4_uf_24_ua_01[1])
  rolled_back_x_r4_uf_24_ua_12 = quantizers._get_rolled_back_shape(
      unrolled_x_r4_uf_24_ua_12[0], roll_axis=unrolled_x_r4_uf_24_ua_12[1])
  rolled_back_x_r4_uf_24_ua_13 = quantizers._get_rolled_back_shape(
      unrolled_x_r4_uf_24_ua_13[0], roll_axis=unrolled_x_r4_uf_24_ua_13[1])
  rolled_back_x_r4_uf_24_ua_34 = quantizers._get_rolled_back_shape(
      unrolled_x_r4_uf_24_ua_34[0], roll_axis=unrolled_x_r4_uf_24_ua_34[1])

  assert_equal(x_r4, rolled_back_x_r4_uf_24_ua_01)
  assert_equal(x_r4, rolled_back_x_r4_uf_24_ua_12)
  assert_equal(x_r4, rolled_back_x_r4_uf_24_ua_13)
  assert_equal(x_r4, rolled_back_x_r4_uf_24_ua_34)

  # List[3] unroll_factor and unroll_axis - Test _get_unrolled_shape
  unrolled_x_r4_uf_242_ua_012 = quantizers._get_unrolled_shape(
      x_r4, unroll_factor=[2, 4, 2], unroll_axis=[0, 1, 2])
  unrolled_x_r4_uf_242_ua_023 = quantizers._get_unrolled_shape(
      x_r4, unroll_factor=[2, 4, 2], unroll_axis=[0, 2, 3])

  assert_equal(unrolled_x_r4_uf_242_ua_012, ([2, 2, 1, 4, 4, 2, 16], [0, 2, 4]))
  assert_equal(unrolled_x_r4_uf_242_ua_023, ([2, 2, 4, 2, 4, 8, 2], [0, 3, 5]))

  # List[3] unroll_factor and unroll_axis - Test _get_rolled_back_shape
  rolled_back_x_r4_uf_242_ua_012 = quantizers._get_rolled_back_shape(
      unrolled_x_r4_uf_242_ua_012[0],
      roll_axis=unrolled_x_r4_uf_242_ua_012[1])
  rolled_back_x_r4_uf_242_ua_023 = quantizers._get_rolled_back_shape(
      unrolled_x_r4_uf_242_ua_023[0],
      roll_axis=unrolled_x_r4_uf_242_ua_023[1])

  assert_equal(x_r4, rolled_back_x_r4_uf_242_ua_012)
  assert_equal(x_r4, rolled_back_x_r4_uf_242_ua_023)

if __name__ == "__main__":
  pytest.main([__file__])
