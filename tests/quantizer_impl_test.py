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
  scale_r1_pcs_true = quantizers._get_scale(
      "auto", x_r1, q_r1, scale_axis=None, per_channel_scale=True)
  scale_r1_pcs_false = quantizers._get_scale(
      "auto", x_r1, q_r1, scale_axis=None, per_channel_scale=False)
  assert_equal(tf.shape(scale_r1_pcs_true).numpy(), [4])
  assert_equal(tf.shape(scale_r1_pcs_false).numpy(), [1])

  # Rank2 tensors
  x_r2 = tf.ones([2, 4])
  q_r2 = tf.ones([2, 4])
  scale_r2_pcs_true = quantizers._get_scale(
      "auto", x_r2, q_r2, scale_axis=None, per_channel_scale=True)
  scale_r2_pcs_false = quantizers._get_scale(
      "auto", x_r2, q_r2, scale_axis=None, per_channel_scale=False)
  assert_equal(tf.shape(scale_r2_pcs_true).numpy(), [1, 4])
  assert_equal(tf.shape(scale_r2_pcs_false).numpy(), [1, 1])

  # Rank3 tensors
  x_r3 = tf.ones([3, 3, 4])
  q_r3 = tf.ones([3, 3, 4])
  scale_r3_pcs_true = quantizers._get_scale(
      "auto", x_r3, q_r3, scale_axis=None, per_channel_scale=True)
  scale_r3_pcs_false = quantizers._get_scale(
      "auto", x_r3, q_r3, scale_axis=None, per_channel_scale=False)
  assert_equal(tf.shape(scale_r3_pcs_true).numpy(), [1, 1, 4])
  assert_equal(tf.shape(scale_r3_pcs_false).numpy(), [1, 1, 1])

  # Rank4 tensors
  x_r4 = tf.ones([1, 1, 3, 4])
  q_r4 = tf.ones([1, 1, 3, 4])
  scale_r4_pcs_true = quantizers._get_scale(
      "auto", x_r4, q_r4, scale_axis=None, per_channel_scale=True)
  scale_r4_pcs_false = quantizers._get_scale(
      "auto", x_r4, q_r4, scale_axis=None, per_channel_scale=False)
  assert_equal(tf.shape(scale_r4_pcs_true).numpy(), [1, 1, 1, 4])
  assert_equal(tf.shape(scale_r4_pcs_false).numpy(), [1, 1, 1, 1])

if __name__ == "__main__":
  pytest.main([__file__])
