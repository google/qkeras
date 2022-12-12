# Copyright 2021 Google LLC
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
"""Tests for quantizers."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_equal
import tensorflow as tf

from tensorflow.keras.layers import *
from qkeras.experimental.quantizers.quantizers_po2 import BaseQuantizerPO2
from qkeras.experimental.quantizers.quantizers_po2 import quantized_bits_learnable_po2
from qkeras.experimental.quantizers.quantizers_po2 import quantized_bits_msqe_po2


#################################
# Test cases for BaseQuantizerPO2
#################################

# Test BaseQuantizerPO2 build
@pytest.mark.parametrize(
    # input attributes
    'bits,'
    'keep_negative,'
    'scale_axis,'
    'per_channel_scale,'
    'init_scale,'
    'use_second_moments_msqe_opt,'
    'second_moments_ema_decay,'
    'use_sqrt_of_msqe_weight,'
    'use_outlier_mask_msqe_weight,'
    'use_stable_scale_exponent,'
    'stable_scale_ema_decay,'
    'is_gradient_based,'
    'inputs_shape,'
    # expected values
    'expected_qn,'
    'expected_qp,'
    'expected_scaled_axes,'
    'expected_reduce_axes,'
    'expected_msqe_weight_shape,'
    'expected_scale_exponent_shape,'
    'expected_scale_axis', [
        (
            3,             # bits
            True,          # keep_negative
            3,             # scale_axis
            True,          # per_channel_scale
            2.5,           # init_scale
            True,          # use_second_moments_msqe_opt
            0.1,           # second_moments_ema_decay
            True,          # use_sqrt_of_msqe_weight
            True,          # use_outlier_mask_msqe_weight
            True,          # use_stable_scale_exponent
            0.1,           # stable_scale_ema_decay
            True,          # is_gradient_based
            [2, 3, 4, 5],  # inputs_shape
            3,             # expected_qn
            3,             # expected_qp
            [0, 1, 2],     # expected_scaled_axes
            [0, 1, 2],     # expected_reduce_axes
            [2, 3, 4, 5],  # expected_msqe_weight_shape
            [1, 1, 1, 5],  # expected_scale_exponent_shape
            3,             # expected_scale_axis
        ),
        (
            3,             # bits
            True,          # keep_negative
            2,             # scale_axis
            True,          # per_channel_scale
            2.5,           # init_scale
            True,          # use_second_moments_msqe_opt
            0.1,           # second_moments_ema_decay
            True,          # use_sqrt_of_msqe_weight
            True,          # use_outlier_mask_msqe_weight
            True,          # use_stable_scale_exponent
            0.1,           # stable_scale_ema_decay
            True,          # is_gradient_based
            [2, 3, 4, 5],  # inputs_shape
            3,             # expected_qn
            3,             # expected_qp
            [0, 1],        # expected_scaled_axes
            [0, 1, 3],     # expected_reduce_axes
            [2, 3, 4, 5],  # expected_msqe_weight_shape
            [1, 1, 4, 1],  # expected_scale_exponent_shape
            2,             # expected_scale_axis
        ),
        (
            3,             # bits
            True,          # keep_negative
            None,          # scale_axis
            True,          # per_channel_scale
            2.5,           # init_scale
            True,          # use_second_moments_msqe_opt
            0.1,           # second_moments_ema_decay
            True,          # use_sqrt_of_msqe_weight
            True,          # use_outlier_mask_msqe_weight
            True,          # use_stable_scale_exponent
            0.1,           # stable_scale_ema_decay
            True,          # is_gradient_based
            [None, 3, 4, 5],  # inputs_shape
            3,             # expected_qn
            3,             # expected_qp
            [0, 1, 2],     # expected_scaled_axes
            [0, 1, 2],     # expected_reduce_axes
            [1, 3, 4, 5],  # expected_msqe_weight_shape
            [1, 1, 1, 5],  # expected_scale_exponent_shape
            3,             # expected_scale_axis
        ),
        (
            3,             # bits
            False,         # keep_negative
            3,             # scale_axis
            False,         # per_channel_scale
            2.5,           # init_scale
            False,         # use_second_moments_msqe_opt
            0.1,           # second_moments_ema_decay
            False,         # use_sqrt_of_msqe_weight
            False,         # use_outlier_mask_msqe_weight
            False,         # use_stable_scale_exponent
            0.1,           # stable_scale_ema_decay
            False,         # is_gradient_based
            [2, 3, 4, 5],  # inputs_shape
            0,             # expected_qn
            7,             # expected_qp
            [0, 1, 2, 3],  # expected_scaled_axes
            [0, 1, 2, 3],  # expected_reduce_axes
            None,          # expected_msqe_weight_shape
            [1, 1, 1, 1],  # expected_scale_exponent_shape
            3,             # expected_scale_axis
        ),
    ])
def test_BaseQuantizerPO2_build(
    # input attributes
    bits,
    keep_negative,
    scale_axis,
    per_channel_scale,
    init_scale,
    use_second_moments_msqe_opt,
    second_moments_ema_decay,
    use_sqrt_of_msqe_weight,
    use_outlier_mask_msqe_weight,
    use_stable_scale_exponent,
    stable_scale_ema_decay,
    is_gradient_based,
    inputs_shape,
    # expected values
    expected_qn,
    expected_qp,
    expected_scaled_axes,
    expected_reduce_axes,
    expected_msqe_weight_shape,
    expected_scale_exponent_shape,
    expected_scale_axis,
):

  q = BaseQuantizerPO2(
      bits=bits,
      init_scale=init_scale,
      keep_negative=keep_negative,
      per_channel_scale=per_channel_scale,
      scale_axis=scale_axis,
      use_second_moments_msqe_opt=use_second_moments_msqe_opt,
      second_moments_ema_decay=second_moments_ema_decay,
      use_sqrt_of_msqe_weight=use_sqrt_of_msqe_weight,
      use_outlier_mask_msqe_weight=use_outlier_mask_msqe_weight,
      use_stable_scale_exponent=use_stable_scale_exponent,
      stable_scale_ema_decay=stable_scale_ema_decay,
      is_gradient_based=is_gradient_based,
  )
  q.build(inputs_shape)

  # Test attributes assignment
  assert_equal(q.bits, bits)
  assert_equal(q.init_scale, init_scale)
  assert_equal(q.keep_negative, keep_negative)
  assert_equal(q.per_channel_scale, per_channel_scale)
  if scale_axis is not None:
    assert_equal(q.scale_axis, scale_axis)
  assert_equal(q.use_second_moments_msqe_opt,
               use_second_moments_msqe_opt)
  assert_equal(q.second_moments_ema_decay, second_moments_ema_decay)
  assert_equal(q.use_sqrt_of_msqe_weight, use_sqrt_of_msqe_weight)
  assert_equal(q.use_outlier_mask_msqe_weight, use_outlier_mask_msqe_weight)
  assert_equal(q.use_stable_scale_exponent, use_stable_scale_exponent)
  assert_equal(q.stable_scale_ema_decay, stable_scale_ema_decay)
  assert_equal(q.is_gradient_based, is_gradient_based)

  # Test build
  assert_equal(q.scale_exponent.shape, expected_scale_exponent_shape)
  assert_equal(q.scale_exponent.trainable, is_gradient_based)
  assert_equal(q.scale.shape, expected_scale_exponent_shape)
  assert_equal(q.qn, expected_qn)
  assert_equal(q.qp, expected_qp)
  assert_equal(q.scaled_axes, expected_scaled_axes)
  assert_equal(q.reduce_axes, expected_reduce_axes)
  assert_allclose(q.init_scale, np.power(2.0, q.scale_exponent))
  assert_equal(q.scale_axis, expected_scale_axis)

  if q.use_second_moments_msqe_opt:
    assert_equal(q.msqe_weight.shape, expected_msqe_weight_shape)
  else:
    assert_equal(q.msqe_weight, None)

  if q.use_stable_scale_exponent:
    assert_equal(q.stable_scale_exponent.shape, expected_scale_exponent_shape)
    assert_equal(q.switch_to_stable_scale.numpy(), False)
    assert_equal(q.should_update_stable_scale_exponent.numpy(), False)
  else:
    assert_equal(q.stable_scale_exponent, None)
    assert_equal(q.switch_to_stable_scale, None)
    assert_equal(q.should_update_stable_scale_exponent, None)

  assert_equal(q.is_initialized.numpy(), False)


# Test BaseQuantizerPO2 for the power-of-2 scale constraint
@pytest.mark.parametrize('inputs_shape, init_scale', [
    ([2,3,4,5], 0.123),
    ([2,3,4,5], 0.1),
    ([2,3,4,5], 0.011),
    ([2,3,4,5], 0.0123),
    ([2,3,4,5], 1.0123),
])
def test_BaseQuantizerPO2_PO2Constraint(inputs_shape, init_scale):

  q = BaseQuantizerPO2(per_channel_scale=False, init_scale=init_scale)
  q.build(inputs_shape)

  # Test initializing the scale exponent
  scale_init = np.power(2.0, q.scale_exponent)
  init_scale_exp = np.log2(init_scale)

  assert_almost_equal(scale_init, init_scale)
  assert_array_equal(
      q._get_po2_scale_exponent(scale_init), np.round(init_scale_exp))
  assert_array_equal(
      q._get_po2_scale(scale_init), np.power(2.0, np.round(init_scale_exp)))

  # Test the optimizers
  inputs = np.random.normal(size=inputs_shape)

  scale_least = q._least_squares_msqe_scale(
      inputs, scale_init, should_return_msqe=False)
  assert_array_equal(np.log2(scale_least), np.round(np.log2(scale_least)))

  scale_fine = q._line_search_msqe_scale(
      inputs, scale_init, should_return_msqe=False)
  assert_array_equal(np.log2(scale_fine), np.round(np.log2(scale_fine)))

  scale_opt, _ = q._optimize_msqe_scale(
      inputs,
      scale_init,
      num_lls_iters=3,
      line_search_range=6)
  assert_array_equal(np.log2(scale_opt), np.round(np.log2(scale_opt)))


# Test BaseQuantizerPO2 the optimizers
@pytest.mark.parametrize('bits, init_scale, keep_negative, inputs, scale_least,'
                         'scale_fine, scale_opt', [
    (4, 1.0, True,
     np.array([-0.17, 2.58, -8.75, -3.56, 1.56,-0.15, 2.15, -0.66, 0.49,]),
     1.0, 2.0, 2.0),
])
def test_BaseQuantizerPO2_scale_optimization(
    bits,
    init_scale,
    keep_negative,
    inputs,
    scale_least,
    scale_fine,
    scale_opt,
):

  q = BaseQuantizerPO2(
      bits=bits,
      keep_negative=keep_negative,
      init_scale=init_scale,
  )
  q.build(inputs.shape)

  scale_init = np.power(2.0, q.scale_exponent)
  scale_init_exp_po2 = np.round(np.log2(scale_init))

  scale_least_est = q._least_squares_msqe_scale(
      inputs, scale_init, should_return_msqe=False)

  scale_fine_est = q._line_search_msqe_scale(
      inputs, scale_least_est, should_return_msqe=False)

  scale_opt_est, _ = q._optimize_msqe_scale(
      inputs,
      scale_init,
      q.reduce_axes,
      q.msqe_weight,
      num_lls_iters=3,
      line_search_range=6)

  assert_almost_equal(scale_least_est, scale_least)
  assert_almost_equal(scale_fine_est, scale_fine)
  assert_almost_equal(scale_opt_est, scale_opt)


# Test BaseQuantizerPO2 '_get_clipped_inputs_mask'
@pytest.mark.parametrize('bits, init_scale, keep_negative, qn, qp, inputs,'
                         'round_mask', [
    (2, 1.0, True, 1, 1,
     np.array([0.5, 0.0, 1.0, 1.5, 1.6, 2.0, -0.5, -1.0, -1.5, -1.6, -2.0,]),
     np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0,  1.0,  1.0,  1.0,  0.0,  0.0,])),
])
def test_BaseQuantizerPO2_clip_mask(
    bits,
    init_scale,
    keep_negative,
    qn,
    qp,
    inputs,
    round_mask,
):

  q = BaseQuantizerPO2(
      bits=bits,
      keep_negative=keep_negative,
      init_scale=init_scale,
  )
  q.build(inputs.shape)

  assert_equal(q.qn, qn)
  assert_equal(q.qp, qp)

  scale_init = np.power(2.0, q.scale_exponent)
  round_mask_est = q._get_clipped_inputs_mask(inputs, scale_init)
  assert_array_equal(round_mask_est, round_mask)


# Test BaseQuantizerPO2 MSQE calculation
@pytest.mark.parametrize(
    'per_channel_scale, keep_negative, scale, inputs, msqe_weight, msqe', [
        (
            False,
            True,
            np.array([1.0]),
            np.array([1.2, 0.9, 1.0, 1.6, 0.3, 1.0]),
            None,
            np.array([0.299999999]),
        ),
        (
            False,
            True,
            np.array([1.0]),
            np.array([1.2, 0.9, 1.0, 1.6, 0.3, 1.0]),
            np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6]),
            np.array([0.4149999999]),
        ),
        (
            True,
            True,
            np.array([1.0, 2.0, 1.0, 4.0, 0.5, 0.25]),
            np.array([1.2, 0.9, 3.1, 1.6, 0.3, 1.0]),
            None,
            np.array([0.04, 0.81, 0.01, 2.56, 0.04, 0.0]),
        ),
    ])
def test_BaseQuantizerPO2_msqe_calculation(
    per_channel_scale,
    keep_negative,
    scale,
    inputs,
    msqe_weight,
    msqe,
):

  q = BaseQuantizerPO2(
      bits=4,
      per_channel_scale=per_channel_scale,
      keep_negative=keep_negative,
  )
  q.build(inputs.shape)

  calculated_msqe = q._calculate_msqe_inputs(
      inputs=inputs, scale=scale, msqe_weight=msqe_weight)
  assert_allclose(calculated_msqe, msqe)


# Test BaseQuantizerPO2 _get_stable_scale
@pytest.mark.parametrize(
    'is_initialized,'
    'should_update_stable_scale_exponent_0,'
    'should_update_stable_scale_exponent_1,'
    # Note: when switch_to_stable_scale is true, the stable scale exponent is
    # NOT updated.
    'switch_to_stable_scale_0,'
    'switch_to_stable_scale_1,'
    # scale input
    'scale_0,'
    'scale_1,'
    # scale output (the stable scale exponent is set to 0.0 by default,
    # i.e., the stable scale is set 1.0 by default.)
    'expected_scale_0,'
    'expected_scale_1,'
    'expected_stable_scale_exponent,'
    , [
        (
            True, # is_initialized
            True, # should_update_stable_scale_exponent_0
            True, # should_update_stable_scale_exponent_1
            True, # switch_to_stable_scale_0
            True, # switch_to_stable_scale_1
            # scale inputs
            np.array([[1.0, 2.0, 3.0]], dtype=np.float32), # scale_0
            np.array([[1.0, 2.0, 3.0]], dtype=np.float32), # scale_1
            # expected scale outputs
            np.array([[1.0, 1.0, 1.0]], dtype=np.float32), # expected_scale_0
            np.array([[1.0, 1.0, 1.0]], dtype=np.float32), # expected_scale_1
            # expected stable scale exponent
            np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        ),
        (
            False, # is_initialized
            True, # should_update_stable_scale_exponent_0
            True, # should_update_stable_scale_exponent_1
            False, # switch_to_stable_scale_0
            True, # switch_to_stable_scale_1
            # scale inputs
            np.array([[1.0, 2.0, 3.0]], dtype=np.float32), # scale_0
            np.array([[1.0, 2.0, 3.0]], dtype=np.float32), # scale_1
            # expected scale outputs
            np.array([[1.0, 2.0, 3.0]], dtype=np.float32), # expected_scale_0
            np.array([[1.0, 2.0, 4.0]], dtype=np.float32), # expected_scale_1
            # expected stable scale exponent
            np.array([[0.0, 1.0, 2.0]], dtype=np.float32),
        ),
        (
            False, # is_initialized
            True, # should_update_stable_scale_exponent_0
            True, # should_update_stable_scale_exponent_1
            False, # switch_to_stable_scale_0
            False, # switch_to_stable_scale_1
            # scale inputs
            np.array([[1.0, 2.0, 3.0]], dtype=np.float32), # scale_0
            np.array([[1.0, 1.0, 1.0]], dtype=np.float32), # scale_1
            # expected scale outputs
            np.array([[1.0, 2.0, 3.0]], dtype=np.float32), # expected_scale_0
            np.array([[1.0, 1.0, 1.0]], dtype=np.float32), # expected_scale_1
            # expected stable scale exponent
            np.array([[0.0, 0.99, 1.98]], dtype=np.float32),
        ),
    ])
def test_BaseQuantizerPO2_get_stable_scale(
    is_initialized,
    should_update_stable_scale_exponent_0,
    should_update_stable_scale_exponent_1,
    switch_to_stable_scale_0,
    switch_to_stable_scale_1,
    scale_0,
    scale_1,
    expected_scale_0,
    expected_scale_1,
    expected_stable_scale_exponent,
):

  q = BaseQuantizerPO2(
      bits=4,
      per_channel_scale=True,
      keep_negative=True,
      use_stable_scale_exponent=True,
      stable_scale_ema_decay=0.99,
  )
  q.build([2,3])

  q.is_initialized.assign(is_initialized)

  # first pass
  q.should_update_stable_scale_exponent.assign(
      should_update_stable_scale_exponent_0)
  q.switch_to_stable_scale.assign(switch_to_stable_scale_0)
  assert_allclose(q._get_stable_scale(scale_0), expected_scale_0)

  q.is_initialized.assign(True)

  # second pass
  q.should_update_stable_scale_exponent.assign(
      should_update_stable_scale_exponent_1)
  q.switch_to_stable_scale.assign(switch_to_stable_scale_1)
  assert_allclose(q._get_stable_scale(scale_1), expected_scale_1)

  assert_allclose(q.stable_scale_exponent, expected_stable_scale_exponent)


#############################################
# Test cases for quantized_bits_learnable_po2
#############################################

# Test quantized_bits_learnable_po2 build
@pytest.mark.parametrize(
    'bits, init_scale, keep_negative, per_channel_scale, scale_axis,'
    'use_second_moments_msqe_opt, second_moments_ema_decay,'
    'use_stable_scale_exponent, stable_scale_ema_decay,'
    'min_init_scale, use_po2_scale_ceil, use_po2_scale_msqe_round,'
    'use_outlier_mask_msqe_weight, use_sqrt_of_msqe_weight,'
    'inputs_shape,', [
        (
            3,  # bits
            2.5,  # init_scale
            True,  # keep_negative
            True,  # per_channel_scale
            3,  # scale_axis
            True,  # use_second_moments_msqe_opt
            0.1,  # second_moments_ema_decay
            True,  # use_stable_scale_exponent
            0.1,  # stable_scale_ema_decay
            0.2,  # min_init_scale
            True,  # use_po2_scale_ceil
            True,  # use_po2_scale_msqe_round
            True,  # use_outlier_mask_msqe_weight
            True,  # use_sqrt_of_msqe_weight
            [2, 3, 4, 5],  # inputs_shape
        ),
        (
            1,  # bits
            None,  # init_scale
            False,  # keep_negative
            False,  # per_channel_scale
            2,  # scale_axis
            False,  # use_second_moments_msqe_opt
            0.1,  # second_moments_ema_decay
            False,  # use_stable_scale_exponent
            0.1,  # stable_scale_ema_decay
            0.2,  # min_init_scale
            False,  # use_po2_scale_ceil
            False,  # use_po2_scale_msqe_round
            False,  # use_outlier_mask_msqe_weight
            False,  # use_sqrt_of_msqe_weight
            [2, 3, 4, 5],  # inputs_shape
        ),
    ])
def test_quantized_bits_learnable_po2_build(
    bits,
    init_scale,
    keep_negative,
    per_channel_scale,
    scale_axis,
    use_second_moments_msqe_opt,
    second_moments_ema_decay,
    use_stable_scale_exponent,
    stable_scale_ema_decay,
    min_init_scale,
    use_po2_scale_ceil,
    use_po2_scale_msqe_round,
    use_outlier_mask_msqe_weight,
    use_sqrt_of_msqe_weight,
    inputs_shape,
):

  q = quantized_bits_learnable_po2(
      bits=bits,
      init_scale=init_scale,
      keep_negative=keep_negative,
      per_channel_scale=per_channel_scale,
      scale_axis=scale_axis,
      use_second_moments_msqe_opt=use_second_moments_msqe_opt,
      second_moments_ema_decay=second_moments_ema_decay,
      use_stable_scale_exponent=use_stable_scale_exponent,
      stable_scale_ema_decay=stable_scale_ema_decay,
      min_init_scale=min_init_scale,
      use_po2_scale_ceil=use_po2_scale_ceil,
      use_po2_scale_msqe_round=use_po2_scale_msqe_round,
      use_outlier_mask_msqe_weight=use_outlier_mask_msqe_weight,
      use_sqrt_of_msqe_weight=use_sqrt_of_msqe_weight,
  )
  q.build(inputs_shape)

  # Assigned
  assert_equal(q.bits, bits)
  assert_equal(q.keep_negative, keep_negative)
  assert_equal(q.scale_axis, scale_axis)
  assert_equal(q.per_channel_scale, per_channel_scale)
  assert_equal(q.init_scale, init_scale)
  assert_equal(q.min_init_scale, min_init_scale)
  assert_equal(q.use_second_moments_msqe_opt,
               use_second_moments_msqe_opt)
  assert_equal(q.second_moments_ema_decay, second_moments_ema_decay)
  assert_equal(q.use_sqrt_of_msqe_weight, use_sqrt_of_msqe_weight)
  assert_equal(q.use_outlier_mask_msqe_weight, use_outlier_mask_msqe_weight)
  assert_equal(q.use_stable_scale_exponent, use_stable_scale_exponent)
  assert_equal(q.stable_scale_ema_decay, stable_scale_ema_decay)
  assert_equal(q.use_po2_scale_ceil, use_po2_scale_ceil)
  assert_equal(q.use_po2_scale_msqe_round, use_po2_scale_msqe_round)
  assert_equal(q.is_gradient_based, True)
  assert_equal(q.scale_exponent.trainable, True)

  # Built
  if q.use_po2_scale_msqe_round:
    assert_equal(q.switch_to_msqe_round.trainable, False)
  else:
    assert_equal(q.switch_to_msqe_round, None)


# Test quantized_bits_learnable_po2 _get_outlier_mask
@pytest.mark.parametrize('bits, scale_exponent, keep_negative, scale_axis, '
                         'per_channel_scale, inputs, outlier_mask', [
    (4, [[0.0]], True, None, False,
     np.array([[6.0, 5.0, 1.0],[8.0, 3.0, -8.0]]),
     np.array([[1.0, 1.0, 1.0],[0.0, 1.0,  0.0]])),
    (4, [[0.0, -1.0, -2.0]], True, 1, True,
     np.array([[6.0, 5.0, 1.0],[8.0, 3.0, 2.0]]),
     np.array([[1.0, 0.0, 1.0],[0.0, 1.0, 0.0]])),
])
def test_quantized_bits_learnable_po2_outlier_mask(
    bits,
    scale_exponent,
    keep_negative,
    scale_axis,
    per_channel_scale,
    inputs,
    outlier_mask,
):

  q = quantized_bits_learnable_po2(
      bits=bits,
      keep_negative=keep_negative,
      scale_axis=scale_axis,
      per_channel_scale=per_channel_scale)
  q.build(inputs.shape)
  q.scale_exponent.assign(scale_exponent)
  assert_array_equal(q._get_outlier_mask(inputs), outlier_mask)


# Test quantized_bits_learnable_po2 _get_msqe_weight
@pytest.mark.parametrize(
    'bits, scale_exponent, keep_negative, scale_axis,'
    'per_channel_scale, use_sqrt_of_msqe_weight,'
    'use_outlier_mask_msqe_weight,'
    'inputs, msqe_weight, msqe_weight_output',
    [
        (4, [[0.0]], True, None, False, False, False,
         np.array([[6.0, 5.0, 1.0], [8.0, 3.0, -8.0]]),
         np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]),
         np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])),
        (4, [[0.0]], True, None, False, True, False,
         np.array([[6.0, 5.0, 1.0], [8.0, 3.0, -8.0]]),
         np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]),
         np.sqrt(np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]))),
        (4, [[0.0]], True, None, False, True, True,
         np.array([[6.0, 5.0, 1.0], [8.0, 3.0, -8.0]]),
         np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]),
         np.sqrt(np.array([[2.0, 2.0, 2.0], [0.0, 2.0, 0.0]]))),
        (4, [[0.0, -1.0, -2.0]], True, 1, True, True, True,
         np.array([[6.0, 5.0, 1.0], [8.0, 3.0, -2.0]]),
         np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]),
         np.sqrt(np.array([[2.0, 0.0, 2.0], [0.0, 2.0, 0.0]]))),
    ])
def test_quantized_bits_learnable_po2_msqe_weight(
    bits,
    scale_exponent,
    keep_negative,
    scale_axis,
    per_channel_scale,
    use_sqrt_of_msqe_weight,
    use_outlier_mask_msqe_weight,
    inputs,
    msqe_weight,
    msqe_weight_output,
):

  q = quantized_bits_learnable_po2(
      bits=bits,
      keep_negative=keep_negative,
      scale_axis=scale_axis,
      per_channel_scale=per_channel_scale,
      use_sqrt_of_msqe_weight=use_sqrt_of_msqe_weight,
      use_outlier_mask_msqe_weight=use_outlier_mask_msqe_weight,
      use_second_moments_msqe_opt=True)
  q.build(inputs.shape)
  q.scale_exponent.assign(scale_exponent)
  if q.use_second_moments_msqe_opt:
    q.msqe_weight.assign(msqe_weight)
  assert_array_almost_equal(q._get_msqe_weight(inputs), msqe_weight_output)


# Test quantized_bits_learnable_po2 _get_init_scale_exponent
@pytest.mark.parametrize(
    'bits, keep_negative, scale_axis, per_channel_scale, min_init_scale,'
    'inputs,'
    'expected_init_scale_exponent', [
        (4, True, None, False, 0.01,
         np.array([[6.0, 5.0, 1.0], [8.0, 3.0, -8.0]],
                  dtype=np.float32), np.array([[1.1528215056499365]])),
        (4, True, None, False, 2.5,
         np.array([[6.0, 5.0, 1.0], [8.0, 3.0, -8.0]],
                  dtype=np.float32), np.array([[1.3219280948873624]])),
        # min_init_scale is greater than the initial scales
        (4, True, None, True, 2.5,
         np.array([[6.0, 5.0, 1.0], [8.0, 3.0, -8.0]], dtype=np.float32),
         np.array([[1.3219280948873624, 1.3219280948873624, 1.3219280948873624]
                  ])),
        (4, True, None, True, 0.0,
         np.array([[6.0, 5.0, 1.0], [8.0, 3.0, -8.0]], dtype=np.float32),
         np.array([[-1.22239242, -1.22239242,  0.94753258]])),
    ])
def test_quantized_bits_learnable_po2_get_init_scale_exponent(
    bits,
    keep_negative,
    scale_axis,
    per_channel_scale,
    min_init_scale,
    inputs,
    expected_init_scale_exponent
):

  q = quantized_bits_learnable_po2(
      bits=bits,
      keep_negative=keep_negative,
      scale_axis=scale_axis,
      per_channel_scale=per_channel_scale,
      min_init_scale=min_init_scale)
  q.build(inputs.shape)

  assert_allclose(
      q._get_init_scale_exponent(inputs), expected_init_scale_exponent)


# Test quantized_bits_learnable_po2 msqe_round
@pytest.mark.parametrize(
    'bits, scale_exponent, keep_negative, scale_axis,'
    'per_channel_scale, use_sqrt_of_msqe_weight,'
    'use_outlier_mask_msqe_weight,'
    'inputs, msqe_weight, msqe_round_output',
    [
        (4, np.log2([[1.5]]), True, None, False, False, False,
         np.array([[6.0, 5.0, 1.0], [8.0, 3.0, -8.0]]),
         np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]),
         np.array([[0.0]])),
        (4, np.log2([[1.5, 1.5, 1.5]]), True, 1, True, False, False,
         np.array([[6.0, 5.0, 1.0], [8.0, 3.0, -8.0]]),
         np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]),
         np.array([[1.0, 0.0, 1.0]])),
        (4, np.log2([[2.1, 0.51, 0.251]]), True, 1, True, False, False,
         np.array([[6.0, 5.0, 1.0], [8.0, 3.0, -8.0]]),
         np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]),
         np.array([[1., 0., -1.]])),
        (4, np.log2([[1.1, 0.9, 0.9]]), True, 1, True, True, False,
         np.array([[6.6, 5.0, 1.0], [8.5, 3.0, -8.0]]),
         np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]),
         np.array([[ 1., 0., 0.]])),
        (4, np.log2([[1.1, 0.9, 0.9]]), True, 1, True, True, True,
         np.array([[6.6, 5.0, 1.0], [8.5, 3.0, -8.0]]),
         np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]),
         np.array([[ 0., 0., 0.]])),
        (4, np.log2([[1.1, 0.9, 0.9]]), True, 1, True, True, False,
         np.array([[6.6, 5.0, 1.0], [8.5, 3.0, -8.0]]),
         np.array([[2.0, 2.0, 2.0], [0.0, 2.0, 0.0]]),
         np.array([[ 0., 0., 0.]])),
    ])
def test_quantized_bits_learnable_po2_msqe_round(
    bits,
    scale_exponent,
    keep_negative,
    scale_axis,
    per_channel_scale,
    use_sqrt_of_msqe_weight,
    use_outlier_mask_msqe_weight,
    inputs,
    msqe_weight,
    msqe_round_output,
):

  q = quantized_bits_learnable_po2(
      bits=bits,
      keep_negative=keep_negative,
      scale_axis=scale_axis,
      per_channel_scale=per_channel_scale,
      use_second_moments_msqe_opt=True,
      use_sqrt_of_msqe_weight=use_sqrt_of_msqe_weight,
      use_outlier_mask_msqe_weight=use_outlier_mask_msqe_weight)
  q.build(inputs.shape)
  q.scale_exponent.assign(scale_exponent)
  if q.use_second_moments_msqe_opt:
    q.msqe_weight.assign(msqe_weight)
  assert_array_equal(q.msqe_round(inputs, q.scale_exponent), msqe_round_output)


# Test quantized_bits_learnable_po2 _get_scale
@pytest.mark.parametrize(
    'use_po2_scale_ceil,'
    'use_po2_scale_msqe_round,'
    'switch_to_msqe_round,'
    'scale_exponent,'
    'inputs,'
    'scale_output,',
    [
        # Round po2 scale
        (False, False, # use_po2_scale_ceil, use_po2_scale_msqe_round
         True, # switch_to_msqe_round
         np.log2([[1.3, 3.1, 0.3]]), # scale_exponent (to be assigned)
         np.array([[6.0, 5.0, 1.0], [8.0, 3.0, -8.0]]), # inputs
         # Expected scale outputs
         np.array([[1.0, 4.0, 0.25]]), # Round po2 scale
         ),

        # Ceil po2 scale
        (True, False, # use_po2_scale_ceil, use_po2_scale_msqe_round
         True, # switch_to_msqe_round
         np.log2([[1.3, 3.1, 0.3]]), # scale_exponent (to be assigned)
         np.array([[6.0, 5.0, 1.0], [8.0, 3.0, -8.0]]), # inputs
         # Expected scale outputs
         np.array([[2.0, 4.0, 0.5]]), # Ceil po2 scale
         ),

        # Ceil po2 scale (MSQE round but switch_to_msqe_round is False)
        (True, True, # use_po2_scale_ceil, use_po2_scale_msqe_round
         False, # switch_to_msqe_round
         np.log2([[1.3, 3.1, 0.3]]), # scale_exponent (to be assigned)
         np.array([[6.0, 5.0, 1.0], [8.0, 3.0, -8.0]]), # inputs
         # Expected scale outputs
         np.array([[2.0, 4.0, 0.5]]), # Ceil po2 scale
         ),

        # Round po2 scale (MSQE round but switch_to_msqe_round is False)
        (False, True, # use_po2_scale_ceil, use_po2_scale_msqe_round
         False, # switch_to_msqe_round
         np.log2([[1.3, 3.1, 0.3]]), # scale_exponent (to be assigned)
         np.array([[6.0, 5.0, 1.0], [8.0, 3.0, -8.0]]), # inputs
         # Expected scale outputs
         np.array([[1.0, 4.0, 0.25]]), # Ceil po2 scale
         ),

        # MSQE round po2 scale
        (True, True, # use_po2_scale_ceil, use_po2_scale_msqe_round
         True, # switch_to_msqe_round
         np.log2([[1.3, 3.1, 0.3]]), # scale_exponent (to be assigned)
         np.array([[6.0, 5.0, 1.0], [8.0, 3.0, -8.0]]), # inputs
         # Expected scale outputs
         np.array([[2.0, 4.0, 0.5]]), # Ceil po2 scale
         ),

        # MSQE round po2 scale
        (True, True, # use_po2_scale_ceil, use_po2_scale_msqe_round
         True, # switch_to_msqe_round
         np.log2([[1.3, 3.1, 0.3]]), # scale_exponent (to be assigned)
         np.array([[6.0, 5.0, 0.3], [8.0, 3.0, -0.1]]), # inputs
         # Expected scale outputs
         np.array([[2.0, 4.0, 0.25]]), # Ceil po2 scale
         ),

        # Ceil po2 scale (MSQE round but switch_to_msqe_round is False)
        (True, True, # use_po2_scale_ceil, use_po2_scale_msqe_round
         False, # switch_to_msqe_round
         np.log2([[1.3, 3.1, 0.3]]), # scale_exponent (to be assigned)
         np.array([[6.0, 5.0, 0.3], [8.0, 3.0, -0.1]]), # inputs
         # Expected scale outputs
         np.array([[2.0, 4.0, 0.5]]), # Ceil po2 scale
         ),
    ])

def test_quantized_bits_learnable_po2_get_scale(
    use_po2_scale_ceil,
    use_po2_scale_msqe_round,
    switch_to_msqe_round,
    scale_exponent,
    inputs,
    scale_output,
):

  q = quantized_bits_learnable_po2(
      bits=4,
      keep_negative=True,
      scale_axis=None,
      per_channel_scale=True,
      use_second_moments_msqe_opt=True,
      use_po2_scale_ceil=use_po2_scale_ceil,
      use_po2_scale_msqe_round=use_po2_scale_msqe_round,
  )
  q.build(inputs.shape)

  if scale_exponent is not None:
    q.scale_exponent.assign(scale_exponent)

  if use_po2_scale_msqe_round:
    q.switch_to_msqe_round.assign(switch_to_msqe_round)

  assert_array_almost_equal(q._get_scale(inputs), scale_output)


# Test quantized_bits_learnable_po2 _quantize
@pytest.mark.parametrize(
    'init_scale,'
    'use_po2_scale_ceil,'
    'use_po2_scale_msqe_round,'
    'use_stable_scale_exponent,'
    'switch_to_stable_scale_0,'
    'switch_to_stable_scale_1,'
    'inputs_0,'
    'inputs_1,'
    'expected_inputs_q_output_0,'
    'expected_inputs_q_output_1,'
    'expected_scale,'
    'expected_scale_exponent,'
    # scale exponent optimized using 'inputs_1' and squared loss.
    'expected_scale_exponent_optimized,',
    [
        (None, # init_scale
         False, False, # use_po2_scale_ceil, use_po2_scale_msqe_round
         False, # use_stable_scale_exponent
         False, False, # switch_to_stable_scale_0, switch_to_stable_scale_1
         # inputs
         np.array([[6.0, 5.0, 1.0], [8.0, 3.0, -8.0]], dtype=np.float32), # 0
         np.array([[6.0, 5.0, 1.0], [1.0, 3.0, -1.0]], dtype=np.float32), # 1
         # expected quantizer outputs
         np.array([[6.0, 4.0, 0.0], [8.0, 4.0, -8.0]], dtype=np.float32), # 0
         np.array([[6.0, 4.0, 0.0], [0.0, 4.0, 0.0]], dtype=np.float32), # 1
         np.array([[2.0]]), # expected_scale
         np.array([[1.1528215056499365]]), # expected_scale_exponent
         np.array([[0.06]]), # expected_scale_exponent_optimized
         ),
        (None, # init_scale
         False, False, # use_po2_scale_ceil, use_po2_scale_msqe_round
         False, # use_stable_scale_exponent
         False, False, # switch_to_stable_scale_0, switch_to_stable_scale_1
         # inputs
         np.array([[6.0, 5.0, 1.0], [1.0, 3.0, -1.0]], dtype=np.float32), # 1
         np.array([[6.0, 5.0, 1.0], [8.0, 3.0, -8.0]], dtype=np.float32), # 0
         # expected quantizer outputs
         np.array([[6.0, 5.0, 1.0], [1.0, 3.0, -1.0]], dtype=np.float32), # 1
         np.array([[6.0, 5.0, 1.0], [7.0, 3.0, -7.0]], dtype=np.float32), # 0
         np.array([[1.0]]), # expected_scale
         np.array([[0.059999888055315004]]), # expected_scale_exponent
         np.array([[0.459674]]), # expected_scale_exponent_optimized
         ),
        (None, # init_scale
         False, False, # use_po2_scale_ceil, use_po2_scale_msqe_round
         False, # use_stable_scale_exponent
         False, False, # switch_to_stable_scale_0, switch_to_stable_scale_1
         # inputs
         np.array([[6.0, 5.0, 1.0], [1.0, 3.0, -1.0]], dtype=np.float32), # 1
         np.array([[6.0, 5.0, 1.0], [8.0, 3.0, -8.0]], dtype=np.float32), # 0
         # expected quantizer outputs
         np.array([[6.0, 5.0, 1.0], [1.0, 3.0, -1.0]], dtype=np.float32), # 1
         np.array([[6.0, 5.0, 1.0], [7.0, 3.0, -7.0]], dtype=np.float32), # 0
         np.array([[1.0]]), # expected_scale
         np.array([[0.059999888055315004]]), # expected_scale_exponent
         np.array([[0.459674]]), # expected_scale_exponent_optimized
         ),
        (2.0, # init_scale
         False, False, # use_po2_scale_ceil, use_po2_scale_msqe_round
         False, # use_stable_scale_exponent
         False, False, # switch_to_stable_scale_0, switch_to_stable_scale_1
         # inputs
         np.array([[6.0, 5.0, 1.0], [1.0, 3.0, -1.0]], dtype=np.float32), # 1
         np.array([[6.0, 5.0, 1.0], [8.0, 3.0, -8.0]], dtype=np.float32), # 0
         # expected quantizer outputs
         np.array([[6.0, 4.0, 0.0], [0.0, 4.0, -0.0]], dtype=np.float32), # 1
         np.array([[6.0, 4.0, 0.0], [8.0, 4.0, -8.0]], dtype=np.float32), # 0
         np.array([[2.0]]), # expected_scale
         np.array([[1.0]]), # expected_scale_exponent
         np.array([[0.306853]]), # expected_scale_exponent_optimized
         ),
    ])
def test_quantized_bits_learnable_po2_quantize(
    init_scale,
    use_po2_scale_ceil,
    use_po2_scale_msqe_round,
    use_stable_scale_exponent,
    switch_to_stable_scale_0,
    switch_to_stable_scale_1,
    inputs_0,
    inputs_1,
    expected_inputs_q_output_0,
    expected_inputs_q_output_1,
    expected_scale,
    expected_scale_exponent,
    expected_scale_exponent_optimized,
):

  q = quantized_bits_learnable_po2(
      bits=4,
      init_scale=init_scale,
      keep_negative=True,
      scale_axis=None,
      per_channel_scale=False,
      use_po2_scale_ceil=use_po2_scale_ceil,
      use_po2_scale_msqe_round=use_po2_scale_msqe_round,
      use_sqrt_of_msqe_weight=False,
      use_outlier_mask_msqe_weight=False,
      use_stable_scale_exponent=use_stable_scale_exponent,
  )
  q.build(inputs_0.shape)

  #####################
  # Initialization test
  #####################
  # Not yet initialized
  assert_array_equal(q.is_initialized.numpy(), False)

  if use_stable_scale_exponent:
    q.switch_to_stable_scale.assign(switch_to_stable_scale_0)

  inputs_q_0 = q._quantize(inputs_0, msqe_weight=None)
  assert_array_almost_equal(inputs_q_0, expected_inputs_q_output_0)

  # Initialized after the first '_quantize' run
  assert_array_equal(q.is_initialized.numpy(), True)

  if use_stable_scale_exponent:
    q.switch_to_stable_scale.assign(switch_to_stable_scale_1)

  inputs_q_1 = q._quantize(inputs_1, msqe_weight=None)
  assert_array_almost_equal(inputs_q_1, expected_inputs_q_output_1)

  # power-of-2 scale used to quantize the inputs
  assert_array_almost_equal(q.scale, expected_scale)

  # non power-of-2 scale exponent initialized from the first inputs
  assert_array_almost_equal(q.scale_exponent, expected_scale_exponent)

  ###################
  # Optimization test
  ###################
  optimizer = tf.keras.optimizers.SGD(
      learning_rate=1.0, nesterov=False, momentum=0.0)
  mse = tf.keras.losses.MeanSquaredError()

  with tf.GradientTape(persistent=False) as g:
    y = mse(q(inputs_1), inputs_1)
  optimizer.minimize(y, [q.scale_exponent], tape=g)
  assert_array_almost_equal(q.scale_exponent, expected_scale_exponent_optimized)


#############################################
# Test cases for quantized_bits_msqe_po2
#############################################

# Test quantized_bits_msqe_po2 build
@pytest.mark.parametrize(
    'bits, init_scale, keep_negative, per_channel_scale, scale_axis,'
    'use_second_moments_msqe_opt, second_moments_ema_decay,'
    'use_stable_scale_exponent, stable_scale_ema_decay,'
    'use_sqrt_of_msqe_weight,'
    'should_line_search,'
    'use_outlier_mask_msqe_weight,'
    'outlier_mask_sigma,'
    'line_search_range,'
    'num_lls_iters,'
    'inputs_shape,'
    'qn, qp,'
    'scaled_axes, reduce_axes,'
    'msqe_weight_shape, scale_exponent_shape,', [
        (
            3, 2.5, True, True, 3,
            True, 0.1,
            True, 0.1,
            True,
            True,
            True,
            1.23,
            11,
            12,
            [2, 3, 4, 5],
            3, 3,
            [0, 1, 2], [0, 1, 2],
            [2, 3, 4, 5], [1, 1, 1, 5],
        ),
        (
            3, 2.5, False, False, 3,
            False, 0.1,
            False, 0.1,
            False,
            False,
            False,
            1.23,
            11,
            12,
            [2, 3, 4, 5],
            0, 7,
            [0, 1, 2, 3], [0, 1, 2, 3],
            [2, 3, 4, 5], [1, 1, 1, 1],
        ),
        (
            3, 2.5, True, False, 3,
            False, 0.1,
            False, 0.1,
            False,
            False,
            False,
            1.23,
            11,
            12,
            [2, 3, 4, 5],
            3, 3,
            [0, 1, 2, 3], [0, 1, 2, 3],
            [2, 3, 4, 5], [1, 1, 1, 1],
        ),
    ])
def test_quantized_bits_msqe_po2_build(
    bits,
    init_scale,
    keep_negative,
    per_channel_scale,
    scale_axis,
    use_second_moments_msqe_opt,
    second_moments_ema_decay,
    use_stable_scale_exponent,
    stable_scale_ema_decay,
    use_sqrt_of_msqe_weight,
    should_line_search,
    use_outlier_mask_msqe_weight,
    outlier_mask_sigma,
    line_search_range,
    num_lls_iters,
    inputs_shape,
    qn,
    qp,
    scaled_axes,
    reduce_axes,
    msqe_weight_shape,
    scale_exponent_shape,
):

  q = quantized_bits_msqe_po2(
      bits=bits,
      init_scale=init_scale,
      keep_negative=keep_negative,
      per_channel_scale=per_channel_scale,
      scale_axis=scale_axis,
      use_second_moments_msqe_opt=use_second_moments_msqe_opt,
      second_moments_ema_decay=second_moments_ema_decay,
      use_stable_scale_exponent=use_stable_scale_exponent,
      stable_scale_ema_decay=stable_scale_ema_decay,
      use_sqrt_of_msqe_weight=use_sqrt_of_msqe_weight,
      should_line_search=should_line_search,
      use_outlier_mask_msqe_weight=use_outlier_mask_msqe_weight,
      outlier_mask_sigma=outlier_mask_sigma,
      line_search_range=line_search_range,
      num_lls_iters=num_lls_iters,
  )
  q.build(inputs_shape)

  # Attributes assignment and build tests
  assert_almost_equal(np.power(2.0, q.scale_exponent), init_scale)
  assert_equal(q.bits, bits)
  assert_equal(q.keep_negative, keep_negative)
  assert_equal(q.per_channel_scale, per_channel_scale)
  assert_equal(q.scale_axis, scale_axis)
  assert_equal(q.use_second_moments_msqe_opt, use_second_moments_msqe_opt)
  assert_equal(q.second_moments_ema_decay, second_moments_ema_decay)
  assert_equal(q.use_stable_scale_exponent, use_stable_scale_exponent)
  assert_equal(q.stable_scale_ema_decay, stable_scale_ema_decay)
  assert_equal(q.is_gradient_based, False)
  assert_equal(q.scale_exponent.trainable, False)
  assert_equal(q.qn, qn)
  assert_equal(q.qp, qp)
  assert_array_equal(q.scaled_axes, scaled_axes)
  assert_array_equal(q.reduce_axes, reduce_axes)
  if use_second_moments_msqe_opt:
    assert_array_equal(q.msqe_weight.shape, msqe_weight_shape)
  assert_array_equal(q.scale_exponent.shape, scale_exponent_shape)
  assert_equal(q.should_line_search, should_line_search)
  assert_equal(q.use_outlier_mask_msqe_weight, use_outlier_mask_msqe_weight)
  assert_equal(q.outlier_mask_sigma, outlier_mask_sigma)
  assert_equal(q.line_search_range, line_search_range)
  assert_equal(q.num_lls_iters, num_lls_iters)
  assert_equal(q.use_sqrt_of_msqe_weight, use_sqrt_of_msqe_weight)


# Test quantized_bits_msqe_po2 _get_init_scale_exponent
@pytest.mark.parametrize(
    'bits, keep_negative, scale_axis, per_channel_scale,'
    'inputs, init_scale_exponent',
    [
        (4, True, None, False,
         np.array([[6.0, 5.0, 1.0], [8.0, 3.0, -8.0]], dtype=np.float32),
         np.array([[0.0]], dtype=np.float32)),
        (4, True, None, False,
         np.array([[6.0, 5.0, 1.0], [16.0, 3.0, -8.0]], dtype=np.float32),
         np.array([[1.0]], dtype=np.float32)),
        (4, True, None, False,
         np.array([[6.0, 5.0, 1.0], [8.0, 3.0, -16.0]], dtype=np.float32),
         np.array([[1.0]], dtype=np.float32)),
        (4, True, 1, True,
         np.array([[6.0, 2.0, 1.0], [8.0, 3.0, -16.0]], dtype=np.float32),
         np.array([[0.0, -1.0, 1.0]], dtype=np.float32)),
    ])
def test_quantized_bits_msqe_po2_get_init_scale_exponent(
    bits,
    keep_negative,
    scale_axis,
    per_channel_scale,
    inputs,
    init_scale_exponent,
):

  q = quantized_bits_msqe_po2(
      bits=bits,
      keep_negative=keep_negative,
      scale_axis=scale_axis,
      per_channel_scale=per_channel_scale)
  q.build(inputs.shape)

  assert_array_equal(q._get_init_scale_exponent(inputs), init_scale_exponent)


# Test quantized_bits_msqe_po2 _get_outlier_mask
@pytest.mark.parametrize(
    'bits, keep_negative, scale_axis, per_channel_scale,'
    'inputs, outlier_mask_sigma, outlier_mask',
    [
        (4, True, None, False,
         np.array([[6.0, 5.0, 1.0], [8.0, 3.0, -8.0]], dtype=np.float32),
         9.0,
         np.ones(shape=(2,3), dtype=np.float32)),
        (4, True, None, False,
         np.array([[6.0, 5.0, 1.0], [8.0, 3.0, -8.0]], dtype=np.float32),
         2.0,
         np.ones(shape=(2,3), dtype=np.float32)),
        (4, True, None, False,
         np.array([[6.0, 5.0, 1.0], [8.0, 3.0, -8.0]], dtype=np.float32),
         1.0,
         np.array([[0.0, 1.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32),),
        (4, True, 1, True,
         np.array([[6.0, 5.0, 1.0], [8.0, 3.0, -8.0]], dtype=np.float32),
         1.0,
         np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32),),
        (4, True, 1, True,
         np.array([[6.0, 5.0, 1.0], [8.0, 3.0, -8.0]], dtype=np.float32),
         3.0,
         np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32),),
    ])
def test_quantized_bits_msqe_po2_get_outlier_mask(
    bits,
    keep_negative,
    scale_axis,
    per_channel_scale,
    inputs,
    outlier_mask_sigma,
    outlier_mask,
):

  q = quantized_bits_msqe_po2(
      bits=bits,
      keep_negative=keep_negative,
      scale_axis=scale_axis,
      per_channel_scale=per_channel_scale,
      outlier_mask_sigma=outlier_mask_sigma)
  q.build(inputs.shape)

  assert_array_equal(q._get_outlier_mask(inputs), outlier_mask)


# Test quantized_bits_msqe_po2 _get_scale
@pytest.mark.parametrize(
    'per_channel_scale,'
    'scale_exponent,'
    'scale,'
    'inputs,'
    'inputs_shape,'
    'expected_scale_output,',
    [
        (False,
         np.log2([[1.5]]),
         np.array([[1.6]]),
         np.array([[6.0, 5.0, 1.0], [8.0, 3.0, -8.0]], dtype=np.float32),
         [2,3],
         np.array([[1.0]], dtype=np.float32),
         ),
        (False,
         np.log2([[1.3]]),
         np.array([[1.6]]),
         None,
         [2,3],
         np.array([[2.0]], dtype=np.float32),
         ),
        (True,
         np.log2([[1.5,1.5,1.5]]),
         np.array([[1.6,1.6,1.6]]),
         np.array([[6.0, 5.0, 1.0], [8.0, 3.0, -8.0]], dtype=np.float32),
         [2,3],
         np.array([[2.0, 1.0, 2.0]], dtype=np.float32),
         ),
        (True,
         np.log2([[0.0012, 3.5, 0.00015]]),
         np.array([[1.6,1.6,1.6]]),
         np.array([[6.0, 5.0, 1.0], [5.0, 3.0, -8.5]], dtype=np.float32),
         [2,3],
         np.array([[1.0, 1.0, 2.0]], dtype=np.float32),
         ),
    ])

def test_quantized_bits_msqe_po2_get_scale(
    per_channel_scale,
    scale_exponent,
    scale,
    inputs, inputs_shape,
    expected_scale_output,
):

  q = quantized_bits_msqe_po2(
      bits=4,
      keep_negative=True,
      scale_axis=None,
      per_channel_scale=per_channel_scale,
      should_line_search=True,
      line_search_range=6,
      num_lls_iters=3,
      use_second_moments_msqe_opt=True,
      use_sqrt_of_msqe_weight=True,
      use_outlier_mask_msqe_weight=False,
      outlier_mask_sigma=3.0,
  )
  q.build(inputs_shape)

  if scale_exponent is not None:
    q.scale_exponent.assign(scale_exponent)

  if scale is not None:
    q.scale.assign(scale)

  assert_array_almost_equal(q._get_scale(inputs), expected_scale_output)


if __name__ == '__main__':
  googletest.main()
