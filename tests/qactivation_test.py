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
from tensorflow.keras import backend as K
import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import Callback
from keras.optimizers import SGD


from qkeras import set_internal_sigmoid
from qkeras import binary
from qkeras import hard_sigmoid
from qkeras import quantized_linear
from qkeras import quantized_bits
from qkeras import quantized_hswish
from qkeras import quantized_po2
from qkeras import quantized_relu
from qkeras import quantized_relu_po2
from qkeras import quantized_sigmoid
from qkeras import quantized_tanh
from qkeras import smooth_sigmoid
from qkeras import stochastic_binary
from qkeras import stochastic_ternary
from qkeras import ternary
from qkeras.quantizers import _default_sigmoid_type
from qkeras import QDense, QConv2D



@pytest.mark.parametrize(
    'bits, max_value, use_stochastic_rounding, quadratic_approximation, '
    'log2_rounding, test_values, expected_values', [
        # bits=4 without max_value. Therefore the max exponent is 4 when
        # quadratic approximiation is enabled. The max and min values from this
        # quantization function are 16 and -16 respectively.
        (
            4, None, 0, 1, "floor",
            np.array(
                [[-10.0, -0.25, 0.25, 1.0, 1.99, 2.0, 5.0, 10.0, 16.0, 32.0]],
                dtype=K.floatx()),
            np.array(
                [[-4.0, -0.25, 0.25, 1.0, 1.0, 1.0, 4.0, 4.0, 16.0, 16.0]],
                dtype=K.floatx()),
        ),
        # bits=3. The minimum exponent is -4. Therefore, the smallest absolute
        # value is 0.0625 in this quantization. The max absolute value is 0.5,
        # which is specified by the second input argument.
        (
            3, 0.5, 0, 0, "floor",
            np.array([[-7, -0.12, -0.03, 0.01, 5]], dtype=K.floatx()),
            np.array([[-0.5, -0.0625, -0.0625, 0.0625, 0.5]], dtype=K.floatx()),
        ),
        (8, None, 0, 0, "floor",
         np.array(
             [[-3, -2, -1.5, -0.5, -0.033, 0.5, 0.667, 1, 1.5, 4, 10]],
             dtype=K.floatx()),
         np.array(
             [[-2, -2, -1, -0.5, -0.03125, 0.5, 0.5, 1, 1, 4, 8]],
             dtype=K.floatx()),
        ),
        (4, None, 0, 0, "floor",
         np.array(
             [[-16, -7, -0.12, -0.03, 0, 0.01, 5, 10]],
             dtype=K.floatx()),
         np.array(
             [[-8, -4, -0.0625, -0.0625, 0.0625, 0.0625, 4, 8]],
             dtype=K.floatx()),
        ),
        (3, 0.5, 0, 0, "floor",
         np.array([[-7, -0.12, -0.03, 0.01, 5]], dtype=K.floatx()),
         np.array([[-0.5, -0.0625, -0.0625, 0.0625, 0.5]], dtype=K.floatx()),
        ),
        (4, 4, 0, 0, "floor",
         np.array([[-7, -0.12, -0.03, 0, 0.01, 5]], dtype=K.floatx()),
         np.array([[-4, -0.0625, -0.0625, 0.0625, 0.0625, 4]], dtype=K.floatx()),
        ),
        (4, None, 0, 1, "floor",
         np.array(
             [[0.01, 0.03, 0.06, 0.5, 1, 2, 5, 10, 16, 32]],
             dtype=K.floatx()),
         np.array(
             [[0.00390625, 0.015625, 0.015625, 0.25, 1, 1, 4, 4, 16, 16]],
             dtype=K.floatx()),
        ),
        (4, None, 0, 1, "floor",
         np.array(
             [[-32, -16, -10, -5, -2, -1, -0.5, -0.03, -0.01]],
             dtype=K.floatx()),
         np.array(
             [[-16, -16, -4, -4, -1, -1, -0.25, -0.015625, -0.00390625]],
             dtype=K.floatx()),
        ),
        (4, None, 0, 1, "floor",
         np.array(
             [[-32, -16, -10, -5, -2, -1, -0.5, -0.03, -0.01]],
             dtype=K.floatx()),
         np.array(
             [[-16, -16, -4, -4, -1, -1, -0.25, -0.015625, -0.00390625]],
             dtype=K.floatx()),
        ),
    ])
def disable_test_quantized_po2(bits,
                       max_value,
                       use_stochastic_rounding,
                       quadratic_approximation,
                       log2_rounding,
                       test_values,
                       expected_values):
  """Test quantized_po2 function."""
  x = K.placeholder(ndim=2)
  f = K.function([x], [quantized_po2(
      bits, max_value, use_stochastic_rounding,
      quadratic_approximation, log2_rounding)(x)])
  result = f([test_values])[0]
  assert_allclose(result, expected_values, rtol=1e-05, atol=1e-05)


@pytest.mark.parametrize(
    'bits, max_value, use_stochastic_rounding, quadratic_approximation, ' +
    'log2_rounding, test_values, expected_values',
    [
        # bits=3 without max_value. Therefore the max exponent is 4 when
        # quadratic approximiation is enabled. The max value from this
        # quantization function is 16. For the negative value, relu enforce it
        # to be the minimum value of this quantization function, which is 2**-4.
        (
            3, None, 0, 1, "floor",
            np.array(
                [[-10.0, -0.25, 0.25, 1.0, 1.99, 2.01, 5.0, 10.0, 16.0, 32.0]],
                dtype=K.floatx()),
            np.array(
                [[0.0625, 0.0625, 0.25, 1.0, 1.0, 1.0, 4.0, 4.0, 16.0, 16.0]],
                dtype=K.floatx()),
        ),
        # bits=3. The minimum exponent is -4. Therefore, the smallest absolute
        # value is 0.0625 in this quantization. The max absolute value is 4,
        # which is specified by the second input argument.
        (3, 4, 0, 0, "floor",
         np.array([[-7.0, -0.12, -0.03, 0, 0.01, 5.0]], dtype=K.floatx()),
         np.array([[0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 4.0]],
                  dtype=K.floatx())
        ),
        (8, None, 0, 0, "floor",
         np.array([[-0.033, 0.5, 0.667, 1, 1.5, 4, 10]], dtype=K.floatx()),
         np.array([[0, 0.5, 0.5, 1, 1, 4, 8]], dtype=K.floatx()),
        ),
        (3, None, 0, 0, "floor",
         np.array(
             [[-16.0, -7.0, -0.12, -0.03, 0, 0.01, 5.0, 10.0]],
             dtype=K.floatx()),
         np.array(
             [[0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 4.0, 8.0]],
             dtype=K.floatx()),
        ),
        (2, 0.5, 0, 0, "floor",
         np.array([[-7.0, -0.12, -0.03, 0.01, 5.0]], dtype=K.floatx()),
         np.array([[0.0625, 0.0625, 0.0625, 0.0625, 0.5]], dtype=K.floatx()),
        ),
        (3, 4, 0, 0, "floor",
         np.array(
             [[-7.0, -0.12, -0.03, 0, 0.01, 5.0]],
             dtype=K.floatx()),
         np.array(
             [[0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 4.0]],
             dtype=K.floatx()),
        ),
        (3, None, 0, 1, "floor",
         np.array(
             [[0.01, 0.03, 0.06, 0.5, 1, 2, 5, 10, 16, 32]],
             dtype=K.floatx()),
         np.array(
             [[0.00390625, 0.015625, 0.015625, 0.25, 1, 1, 4, 4, 16, 16]],
             dtype=K.floatx()),
        ),
    ])
def disable_test_quantized_relu_po2(bits, max_value, use_stochastic_rounding,
                                    quadratic_approximation, log2_rounding,
                                    test_values, expected_values):
  """Test quantized_po2 function."""
  x = K.placeholder(ndim=2)
  f = K.function([x],
                 [quantized_relu_po2(bits, max_value, 0,
                                     use_stochastic_rounding,
                                     quadratic_approximation,
                                     log2_rounding)(x)])
  result = f([test_values])[0]
  assert_allclose(result, expected_values, rtol=1e-05, atol=1e-05)


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
    'bits, sigmoid_type, use_real_sigmoid, test_values, expected_values', [
        (
            6,
            "hard",
            False,
            np.array(
                [[-1., -0.75, -0.5, -0.25,  0.,  0.25,  0.5,  0.75]],
                dtype=K.floatx()),
            np.array([[0.015625, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]],
                     dtype=K.floatx()),
        ),
        (
            6,
            "smooth",
            False,
            np.array(
                [[-1., -0.75, -0.5, -0.25,  0.,  0.25,  0.5,  0.75]],
                dtype=K.floatx()),
            np.array([[0.3125, 0.359375, 0.40625, 0.453125, 0.5, 0.546875, 0.59375, 0.640625]],
                     dtype=K.floatx()),
        ),
        (
            6,
            "real",
            True,
            np.array(
                [[-1., -0.75, -0.5, -0.25,  0.,  0.25,  0.5,  0.75]],
                dtype=K.floatx()),
            np.array([[0.265625, 0.328125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.671875]],
                     dtype=K.floatx()),
        ),
    ])
def test_quantized_sigmoid(bits, sigmoid_type, use_real_sigmoid, test_values, expected_values):
  """Test quantized_sigmoid function with three different sigmoid variants."""

  set_internal_sigmoid(sigmoid_type)
  x = K.placeholder(ndim=2)
  f = K.function([x], [quantized_sigmoid(bits, symmetric=True, use_real_sigmoid=use_real_sigmoid)(x)])
  set_internal_sigmoid(_default_sigmoid_type)

  result = f([test_values])[0]
  assert_allclose(result, expected_values, rtol=1e-05)


@pytest.mark.parametrize(
    'bits, sigmoid_type, use_real_sigmoid, test_values, expected_values', [
        (
            4,
            "hard",
            False,
            np.array(
                [-15,  15],
                dtype=K.floatx()),
            np.array([0.0625, 0.9375],
                     dtype=K.floatx()),
        ),
        (
            4,
            "smooth",
            False,
            np.array(
                [-15,  15],
                dtype=K.floatx()),
            np.array([0.0625, 0.9375],
                     dtype=K.floatx()),
        ),
        (
            4,
            "real",
            True,
            np.array(
                [-15,  15],
                dtype=K.floatx()),
            np.array([0.0625, 0.9375],
                     dtype=K.floatx()),
        ),
    ])

def test_quantized_sigmoid_limits(bits, sigmoid_type, use_real_sigmoid, test_values, expected_values):
  """Test the min and max values of quantized_sigmoid function with three different sigmoid variants."""

  set_internal_sigmoid(sigmoid_type)
  x = K.placeholder(ndim=2)
  f = K.function([x], [quantized_sigmoid(bits, symmetric=True, use_real_sigmoid=use_real_sigmoid)(x)])
  set_internal_sigmoid(_default_sigmoid_type)

  result = f([test_values])[0]
  min_max = np.array(
                    [quantized_sigmoid(bits, symmetric=True, use_real_sigmoid=use_real_sigmoid).min(),
                     quantized_sigmoid(bits, symmetric=True, use_real_sigmoid=use_real_sigmoid).max()])

  assert_allclose(result, expected_values, rtol=1e-05)
  assert_allclose(result, min_max, rtol=1e-05)


@pytest.mark.parametrize(
    'bits, use_real_tanh, test_values, expected_values', [
        (
            4,
            False,
            np.array(
                [[-1., -0.75, -0.5, -0.25,  0.,  0.25,  0.5,  0.75]],
                dtype=K.floatx()),
            np.array([[-0.875, -0.75, -0.5, -0.25,  0.,  0.25,  0.5,  0.75]],
                     dtype=K.floatx()),
        ),
        (
            4,
            True,
            np.array(
                [[-1., -0.75, -0.5, -0.25,  0.,  0.25,  0.5,  0.75]],
                dtype=K.floatx()),
            np.array([[-0.75, -0.625, -0.5, -0.25,  0.,  0.25,  0.5,  0.625]],
                     dtype=K.floatx()),
        )
    ])
def test_quantized_tanh(bits, use_real_tanh, test_values, expected_values):
  """Test quantized_tanh function with three different sigmoid variants."""
  # store previous sigmoid type

  set_internal_sigmoid('hard')
  x = K.placeholder(ndim=2)
  f = K.function([x], [quantized_tanh(bits, symmetric=True, use_real_tanh=use_real_tanh)(x)])
  set_internal_sigmoid(_default_sigmoid_type)

  result = f([test_values])[0]
  assert_allclose(result, expected_values, rtol=1e-05)


@pytest.mark.parametrize(
    'bits, sigmoid_type, use_real_tanh, test_values, expected_values', [
        (
            4,
            "hard",
            False,
            np.array(
                [-15, 15],
                dtype=K.floatx()),
            np.array([-0.875, 0.875],
                     dtype=K.floatx()),
        ),
        (
            4,
            "smooth",
            False,
            np.array(
                [-15, 15],
                dtype=K.floatx()),
            np.array([-0.875, 0.875],
                     dtype=K.floatx()),
        ),
        (
            4,
            "real",
            True,
            np.array(
                [-15, 15],
                dtype=K.floatx()),
            np.array([-0.875, 0.875],
                     dtype=K.floatx()),
        ),
    ])
def test_quantized_tanh_limits(bits, sigmoid_type, use_real_tanh, test_values, expected_values):
  """Test the min and max values of quantized_tanh function with three different sigmoid variants."""

  set_internal_sigmoid(sigmoid_type)
  x = K.placeholder(ndim=2)
  f = K.function([x], [quantized_tanh(bits, symmetric=True, use_real_tanh=use_real_tanh)(x)])
  set_internal_sigmoid(_default_sigmoid_type)

  result = f([test_values])[0]
  min_max = np.array(
                    [quantized_tanh(bits, symmetric=True, use_real_tanh=use_real_tanh).min(),
                     quantized_tanh(bits, symmetric=True, use_real_tanh=use_real_tanh).max()])

  assert_allclose(result, expected_values, rtol=1e-05)
  assert_allclose(result, min_max, rtol=1e-05)


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
    (
        "bits, integer, symmetric, keep_negative, test_values, expected_values,"
        " rtol"
    ),
    [
        (
            8,
            100,
            1,
            True,
            np.array([[1.25e+29, 3, -1.1e+30, 4.0e+32]], dtype=K.floatx()),
            np.array([[1.23794004e+29, 0.0, -1.09929075e+30, 1.26269884e+30]],
                     dtype=K.floatx()),
            5.0e+27,  # Effective quantization step size
        ),
        (
            6,
            2,
            0,
            True,
            np.array([[-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1, 4, 10]],
                     dtype=K.floatx()),
            np.array([[-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1, 3.875, 3.875]],
                     dtype=K.floatx()),
            1e-05,
        ),
        (
            6,
            2,
            0,
            False,
            np.array([[-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1, 4, 10]],
                     dtype=K.floatx()),
            np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1, 3.9375, 3.9375]],
                     dtype=K.floatx()),
            1e-05,
        ),
        (
            6,
            2,
            1,
            True,
            np.array([[-10, -4, -1.0, -0.5, 0.0, 0.5, 1, 4, 10]],
                     dtype=K.floatx()),
            np.array([[-3.875, -3.875, -1.0, -0.5, 0.0, 0.5, 1, 3.875, 3.875]],
                     dtype=K.floatx()),
            1e-05,
        )
    ])
def test_quantized_bits(bits, integer, symmetric, keep_negative, test_values,
                        expected_values, rtol):
  x = K.placeholder(ndim=2)
  f = K.function([x],
                 [quantized_bits(bits, integer, symmetric, keep_negative)(x)])
  result = f([test_values])[0]
  assert_allclose(result, expected_values, rtol=rtol)

class TestQuantizedLinear:
  """Tests for quantized_linear"""

  def test_sign_function(self):
    "Test to make sure that sign function is working properly"

    quantizer = quantized_linear(bits=1, keep_negative=True)
    x = tf.constant([-1.0, 0.0, 1.0, 2.0, 3.0])

    res = quantizer(x)
    expected_res = quantizer.max() * tf.constant([-1.0, 1.0, 1.0, 1.0, 1.0])

    assert tf.reduce_all(tf.equal(res, expected_res))

  @pytest.mark.parametrize('shape', [(1,), (2, 3), (2, 2, 4)])
  def test_scale_shape(self, shape):
    """Test to make sure that scale is the right shape for auto-alphas"""

    auto_quantizer = quantized_linear(alpha='auto')
    auto_po2_quantizer = quantized_linear(alpha='auto_po2')
    x = tf.ones(shape)
    auto_quantizer(x)
    auto_po2_quantizer(x)

    if len(shape) == 1:
      expected_shape = (1,)
    else:
      ones = [1 for _ in range(len(shape) - 1)]
      expected_shape = tuple(ones) + shape[-1:]

    assert auto_quantizer.scale.shape == expected_shape
    assert auto_po2_quantizer.scale.shape == expected_shape

  @pytest.mark.parametrize(
    'inputs, expected_auto_scale, expected_auto_po2_scale',
    [
      ( # rank 1
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [2.0, 4.0, 6.0, 8.0, 10.0],
        [2.0, 4.0, 8.0, 8.0, 8.0]
      ),
      ( # rank 2
        [[-1.0, 0.0, 1.0,  2.5,  4.0],
         [-1.0, 4.0, 5.0, -2.0, -1.0],
         [ 1.0, 2.0, 3.0,  0.0, 20.0]],
        [[ 2.0, 8.0, 10.0, 5.0, 40.0,]],
        [[ 2.0, 8.0,  8.0, 4.0, 32.0,]],
      ),
      ( # rank 3
        [[[0.0, 1.0], [2.0, 3.0]], 
         [[4.0, 5.0], [6.0, 7.0]]], 
        [[[12.0, 14.0]]],
        [[[16.0, 16.0]]],        
      )
    ]
  )
  def test_scale_values(
    self, inputs, expected_auto_scale, expected_auto_po2_scale):
    """
    Test to make sure that scale is the right value for auto-alphas
    
    Note that since bits=2, the data type scale will be 0.5. This means that
    the scale values will be 2x the quantization_scale values.
    """

    auto_quantizer = quantized_linear(alpha='auto', bits=2)
    auto_po2_quantizer = quantized_linear(alpha='auto_po2', bits=2)
    
    auto_quantizer(inputs)
    auto_po2_quantizer(inputs)

    tf.debugging.assert_equal(auto_quantizer.scale, expected_auto_scale)
    tf.debugging.assert_equal(auto_po2_quantizer.scale, expected_auto_po2_scale)

  @pytest.mark.parametrize('layer_type', ['QDense', 'QConv2D'])
  @pytest.mark.parametrize('alpha', ['auto', 'auto_po2'])
  def test_training_eval_equivalence(self, layer_type, alpha):
    """Test the behavior of quantizer during training and eval"""

    np.random.seed(42)

    # Define the quantizer
    quantizer = quantized_linear(alpha=alpha)
    model = Sequential()

    if layer_type == 'QConv2D':
      input_shape = (28, 28, 1) # Example shape for a grayscale image
      weight_shape = (3, 3, 1, 1)
      conv_layer = QConv2D(
        1, (3, 3), kernel_quantizer=quantizer, use_bias=False)
      model.add(conv_layer)
      # Create fake data with the corresponding shape
      X = np.random.rand(100, *input_shape)
    elif layer_type == 'QDense':
      input_shape = (2,) # Example shape for 2 input features
      weight_shape = (2, 3)
      dense_layer = QDense(
        3, input_shape=input_shape, kernel_quantizer=quantizer, use_bias=False)
      model.add(dense_layer)
      # Create fake data with the corresponding shape
      X = np.random.rand(100, *input_shape)

    # Set learning rate to zero
    opt = SGD(learning_rate=0.0)
    model.compile(optimizer=opt, loss='mse')

    # Initialize the weights
    model.build((None, *input_shape))
    initial_weights = np.random.rand(*weight_shape)
    model.layers[0].set_weights([initial_weights])

    # Create fake output data
    output_shape = model.output_shape[1:]
    y = np.random.rand(100, *output_shape)

    # Define a callback to capture weights during training
    class CaptureQuantizedWeightsCallback(Callback):
      def __init__(self):
          
        self.quantized_weights = []
        self.scales = []

      def on_train_batch_begin(self, batch, logs=None):
        weights = self.model.layers[0].get_weights()[0]
        qweights = self.model.layers[0].kernel_quantizer_internal(weights)
        scale = self.model.layers[0].kernel_quantizer_internal.scale
        self.quantized_weights.append(qweights)
        self.scales.append(scale)

    capture_weights_callback = CaptureQuantizedWeightsCallback()

    # Train the model
    model.fit(X, y, epochs=1, callbacks=[capture_weights_callback])

    # Capture the weights during evaluation (testing phase)
    weights = model.layers[0].get_weights()[0]
    eval_quantized_weights = model.layers[0].kernel_quantizer_internal(weights)
    eval_scale = model.layers[0].kernel_quantizer_internal.scale

    # Compare the weights during training and evaluation
    for train_quantized_weights in capture_weights_callback.quantized_weights:
      assert np.allclose(train_quantized_weights, eval_quantized_weights)

    # Compare the scales during training and evaluation
    for train_scale in capture_weights_callback.scales:
      assert np.allclose(train_scale, eval_scale)

  @pytest.mark.parametrize('alpha', [None, 2.0])
  @pytest.mark.parametrize('symmetric,keep_negative', 
                          [(True, True), (False, True), (False, False)])
  @pytest.mark.parametrize('bits', [1, 8])
  def test_gradient_formula(self, bits, symmetric, keep_negative, alpha):
    """Test to make sure that the gradient formula is correct"""

    quantizer = quantized_linear(bits=bits, symmetric=symmetric,
                                 keep_negative=keep_negative, alpha=alpha)
    x = tf.Variable([-1.0, 0.0, 1.0, 2.0, 3.0])

    with tf.GradientTape() as tape:
      res = quantizer(x)

    grad = tape.gradient(res, x)
    expected_grad = ((x >= quantizer.min()) & (x <= quantizer.max()))
    expected_grad = K.cast_to_floatx(expected_grad)
    assert grad is not None
    tf.debugging.assert_equal(grad, expected_grad)

  @pytest.mark.parametrize(
    'keep_negative, bits, expected_gradient',
    [(True, 1, [0, 1, 1, 1, 0]),
    (True, 8, [0, 1, 1, 1, 0]),
    (False, 1, [0, 0, 1, 1, 0]),
    (False, 8, [0, 0, 1, 1, 0])]
  )
  def test_gradients_explicit(self, keep_negative, bits, expected_gradient):
    """Tests on specific gradient values"""

    inputs = tf.Variable([-1.1, -0.1, 0.0, 0.1, 1.1])
    expected_gradient = tf.Variable(expected_gradient)

    q = quantized_linear(bits=bits, keep_negative=keep_negative)
    with tf.GradientTape() as tape:
      tape.watch(inputs)
      outputs = q(inputs)

    gradients = tape.gradient(outputs, inputs)
    assert_allclose(gradients, expected_gradient)

class TestBackwardsCompatibilityForQuantizedLinear:
  """Regression tests for quantized_linear, comparing to quantized_bits"""

  QUANTIZED_BITS_PARAMS = {
    "alpha": (None, "auto", "auto_po2"),
    "bits": (1, 4, 8),
    "integer": (0, 1),
    "symmetric": (True, False),
    "keep_negative": (True, False),
    "qnoise_factor": (1.0, 0.5, 0.0),
    "use_stochastic_rounding": (True, False),
    "scale_axis": (None, 0, 1),
  }

  TEST_X_VALUES = (
    0,
    *np.linspace(-2, 2, 10).tolist(),
    tf.random.uniform((2, )),
    tf.random.normal((2, 2)),
    tf.random.normal((2, 2, 2)),
  )

  # get list of kwargs for test iteration
  kwargs_list = []

  for param_name, param_values in QUANTIZED_BITS_PARAMS.items():
    for param_value in param_values:
      kwargs = {param_name: param_value}
      kwargs_list.append(kwargs)

  # extra kwargs for special cases
  extra_kwargs_list = [
      {
          "alpha": "auto",
          "symmetric": True,
          "keep_negative": True
      },
      {
          "alpha": "auto_po2",
          "symmetric": True,
          "keep_negative": True
      },
      {
          "alpha": "auto",
          "symmetric": True,
          "keep_negative": True,
          "integer": 2
      },
      {
          "alpha": "auto_po2",
          "symmetric": True,
          "keep_negative": True,
          "integer": 2
      },
  ]

  kwargs_list = extra_kwargs_list + kwargs_list

  @pytest.mark.parametrize('kwargs', kwargs_list)
  def test_regression(self, kwargs):
    """Check that the alt_quantized_bits and qkeras.quantized_bits
        return the same result for all test values"""

    bits = kwargs.get("bits", 8)
    integer = kwargs.get("integer", 0)
    keep_negative = kwargs.get("keep_negative", True)
    alpha = kwargs.get("alpha", None)
    symmetric = kwargs.get("symmetric", True)
    # defaults for quantized_bits and quantized_linear are different, need to 
    # specify in kwargs
    kwargs["symmetric"] = symmetric
    # variable to determine if checking for errors only, not correctness
    check_errors_only = False

    # decidedly raises an error
    if bits < integer + keep_negative:
      return
    # Not implemented in quantized_bits
    if alpha in ("auto", "auto_po2") and (not symmetric or not keep_negative):
      check_errors_only = True
    # bug in quantized_bits
    if bits - keep_negative == 0 and alpha in ("auto", "auto_po2"):
      check_errors_only = True
    # new implementation in quantized_linear
    if bits == 1 and keep_negative:
      check_errors_only = True

    old = quantized_bits(**kwargs)
    new = quantized_linear(**kwargs)

    for x in self.TEST_X_VALUES:
      # reset variable in loop
      check_errors_only_ = check_errors_only
      # bug in quantized_bits
      if tf.rank(x) == 0 and alpha in ("auto", "auto_po2"):
        continue
      # Changed default scale axis for rank-1 tensors
      if tf.rank(x) == 1 and alpha in ("auto", "auto_po2"):
        check_errors_only_ = True
      self._check_correctness(new, old, x, kwargs, 
                              check_errors_only=check_errors_only_)

  @pytest.mark.parametrize('kwargs', kwargs_list)
  def test_config(self, kwargs):

    symmetric = kwargs.get("symmetric", True)
    # defaults for quantized_bits and quantized_linear are different, need to 
    # specify in kwargs
    kwargs["symmetric"] = symmetric

    old = quantized_bits(**kwargs)
    new = quantized_linear(**kwargs)

    assert old.get_config() == new.get_config()

  @pytest.mark.parametrize('kwargs', kwargs_list)
  def test_string(self, kwargs):

    symmetric = kwargs.get("symmetric", True)
    # defaults for quantized_bits and quantized_linear are different, need to 
    # specify in kwargs
    kwargs["symmetric"] = symmetric

    old = quantized_bits(**kwargs)
    new = quantized_linear(**kwargs)

    old_str = str(old)
    new_str = str(new)

    old_str = old_str.replace("quantized_bits", "quantized_linear")

    assert old_str == new_str

  @pytest.mark.parametrize('qnoise_factor', [0.0, 0.5, 1.0])
  def test_qnoise_and_gradient(self, qnoise_factor):
    """Make sure that gradient calculations vary w.r.t. qnoise correctly"""

    qlinear = quantized_linear(qnoise_factor=qnoise_factor)
    qbits = quantized_bits(qnoise_factor=qnoise_factor)

    x = np.linspace(
      qlinear.min() + K.epsilon(), 
      qlinear.max() - K.epsilon(), 
      10)
    x = tf.Variable(x)

    with tf.GradientTape() as tape:
      res_linear = qlinear(x)

    grad_linear = tape.gradient(res_linear, x)

    with tf.GradientTape() as tape:
      res_bits = qbits(x)

    grad_bits = tape.gradient(res_bits, x)
    tf.debugging.assert_equal(grad_linear, grad_bits)

  def _check_correctness(self, new_func, old_func, x, kwargs, 
                         check_errors_only=False):
    """Check that the new_func and old_func return the same result for x"""

    old_res = old_func(x).numpy()
    new_res = new_func(x).numpy()
    old_scale = np.array(old_func.scale)
    new_scale = np.array(new_func.scale)

    # not checking if new matches old
    if check_errors_only:
      return
    err_msg = (f"Failed for {kwargs} with x = {x}. \n"
              f"old_res = {old_res}, alt_res = {new_res}. \n"
              f"old_scale = {old_scale}, new_scale = {new_scale}")
    if not np.allclose(old_res, new_res):
      assert False, err_msg
    if not np.allclose(old_scale, new_scale) and K.max(x) > 0:
      assert False, err_msg


@pytest.mark.parametrize('alpha, threshold, test_values, expected_values', [
    (1.0, 0.33,
     np.array([[-3.0, -2.0, -1.0, -0.2, 0.0, 0.3, 1, 4, 10]], dtype=K.floatx()),
     np.array([[-1.0, -1.0, -1.0, 0, 0.0, 0.0, 1, 1, 1]], dtype=K.floatx())),
    (10.0, 5.0,
     np.array([[-11.0, -7.0, -4.0, -0.2, 0.0, 0.3, 1, 4, 10]],
              dtype=K.floatx()),
     np.array([[-10.0, -10.0, 0.0, 0, 0.0, 0.0, 0, 0, 10]], dtype=K.floatx())),
])
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


@pytest.mark.parametrize('test_values, expected_values', [
    (np.array([[42.0] * 100000], dtype=K.floatx()), 42.0),
    (np.array([[100.0] * 100000], dtype=K.floatx()), 100.0),
    (np.array([[48.0] * 100000], dtype=K.floatx()), 48.0),
    (np.array([[-141.0] * 100000], dtype=K.floatx()), -141.0),
    (np.array([[-32.0] * 100000], dtype=K.floatx()), -32.0),
    (np.array([[32.0] * 100000], dtype=K.floatx()), 32.0),
    (np.array([[10031.0] * 100000], dtype=K.floatx()), 10031.0),
    (np.array([[0.0] * 100000], dtype=K.floatx()), 0.0),
])
def test_stochastic_round_quantized_po2(test_values, expected_values):
  K.set_learning_phase(1)
  np.random.seed(666)
  x = K.placeholder(ndim=2)
  q = quantized_po2(use_stochastic_rounding=True)
  f = K.function([x], [q(x)])
  res = f([test_values])[0]
  res = np.average(res)
  assert_allclose(res, expected_values, rtol=1e-01, atol=1e-6)


@pytest.mark.parametrize('test_values, expected_values', [
    (np.array([[42.0] * 100000], dtype=K.floatx()), 42.0),
    (np.array([[-42.0] * 100000], dtype=K.floatx()), 0.0),
    (np.array([[0.0] * 100000], dtype=K.floatx()), 0.0),
    (np.array([[100.0] * 100000], dtype=K.floatx()), 100.0),
    (np.array([[48.0] * 100000], dtype=K.floatx()), 48.0),
])
def test_stochastic_round_quantized_relu_po2(test_values, expected_values):
  K.set_learning_phase(1)
  np.random.seed(666)
  x = K.placeholder(ndim=2)
  q = quantized_relu_po2(use_stochastic_rounding=True)
  f = K.function([x], [q(x)])
  res = f([test_values])[0]
  res = np.average(res)
  assert_allclose(res, expected_values, rtol=1e-01, atol=1e-6)


def test_stochastic_binary():
  np.random.seed(42)
  K.set_learning_phase(1)

  x = np.random.uniform(-0.01, 0.01, size=10)
  x = np.sort(x)
  # Adding a dimension to have a common channel axis for quantization. This is
  # to cope with a bug fix in "_get_scale" without changing the test cases.
  x = np.expand_dims(x, axis=1)

  s = stochastic_binary(alpha="auto_po2")

  ty = np.zeros_like(s)
  ts = 0.0

  n = 1000

  for _ in range(n):
    y = K.eval(s(K.constant(x)))
    scale = K.eval(s.scale)[0]
    ts = ts + scale
    ty = ty + (y / scale)

  # Perform squeezing to remove the common channel axis.
  result = (ty/n).astype(np.float32)
  result = np.squeeze(result)
  scale = np.array([ts/n])
  scale = np.squeeze(scale)

  expected = np.array(
      [-1., -1., -1., -0.852, 0.782, 0.768, 0.97, 0.978, 1.0, 1.0]
  ).astype(np.float32)
  expected_scale = np.array([0.003906])

  assert_allclose(result, expected, atol=0.1)
  assert_allclose(scale, expected_scale, rtol=0.1)


@pytest.mark.parametrize('alpha, test_values, expected_values', [
    (1.0,
     np.array([[-3.0, -2.0, -1.0, -0.2, 0.0, 0.3, 1, 4, 10]], dtype=K.floatx()),
     np.array([[-1.0, -1.0, -1.0, -1.0, 1, 1, 1, 1, 1]], dtype=K.floatx())),
    (5.0,
     np.array([[-11.0, -7.0, -4.0, -0.2, 0.0, 0.3, 1, 4, 10]],
              dtype=K.floatx()),
     np.array([[-5.0, -5.0, -5.0, -5, 5.0, 5.0, 5, 5, 5]], dtype=K.floatx()))
])
def test_stochastic_binary_inference_mode(alpha, test_values, expected_values):
  K.set_learning_phase(0)
  x = K.placeholder(ndim=2)
  q = stochastic_binary(alpha)
  f = K.function([x], [q(x)])
  result = f([test_values])[0]
  assert_allclose(result, expected_values, rtol=1e-05)


@pytest.mark.parametrize(
    'bound, alpha, temperature, expected_values, expected_scale', [
    (
        0.01,
        "auto",
        8,
        np.array([-0.973, -0.903, -0.759, -0.574, -0.242,  0.161,  0.508,  0.723,
            0.874,  0.975]).astype(np.float32),
        np.array([0.008427, 0.007001, 0.0057  , 0.004457, 0.003537, 0.003416,
            0.004507, 0.005536, 0.006853, 0.008282]).astype(np.float32)
    ),
    (
        0.01,
        "auto_po2",
        8,
        np.array([-0.979, -0.877, -0.639, -0.586, -0.23 ,  0.154,  0.327,  0.603,
            0.83 ,  0.986]).astype(np.float32),
        np.array([0.007812, 0.007812, 0.007812, 0.003906, 0.003906, 0.003906,
            0.007812, 0.007812, 0.007812, 0.007812]).astype(np.float32)
    )
])
def test_stochastic_ternary(bound, alpha, temperature, expected_values, expected_scale):
  np.random.seed(42)
  K.set_learning_phase(1)

  n = 1000

  x = np.random.uniform(-bound, bound, size=(n, 10))
  x = np.sort(x, axis=1)

  s = stochastic_ternary(alpha=alpha, temperature=temperature)

  y = K.eval(s(K.constant(x)))
  scale = K.eval(s.scale).astype(np.float32)[0]

  ty = np.zeros_like(s)
  for i in range(n):
    ty = ty + (y[i] / scale)

  result = (ty/n).astype(np.float32)

  assert_allclose(result, expected_values, atol=0.1)
  assert_allclose(scale, expected_scale, rtol=0.1)


@pytest.mark.parametrize('alpha, threshold, test_values, expected_values', [
    (1.0, 0.33,
     np.array([[-3.0, -2.0, -1.0, -0.2, 0.0, 0.3, 1, 4, 10]], dtype=K.floatx()),
     np.array([[-1.0, -1.0, -1.0, 0, 0.0, 0.0, 1, 1, 1]], dtype=K.floatx())),
    (10.0, 5.0,
     np.array([[-11.0, -7.0, -4.0, -0.2, 0.0, 0.3, 1, 4, 10]],
              dtype=K.floatx()),
     np.array([[-10.0, -10.0, 0.0, 0, 0.0, 0.0, 0, 0, 10]], dtype=K.floatx())),
])
def test_stochastic_ternary_inference_mode(alpha, threshold, test_values, expected_values):
  K.set_learning_phase(0)
  x = K.placeholder(ndim=2)
  q = stochastic_ternary(alpha, threshold)
  f = K.function([x],
                 [q(x)])
  result = f([test_values])[0]
  assert_allclose(result, expected_values, rtol=1e-05)


@pytest.mark.parametrize(
    # y = x * relu6(x+3)/6, the total world length is 6 bits with 2 integer
    # bits. The quantization is in asymmetric mode.
    ('bits, integer, symmetric, relu_shift, relu_upper_bound,'
     'test_values, expected_values'), [
         (
          6, 2, 0, 3, 6,
          np.array([[-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1, 4, 10]],
                   dtype=K.floatx()),
          np.array([[0., -0.375, -0.375, -0.25,  0., 0.25, 0.625,
                     3.875, 3.875]], dtype=K.floatx()),
         ),
         (
          6, 4, 1, 3, 6,
          np.array([[-10.0, -2.0, -2.3, -0.25, 0.0, 0.5, 1, 4, 10]],
                   dtype=K.floatx()),
          np.array([[0., -0.5, -0.5, 0., 0., 0.5, 0.5, 4., 10.]],
                   dtype=K.floatx()),
         ),
         (
          2, 0, 0, 3, 6,
          np.array([[-10.0, -2.0, -2.3, -0.25, 0.0, 0.5, 1, 4, 10]],
                   dtype=K.floatx()),
          np.array([[0., -0.5, -0.5, 0., 0., 0.5, 0.5, 0.5, 0.5]],
                   dtype=K.floatx()),
         ),
        ])
def test_quantized_hswish(bits, integer, symmetric, relu_shift,
                          relu_upper_bound, test_values, expected_values):
  x = K.placeholder(ndim=2)
  f = K.function(
      [x], [quantized_hswish(bits, integer, symmetric,relu_shift=relu_shift,
                           relu_upper_bound=relu_upper_bound)(x)])
  result = f([test_values])[0]
  assert_allclose(result, expected_values, rtol=1e-05)


if __name__ == '__main__':
  pytest.main([__file__])
