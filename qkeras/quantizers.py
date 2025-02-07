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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from typing import Any, List, Tuple, cast

import numpy as np
import six
from six.moves import range
import tensorflow.compat.v2 as tf
from tensorflow.keras import initializers
import tensorflow.keras.backend as K
from tensorflow.keras.utils import deserialize_keras_object

from . import base_quantizer
from . import quantizer_registry
# from .google_internals.experimental_quantizers import parametric_quantizer_d_xmax
# from .google_internals.experimental_quantizers import quantized_bits_learnable_scale
from .safe_eval import safe_eval
from tensorflow.python.framework import smart_cond as tf_utils

#
# Library of auxiliary functions
#


def get_weight_scale(quantizer, x=None):
  """Gets the scales of weights for (stochastic_)binary and ternary quantizers.

  Arguments:
    quantizer: A binary or teneray quantizer class.
    x: A weight tensor.  We keep it here for now for backward compatibility.

  Returns:
    Weight scale per channel for binary and ternary
    quantizers with auto or auto_po2 alpha/threshold.
  """
  if hasattr(quantizer, "scale") and quantizer.scale is not None:
    return K.eval(quantizer.scale)
  return 1.0


def _get_integer_bits(min_value,
                      max_value,
                      bits=8,
                      symmetric=False,
                      keep_negative=False,
                      is_clipping=True):
  """Estimates the integer bit(number of bits to the left of the binary point)
  satisfying the input argument constraints.

  Args:
    min_value: A tensor object. Its elements are in float representing the
      minimum values of ranges.
    max_value: A tensor object. Its elements are in float representing the
      maximum values of ranges.
    bits: number of bits to perform quantization.
    symmetric: boolean type. if true, it enforces negative and positive ranges
      to be symmetric.
    keep_negative: boolean type. if true, we do not clip negative numbers.
    is_clipping: boolean type. if true, the min_value and max_value are clipped
      to nearest powers-of-2.

  Returns:
    integer_bits : number of bits to the left of the binary point.
  """
  # Max the min and max values positive if only using positive values
  if not keep_negative:
    min_value = K.maximum(min_value, 0)
    max_value = K.maximum(max_value, 0)

  # The number of bits excluding the sign bit
  unsigned_bits = bits - keep_negative

  # log2 of absolute min_value and max_value
  min_value_log2 = K.log(K.abs(min_value)) / np.log(2.0)
  max_value_log2 = K.log(K.abs(max_value)) / np.log(2.0)

  # Estimate integer_bits
  if is_clipping:
    min_int_bits = tf.math.round(
        tf.where(min_value_log2 > 0, min_value_log2, 0))
    max_int_bits = tf.math.round(
        tf.where(max_value_log2 > 0, max_value_log2, 0))
  else:
    min_int_bits = tf.math.ceil(tf.where(min_value_log2 > 0, min_value_log2, 0))
    max_int_bits = tf.math.ceil(tf.where(max_value_log2 > 0, max_value_log2, 0))
    # Checks max_value is bounded by the maximum positive value of
    # pow(2,integer_bits) - pow(2,-fractional_bits).
    max_value_po2 = pow(2.0, max_int_bits) - pow(
        2.0, K.minimum(max_int_bits - unsigned_bits, 0))
    max_int_bits = tf.where(max_value <= max_value_po2, max_int_bits,
                            max_int_bits + 1)
    if symmetric:
      # Checks min_value is bounded by the minimum negative value of
      # - pow(2,integer_bits) + pow(2,-fractional_bits).
      min_value_po2 = -pow(2.0, min_int_bits) + pow(
          2.0, K.minimum(min_int_bits - unsigned_bits, 0))
      min_int_bits = tf.where(min_value_po2 <= min_value, min_int_bits,
                              min_int_bits + 1)

  # To cover both negative and positive ranges with integer_bits.
  # (For keep_negative=False, min_int_bits is 0.)
  integer_bits = tf.cast(K.maximum(min_int_bits, max_int_bits), dtype=tf.int32)
  # It assumes that integer_bits cannot be greater than unsigned_bits
  integer_bits = K.minimum(unsigned_bits, integer_bits)

  return integer_bits


def _get_scaling_axis(scale_axis: Any, len_axis: int) -> List[int]:
  """Get the axis/axes to perform auto scaling at.

  Args:
    scale_axis: int or List[int] representing which axis/axes to calculate
     scale at.
    len_axis: int representing the shape of the tensor on which scaling is
      performed.

  Returns:
    List[int] representing the scaling axes.

  """

  if scale_axis is not None:
    # if scale_axis is set, scale over all axis except the scale_axis.
    if isinstance(scale_axis, list):
      axis = [i for i in range(len_axis) if i not in scale_axis]
    else:
      axis = tf.range(scale_axis)
      axis = tf.concat([axis, tf.range(scale_axis + 1, len_axis)], axis=0)
  else:
    # if scale_axis is not set, scale over all axis except the channel axis.
    if K.image_data_format() == "channels_last":
      axis = tf.range(tf.math.maximum(len_axis - 1, 0))
    else:
      axis = tf.range(1, len_axis)
  return axis


def _get_unrolled_shape(input_shape: List[int], unroll_factor: Any,
                        unroll_axis: Any) -> Tuple[List[int], Any]:
  """Gets the shape of the unrolled tensor given unroll_factor and unroll_axis.

  Both unroll_factor and unroll_axis can either be ints or List[int]. If they
  are List[int], their lengths must match, and their values represent every
  unroll axis and its corresponding unroll factor.

  Examples:
  1. If input_shape = [16, 32], the unroll_factor = 4, and unroll_axis = 1. This
     means that axis 1 of the input should be unrolled by a factor of 4. This
     function would return a tuple; the first element represents the unrolled
     shape [16, 8, 4], and the second element represents the updated unroll axis
     in the unrolled shape which, in this case, is still 1.
  2. If input_shape = [16, 32], the unroll_factor = [2, 4], and unroll_axis =
     [0, 1]. This means that axis 0 of the input should be unrolled by a factor
     of 2, while axis 1 of the input should be unrolled by a factor of 4. This
     function would return a tuple; the first element represents the unrolled
     shape [4, 2, 8, 4], and the second element represents the updated unroll
     axis in the unrolled shape which, in this case, will be [0, 2].

  Args:
    input_shape: List[int]. The shape of the input tensor to be unrolled.
    unroll_factor: int or List[int] representing the unrolling factor(s) across
      various dimensions of the input tensors. If a list is used, its length has
      to match unroll_axis.
    unroll_axis: int or List[int] representing which axis/axes to unroll. If
      a list is used, its length has to match unroll_factor.

  Returns:
    Tuple of (List of ints representing the shape of the unrolled tensor,
    Int or List[int] representing updated scale_axis after unrolling.
  """
  def _unroll_one_axis(shape, factor, axis):
    shape[axis] = shape[axis] // factor
    shape.insert(axis + 1, factor)

  unrolled_shape = input_shape.copy()

  if isinstance(unroll_factor, int) and isinstance(unroll_axis, int):
    unrolled_scale_axis = unroll_axis
    _unroll_one_axis(unrolled_shape, unroll_factor, unroll_axis)

  elif isinstance(unroll_factor, list) and isinstance(unroll_axis, list):
    # axis_shift shifts the pre-defined axis every time we add a new
    # unrolled axis
    assert len(unroll_axis) == len(unroll_factor), (
        "unroll_axis and unroll_factor must have the same length")

    unrolled_scale_axis = unroll_axis.copy()
    axis_shift = 0
    for idx, (axis, factor) in enumerate(zip(unroll_axis, unroll_factor)):
      unrolled_scale_axis[idx] += axis_shift
      _unroll_one_axis(unrolled_shape, factor, axis+axis_shift)
      axis_shift += 1
  else:
    raise ValueError(
        "Both unroll_factor and unroll_axis has to be either ints or lists"
    )
  return unrolled_shape, unrolled_scale_axis


def _get_rolled_back_shape(input_shape: List[int], roll_axis: Any) -> List[int]:
  """Gets the shape of the rolled back tensor given roll_axis.

  If roll_axis is an int, the input shape will be rolled back once along the
  roll_axis. If roll_axis is List[int], the input shape will be rolled back
  len(roll_axis) times.

  Examples:
  1. If input_shape = [4, 2, 8, 4] and roll_axis = 1. This means that the axis
     following axis 1 will be rolled back to axis 1. This function would return
     a the rolled back shape which is [4, 16, 4] in this case.
  2. If input_shape = [4, 2, 8, 4] and roll_axis = [0, 2]. This means that the
     axis following axis 0 will be rolled back to axis 0, and the axis following
     axis 2 will be rolled back to axis 2. This function would return the rolled
     back shape which is [16, 32] in this case.

  Args:
    input_shape: List[int]. The shape of the input tensor to be rolled back.
    roll_axis: int or List[int] representing which axis/axes of the tensor to
      roll back.

  Returns:
    List of ints representing the shape of the rolled back tensor.
  """
  def _roll_back_one_axis(shape, axis):
    shape[axis] *= shape[axis+1]
    shape.pop(axis + 1)

  rolled_shape = input_shape.copy()

  if isinstance(roll_axis, int):
    _roll_back_one_axis(rolled_shape, roll_axis)

  elif isinstance(roll_axis, list):
    # axis_shift shifts the pre-defined axis every time we roll back an axis.
    axis_shift = 0
    for axis in roll_axis:
      _roll_back_one_axis(rolled_shape, axis+axis_shift)
      axis_shift -= 1

  return rolled_shape


def _validate_axis_and_eps(x_shape: List[int], scale_axis: Any,
                           elements_per_scale: Any) -> Tuple[Any, Any]:
  """Validates scale_axis and elements_per_scale.

  This function verifies that the values for scale_axis and elements_per_scale
  are valid and perform any required transformations returning a Tuple of
  verified (scale_axis, elements_per_scale)

  This fuction accepts scale_axis and elements_per_scale to be either ints or
  list of ints, so it verifies 4 different scenarios:
  1. If both scale_axis and elements_per_scale are ints. The function verifies
     that the x_shape is divisible by elements_per_scale at the scale_axis.
  2. If scale_axis is an int while elements_per_scale is a list. The function
     raises an error since this is an ambigious state.
  3. If scale_axis is a list and elements_per_scale is an int. The function
     modifies elements_per_scale to a list of length scale_axis, and it verifies
     that the x_shape is divisible by the elements_per_scale at the
     corresponding scale_axis.
  4. If scale_axis is a list and elements_per_scale is a list. The function
     verifies that the length of the two lists match, and that the x_shape is
     divisible by the corresponding elements_per_scale at the corresponding
     scale_axis.

  Examples:
  - Input_shape=[16, 32, 4], scale_axis=0, and elements_per_scale=4 --> Valid
  - Input_shape=[16, 32, 4], scale_axis=0, and elements_per_scale=3 --> Invalid
  - Input_shape=[16, 32, 4], scale_axis=0, and elements_per_scale=[2, 4]
    --> Invalid
  - Input_shape=[16, 32, 4], scale_axis=[0, 1], and elements_per_scale=2
    --> Valid
  - Input_shape=[16, 32, 4], scale_axis=[0, 1], and elements_per_scale=[2, 4]
    --> Valid
  - Input_shape=[16, 32, 4], scale_axis=[0, 1], and elements_per_scale=[1, 2, 4]
    --> Invalid

  Args:
    x_shape: List[int] representing the shape of the input tensor.
    scale_axis: Int or List[int] representing the axis/axes to perform auto
      scaling at.
    elements_per_scale: Int or List[int] representing the number of
     elements/values associated with every scale along the corresponding
     scale_axis.

  Returns:
    A Tuple of verified (scale_axis, elements_per_scale).
  """

  assert (
      scale_axis is not None
  ), "scale_axis must be set if elements_per_scale is used."

  # if both are ints
  if isinstance(scale_axis, int) and isinstance(elements_per_scale, int):
    assert x_shape[scale_axis] % elements_per_scale == 0, (
        f"scaling axis of dimension {x_shape[scale_axis]} has to be divisible "
        f"by thenumber of elements per scale, given {elements_per_scale}."
    )

  # if scale_axis is int and elements_per_scale is a list of ints
  elif isinstance(scale_axis, int) and isinstance(elements_per_scale, list):
    raise ValueError(
        f"scale_axis is an integer {scale_axis}, "
        f"while {elements_per_scale} is a list of values which is ambigious."
    )

  # if scale_axis is list of ints and elements_per_scale is an int
  elif isinstance(scale_axis, list) and isinstance(elements_per_scale, int):
    for axis in scale_axis:
      assert x_shape[axis] % elements_per_scale == 0, (
          f"scaling axis of dimension {x_shape[axis]} has to be divisible by "
          f"number of elements per scale, given {elements_per_scale}."
      )
    # duplicate the elements_per_scale to match length of scale_axis
    elements_per_scale = [elements_per_scale] * len(scale_axis)

  # if both scale_axis and elements_per_scale are lists
  else:
    assert len(scale_axis) == len(
        elements_per_scale
    ), (f"both scale_axis and elements_per_scale lists must match in length; "
        f"Got {len(scale_axis)} and {len(elements_per_scale)}")
    for axis, eps in zip(scale_axis, elements_per_scale):
      assert x_shape[axis] % eps == 0, (
          f"scaling axis of dimension {x_shape[axis]} has to be divisible by"
          f" the corresponding number of elements per scale, given {eps}."
      )

  assert (
      isinstance(scale_axis, int) and isinstance(elements_per_scale, int)
  ) or (isinstance(scale_axis, list) and isinstance(elements_per_scale, list))

  return scale_axis, elements_per_scale


def _repeat_along_axis(x: tf.Tensor, axis: int, repeats: int) -> tf.Tensor:
  """Repeats the elements in a tensor along the specified axis."""
  return tf.repeat(x, repeats=repeats, axis=axis)


def _repeat_along_axes(x: tf.Tensor, axis: Any, repeats: Any) -> tf.Tensor:
  """Repeats the elements in a tensor along the specified axes."""
  if isinstance(axis, int) and isinstance(repeats, int):
    x = _repeat_along_axis(x, axis, repeats)
  elif isinstance(axis, list) and isinstance(repeats, list):
    for a, r in zip(axis, repeats):
      x = _repeat_along_axis(x, axis=a, repeats=r)
  return x


def _get_scale_mean(
    scale_axis: Any, x: tf.Tensor, q: tf.Tensor, elements_per_scale: Any
):
  """Gets the mean of the tensor along the specified scaling axis/axes.

  Args:
    scale_axis: int or List[int] representing which axis/axes to calculate
     scale at.
    x: A tensor object. Its elements are in float.
    q: A tensor object. Its elements are in quantized format of x.
    elements_per_scale: if set to an int or List[int], we create multiple scales
      per axis across scale_axis, where 'elements_per_scale' represents the
      number of elements/values associated with every separate scale value.

  Returns:
    A tuple of two tensors representing the mean of x and its quantized format
    along the specified scaling axis/axes.
  """
  if elements_per_scale is not None:
    # Get the input shape
    x_shape = x.shape.as_list()

    scale_axis, elements_per_scale = _validate_axis_and_eps(
        x_shape, scale_axis, elements_per_scale)

    # get the shape of unrolled tensors x and q
    unrolled_shape, unrolled_scale_axis = _get_unrolled_shape(
        x_shape, elements_per_scale, scale_axis)

    # Unroll x and q
    x1 = tf.reshape(x, unrolled_shape)
    q1 = tf.reshape(q, unrolled_shape)

    # Get the mean along the unroll axis/axes
    axes_of_mean = _get_scaling_axis(unrolled_scale_axis, len(unrolled_shape))
    qx = K.mean(tf.math.multiply(x1, q1), axis=axes_of_mean, keepdims=True)
    qq = K.mean(tf.math.multiply(q1, q1), axis=axes_of_mean, keepdims=True)

    # Reshape qx and qq to be divisible by the input shape.
    # To achieve this, qx and qq are first rolled back along unroll axis.
    # Then, the values along the scale_axis are repeated "elements_per_scale"
    # times to match the original shape.
    rolled_back_shape = _get_rolled_back_shape(qx.shape.as_list(),
                                               roll_axis=unrolled_scale_axis)

    qx = tf.reshape(qx, rolled_back_shape)
    qx = _repeat_along_axes(qx, repeats=elements_per_scale, axis=scale_axis)

    qq = tf.reshape(qq, rolled_back_shape)
    qq = _repeat_along_axes(qq, repeats=elements_per_scale, axis=scale_axis)
  else:
    len_axis = len(x.shape)
    axis = _get_scaling_axis(scale_axis, len_axis)
    qx = K.mean(tf.math.multiply(x, q), axis=axis, keepdims=True)
    qq = K.mean(tf.math.multiply(q, q), axis=axis, keepdims=True)
  return qx, qq


def _clip_po2_scale(scale: tf.Tensor, min_po2_exponent: Any,
                    max_po2_exponent: Any):
  """Clip power-of-two scales given minimum and maximum po2 exponenets."""

  min_po2 = None if min_po2_exponent is None else 2**min_po2_exponent
  max_po2 = None if max_po2_exponent is None else 2**max_po2_exponent
  scale = K.clip(scale, min_value=min_po2, max_value=max_po2)
  return scale


def _get_least_squares_scale(
  alpha: Any, x: tf.Tensor, q: tf.Tensor, scale_axis: Any = None,
  per_channel_scale: bool = True, elements_per_scale: Any = None,
  min_po2_exponent: Any = None, max_po2_exponent: Any = None):
  """Gets scaling factor for scaling the tensor per channel.

  It uses the least squares method to find the scaling factor.

  (https://en.wikipedia.org/wiki/Linear_least_squares)

  Arguments:
    alpha: A float or string. When it is string, it should be either "auto" or
      "auto_po2", and scale = sum(x * q, axis=all but last) / sum(q * q,
      axis=all but last)
    x: A tensor object. Its elements are in float.
    q: A tensor object. Its elements are in quantized format of x.
    scale_axis: int or List[int] representing which axis/axes to calculate
     scale from.
    per_channel_scale: A bool. Whether to perform per-channel scaling or not.
    elements_per_scale: if set to an int or List[int], we create multiple scales
      per axis across scale_axis, where 'elements_per_scale' represents the
      number of elements/values associated with every separate scale value.
    min_po2_exponent: if set while using "auto_po2", it represents the minimum
      allowed power of two exponent.
    max_po2_exponent: if set while using "auto_po2", it represents the maximum
      allowed power of two exponent.

  Returns:
    A scaling factor tensor or scalar for scaling tensor per channel.
  """

  if isinstance(alpha, six.string_types) and "auto" in alpha:
    assert alpha in ["auto", "auto_po2"]
    # in different tensorflow version (e.g., 2.4)
    # x.shape is a tuple which doesn't have as_list() method
    try:
      x_shape = x.shape.as_list()
    except AttributeError:
      x_shape = list(x.shape)

    len_axis = len(x_shape)
    if not per_channel_scale:
      qx = K.mean(x * q, keepdims=True)
      qq = K.mean(q * q, keepdims=True)
    else:
      if len_axis > 1:
        qx, qq = _get_scale_mean(scale_axis, x, q, elements_per_scale)
      else:
        # No summing (averaging) along the channel axis to get per-channel
        # scales.
        qx = x * q
        qq = q * q

    scale = qx / (qq + K.epsilon())
    if alpha == "auto_po2":
      scale = K.pow(2.0,
                    tf.math.round(K.log(scale + K.epsilon()) / np.log(2.0)))

      if min_po2_exponent is not None or max_po2_exponent is not None:
        scale = _clip_po2_scale(scale, min_po2_exponent, max_po2_exponent)

  elif alpha is None:
    scale = 1.0
  elif isinstance(alpha, np.ndarray):
    scale = alpha
  else:
    scale = float(alpha)
  return scale

def _get_scale(*args, **kwargs):
  """Old name for _get_least_squares_scale. Kept for backwards compatibility."""
  return _get_least_squares_scale(*args, **kwargs)

def smooth_sigmoid(x):
  """Implements a linear approximation of a sigmoid function."""

  # if we use 2.65 as the clipping point, MSE w.r.t. original sigmoid is
  # smaller than hard_simoid but the arithmetic for it is (x >> 3) +
  # (x >> 4) + 0.5, which is also not bad.

  return tf.keras.backend.clip(0.1875 * x + 0.5, 0.0, 1.0)


def hard_sigmoid(x):
  """Computes hard_sigmoid function that saturates between 0 and 1."""

  return tf.keras.backend.clip(0.5 * x + 0.5, 0.0, 1.0)


def binary_sigmoid(x):
  """Computes binary_sigmoid."""

  return _round_through(hard_sigmoid(x))


# we use a version of approximated sigmoid everywhere in this code.
# we can set it to hard_sigmoid(x) or smooth_sigmoid(x).

_default_sigmoid_type = "hard"
_sigmoid = None


def set_internal_sigmoid(mode):
  """Sets _sigmoid to either real, hard or smooth."""

  global _sigmoid

  if mode not in ["real", "hard", "smooth"]:
    raise ValueError("mode has to be 'real', 'hard' or 'smooth'.")

  if mode == "hard":
    _sigmoid = hard_sigmoid
  elif mode == "smooth":
    _sigmoid = smooth_sigmoid
  elif mode == "real":
    _sigmoid = tf.keras.backend.sigmoid


set_internal_sigmoid(_default_sigmoid_type)


def binary_tanh(x):
  """Computes binary_tanh function that outputs -1 and 1."""
  return 2.0 * binary_sigmoid(x) - 1.0


def hard_tanh(x):
  """Computes hard_tanh function that saturates between -1 and 1."""
  return 2.0 * hard_sigmoid(x) - 1.0


def smooth_tanh(x):
  """Computes smooth_tanh function that saturates between -1 and 1."""
  return 2.0 * smooth_sigmoid(x) - 1.0


def stochastic_round(x, precision=0.5):
  """Performs stochastic rounding to the first decimal point."""
  scale = 1.0 / precision
  scale_x = x * scale
  fraction = scale_x - tf.floor(scale_x)

  result = tf.where(fraction < tf.random.uniform(tf.shape(x)),
                    tf.math.floor(scale_x), tf.math.ceil(scale_x))
  return result / scale


def stochastic_round_po2(x):
  """Performs stochastic rounding for the power of two."""
  # TODO(b/237832905): test stochastic_round_po2 and constraint.
  # because quantizer is applied after constraint.
  y = tf.abs(x)
  eps = tf.keras.backend.epsilon()
  log2 = tf.keras.backend.log(2.0)

  x_log2 = tf.round(tf.keras.backend.log(y + eps) / log2)
  po2 = tf.cast(pow(2.0, tf.cast(x_log2, dtype="float32")), dtype="float32")
  left_val = tf.where(po2 > y, x_log2 - 1, x_log2)
  right_val = tf.where(po2 > y, x_log2, x_log2 + 1)
  # sampling in [2**left_val, 2**right_val].
  minval = 2 ** left_val
  maxval = 2 ** right_val
  val = tf.random.uniform(tf.shape(y), minval=minval, maxval=maxval)
  # use y as a threshold to keep the probabliy [2**left_val, y, 2**right_val]
  # so that the mean value of the sample should be y
  x_po2 = tf.where(y < val, left_val, right_val)
  """
  x_log2 = stochastic_round(tf.keras.backend.log(y + eps) / log2)
  sign = tf.sign(x)
  po2 = (
      tf.sign(x) *
      tf.cast(pow(2.0, tf.cast(x_log2, dtype="float32")), dtype="float32")
  )
  """
  return x_po2


def _round_through(x, use_stochastic_rounding=False, precision=0.5):
  """Rounds x but using straight through estimator.

  We use the trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182).

  Straight through estimator is a biased estimator for the rounding
  operation defined by Hinton"s Coursera Lecture 9c where dL/dx is made
  equal to dL/dy for y = f(x) during gradient computation, where f(x) is
  a non-derivable function. In that case, we assume df/dx = 1 in:

  dL   dL df   dL
  -- = -- -- = --
  dx   df dx   dy

  (https://www.youtube.com/watch?v=LN0xtUuJsEI&list=PLoRl3Ht4JOcdU872GhiYWf6jwrk_SNhz9&index=41)

  Arguments:
    x: tensor to perform round operation with straight through gradient.
    use_stochastic_rounding: if true, we perform stochastic rounding.
    precision: by default we will use 0.5 as precision, but that can overriden
      by the user.

  Returns:
    Rounded tensor.
  """
  if use_stochastic_rounding:
    output = tf_utils.smart_cond(
        K.learning_phase(),
        lambda: x + tf.stop_gradient(-x + stochastic_round(x, precision)),
        lambda: x + tf.stop_gradient(-x + tf.round(x)))
  else:
    output = x + tf.stop_gradient(-x + tf.round(x))
  return output


def _sign_through(x):
  """Computes the sign operation using the straight through estimator."""

  # tf.sign generates -1, 0 or +1, so it should not be used when we attempt
  # to generate -1 and +1.

  k_sign = tf.sign(x)

  return x + tf.stop_gradient(-x + k_sign)


def _ceil_through(x):
  """Computes the ceiling operation using straight through estimator."""

  return x + tf.stop_gradient(-x + tf.ceil(x))


def _floor_through(x):
  """Computes the floor operation using straight through estimator."""

  return x + tf.stop_gradient(-x + tf.floor(x))

#
# Activation functions for quantized networks.
#
# Please note some of these functions can be used as well
# as quantizer functions for weights of dense and convolutional
# layers.
#


@quantizer_registry.register_quantizer
class quantized_linear(base_quantizer.BaseQuantizer):
  """Linear quantization with fixed number of bits.

  This quantizer maps inputs to the nearest value of a fixed number of
  outputs that are evenly spaced, with possible scaling and stochastic
  rounding. This is an updated version of the legacy quantized_bits.

  The core computation is:
    1. Divide the tensor by a quantization scale
    2. Clip the tensor to a specified range
    3. Round to the nearest integer
    4. Multiply the rounded result by the quantization scale

  This clip range is determined by
    - The number of bits we have to represent the number
    - Whether we want to have a symmetric range or not
    - Whether we want to keep negative numbers or not

    The quantization scale is defined by either the quantizer parameters or the
    data passed to the __call__ method. See documentation for the `alpha`
    parameter to find out more.

    For backprop purposes, the quantizer uses the straight-through estimator
    for the rounding step (https://arxiv.org/pdf/1903.05662.pdf). Thus the
    gradient of the __call__ method is 1 on the interval
    [quantization_scale * clip_min, quantization_scale * clip_max] and 0
    elsewhere.

  The quantizer also supports a number of other optional features:
  - Stochastic rounding (see the `stochastic_rounding` parameter)
  - Quantization noise (see the `qnoise_factor` parameter)

  Notes on the various "scales" in quantized_linear:

      - The quantization scale is the scale used in the core computation (see
        above). You can access it via the `quantization_scale` attribute.
      - The data type scale is the scale is determined by the type of data
        stored on hardware on a small device running a true quantized model.
        It is the quantization scale needed to represent `bits` bits, `integer`
        of which are integer bits, and one bit is reserved for the sign if
        `keep_negative` is True. It can be calculated as
        2 ** (integer - bits + keep_negative). You can access it via the
        `data_type_scale` attribute.
      - The `scale` attribute stores the quotient of the quantization scale and
        the data type scale. This is also the scale that can be directly
        specified by the user, via the `alpha` parameter.

    These three quantities are related by the equation
    scale = quantization_scale / data_type_scale.

    See the diagram below of scale usage in a quantized conv layer.

    +------------------------------------------------------------------------+
    |     data_type_scale        --------------->     stored_weights         |
    | (determines decimal point)                            |                |
    |                                                       V                |
    |                                                    conv op             |
    |                                                       |                |
    |                                                       V                |
    |                                                  accumulator           |
    |                                                       |                |
    |  determines quantization                              V                |
    |    range and precision     --------------->   quantization_scale       |
    |       (per channel)                                   |                |
    |                                                       V                |
    |                                                   activation           |
    +------------------------------------------------------------------------+

      # TODO: The only fundamentally necessary scale is the quantization scale.
      # We should consider removing the data type scale and scale attributes,
      # but know that this will require rewriting much of how qtools and HLS4ML
      # use these scale attributes.

    Note on binary quantization (bits=1):
      The core computation is modified here when `keep_negative` is True to
      perform a scaled sign function. This is needed because the core
      computation as defined above requires that 0 be mapped to 0, which does
      not allow us to keep both positive and negative outputs for binary
      quantization. Special shifting operations are used to achieve this.

    Example usage:

  # 8-bit quantization with 3 integer bits
  >>> q = quantized_linear(8, 3)
  >>> x = tf.constant([0.0, 0.5, 1.0, 1.5, 2.0])
  >>> q(x).numpy()
  array([0., 0., 1., 2., 2.], dtype=float32)

  # 2-bit quantization with "auto" and tensor alphas
  >>> q_auto = quantized_linear(2, alpha="auto")
  >>> x = tf.constant([0.0, 0.5, 1.0, 1.5, 2.0])
  >>> q_auto(x).numpy()
  array([0., 0., 0., 2., 2.], dtype=float32)
  >>> q_auto.scale.numpy()
  array([4.], dtype=float32)
  >>> q_auto.quantization_scale.numpy()
  array([2.], dtype=float32)
  >>> q_fixed = quantized_linear(2, alpha=q_auto.scale)
  >>> q_fixed(x)
  array([0., 0., 0., 2., 2.], dtype=float32)

    Args:
      bits (int): Number of bits to represent the number. Defaults to 8.
      integer (int): Number of bits to the left of the decimal point, used for
        data_type_scale. Defaults to 0.
      symmetric (bool): If true, we will have the same number of values
        for positive and negative numbers. Defaults to True.
      alpha (str, Tensor, None): Instructions for determining the quantization
        scale. Defaults to None.
        - If None: the quantization scale is the data type scale, determined
          by `integer`, `bits`, and `keep_negative`.
        - If "auto", the quantization scale is calculated as the minimum
          floating point scale per-channel that does not clip the max of x.
        - If "auto_po2", the quantization scale is chosen as the
          power of two per-channel that minimizes squared error between the
          quantized x and the original x.
        - If Tensor: The quantization scale is the Tensor passed in
          multiplied by the data type scale.
      keep_negative (bool): If false, we clip negative numbers. Defaults to
        True.
      use_stochastic_rounding (bool): If true, we perform stochastic rounding
        (https://arxiv.org/pdf/1502.02551.pdf).
      scale_axis (int, None): Which axis to calculate scale from. If None, we
        perform per-channel scaling based off of the image data format. Note
        that each entry of a rank-1 tensor is considered its own channel by
        default. See `_get_scaling_axis` for more details. Defaults to None.
      qnoise_factor (float): A scalar from 0 to 1 that represents the level of
        quantization noise to add. This controls the amount of the
        quantization noise to add to the outputs by changing the weighted
        sum of (1 - qnoise_factor) * unquantized_x + qnoise_factor *
        quantized_x. Defaults to 1.0, which means that the result is fully
        quantized.
      use_variables (bool): If true, we use tf.Variables to store certain
        parameters. See the BaseQuantizer implementation for more details.
        Defaults to False. If set to True, be sure to use the special attribute
        update methods detailed in the BaseQuantizer.
      var_name (str or None): A variable name shared between the tf.Variables
        created in on initialization, if use_variables is true. If None, the
        variable names are generated automatically based on the parameter names
        along with a uid. Defaults to None.

  Returns:
    function: Function that computes linear quantization.

  Raises:
    ValueError:
      - If `bits` is not positive, or is too small to represent `integer`.
      - If `integer` is negative.
      - If `alpha` is a string but not one of ("auto", "auto_po2").
  """

  # string options for alpha parameter
  ALPHA_STRING_OPTIONS = ("auto", "auto_po2")

  def __init__(
      self,
      bits=8,
      integer=0,
      symmetric=1,
      keep_negative=True,
      alpha=None,
      use_stochastic_rounding=False,
      scale_axis=None,
      qnoise_factor=1.0,
      var_name=None,
      use_variables=False,
  ):
    super().__init__()

    self.var_name = var_name

    # Error checking
    self._check_bits(bits)
    self._check_alpha(alpha)

    # Set non-modifyable attributes
    self._bits = bits
    self._integer = integer
    self._keep_negative = keep_negative
    self._use_stochastic_rounding = use_stochastic_rounding
    self._scale_axis = scale_axis
    self._use_variables = use_variables

    # Set modifyable attributes
    self.alpha = alpha
    self.qnoise_factor = qnoise_factor
    self.symmetric = symmetric

    # Set default quantization scale
    self.quantization_scale = self.default_quantization_scale

  def _check_bits(self, bits):
    """Error checking for bits parameter"""
    err_msg = f"Bit count {bits} must be positive"
    if bits <= 0:
      raise ValueError(err_msg)

  def _check_alpha(self, alpha):
    """Error checking for alpha parameter"""

    if isinstance(alpha, six.string_types):
      # Check the quantizer has been given a valid alpha string
      if not alpha in self.ALPHA_STRING_OPTIONS:
        raise ValueError(
            f"Invalid alpha '{alpha}' for auto alpha computation. "
            f"Must be one of {self.ALPHA_STRING_OPTIONS}")
    elif alpha is not None: # alpha is a tensor
      try:
        # any allowable array type can be cast as a numpy array
        np.array(alpha)
      except TypeError:
        raise TypeError(
            f"alpha must be, a string, an array, or None, not {type(alpha)}")

  @property
  def bits(self):
    return self._bits

  @property
  def integer(self):
    return self._integer

  @property
  def keep_negative(self):
    return self._keep_negative

  @property
  def use_stochastic_rounding(self):
    return self._use_stochastic_rounding

  @property
  def scale_axis(self):
    return self._scale_axis

  @property
  def use_variables(self):
    return self._use_variables

  @property
  def scale(self):
    return self.quantization_scale / self.data_type_scale

  @property
  def data_type_scale(self):
    """Quantization scale for the data type"""
    # integer is sometimes cast as int32, so cast to float32 to avoid errors
    integer = tf.cast(self.integer, tf.float32)
    return K.pow(2.0, integer - self.bits + self.keep_negative)

  @property
  def auto_alpha(self):
    """Returns true if using a data-dependent alpha"""

    return isinstance(self.alpha, six.string_types)

  @property
  def use_sign_function(self):
    """Return true if using sign function for quantization"""

    return (self.bits == 1.0) and self.keep_negative

  @property
  def default_quantization_scale(self):
    """Calculate and set quantization_scale default"""

    # Set default quantization scale
    quantization_scale = self.data_type_scale

    # Quantization scale given by alpha
    if self.alpha is not None and not self.auto_alpha:
      quantization_scale = self.alpha * self.data_type_scale

    return quantization_scale

  def get_clip_bounds(self):
    """Get bounds of clip range"""

    if self.use_sign_function:
      clip_min = K.cast_to_floatx(-0.5)
      clip_max = K.cast_to_floatx(0.5)
    else:
      unsigned_bits_po2 = K.pow(2.0, self.bits - self.keep_negative)
      # if symmetric, clip_min is negative of clip_max. Otherwise clip_min is
      # lowered by 1, giving us one more representable number
      clip_min = self.keep_negative * (-unsigned_bits_po2 + self.symmetric)
      clip_max = unsigned_bits_po2 - K.cast_to_floatx(1.0)

    return clip_min, clip_max

  def __call__(self, x):
    """Core quantization function"""

    # Build if not already built
    self._build()

    # Data type conversion
    x = K.cast_to_floatx(x)
    shape = x.shape

    if self.auto_alpha:
      # get data-dependent quantization scale
      quantization_scale = self._get_auto_quantization_scale(x)
    else:
      # quantization scale determined by quantizer params, not data
      # see default_quantization_scale property for more info
      quantization_scale = self.quantization_scale

    scaled_xq = self._scale_clip_and_round(x, quantization_scale)
    xq = scaled_xq * quantization_scale

    res = x + self.qnoise_factor * (xq - x)
    res.set_shape(shape)

    return res

  def _scale_clip_and_round(self, x, quantization_scale):
    """Scale, clip, and round x to an integer value in a limited range
    Note that the internal shift is needed for 1-bit quantization to ensure
    that a sign function is used. Otherise, the binary quantizer would have
    three output values"""

    # special shifting needed to compute a sign function.
    shift = self.use_sign_function * 0.5

    clip_min, clip_max = self.get_clip_bounds()

    scaled_x = x / quantization_scale
    clipped_scaled_x = K.clip(scaled_x, clip_min, clip_max)
    # Round through to nearest integer, using straight-through estimator
    # for gradient computations.
    scaled_xq = _round_through(
      clipped_scaled_x - shift,
      use_stochastic_rounding=self.use_stochastic_rounding,
      precision=1.0, # using 1.0 precision so that we round to a nearby integer
    )

    return scaled_xq + shift

  def _get_auto_quantization_scale(self, x):
    """Get quantization_scale, either from self or from input x"""

    # Get the minimum floating point scale that does not clip the max of x
    # This is the quantization scale for alpha="auto"
    quantization_scale = self._get_quantization_scale_from_max_data(x)

    if self.alpha == "auto_po2":
      quantization_scale = self._po2_autoscale(x, quantization_scale)

    # update quantization_scale variable
    # stop_gradient on quantization_scale to ignore dependence on x
    self.quantization_scale = tf.stop_gradient(quantization_scale)

    # very important that return value is a tf.Variable with shape None
    return self.quantization_scale

  def _get_quantization_scale_from_max_data(self, x):
    """Get the minimum floating point scale that does not clip the max
    of x"""

    axis = _get_scaling_axis(self.scale_axis, tf.rank(x))

    clip_min, clip_max = self.get_clip_bounds()
    clip_range = clip_max - clip_min

    # get quantization scale- depends on whether we are keeping negative
    # divide by clip range to ensure that we clip right at the max of x
    if self.keep_negative:
      data_max = K.max(tf.math.abs(x), axis=axis, keepdims=True)
      quantization_scale = (data_max * 2) / clip_range
    else:
      data_max = K.max(x, axis=axis, keepdims=True)
      quantization_scale = data_max / clip_range

    return tf.math.maximum(quantization_scale, K.epsilon())

  def _po2_autoscale(self, x, quantization_scale):
    """Get an approximation of the "best" po2 scale using least squares"""

    # set alpha scale to a near power of two
    quantization_scale = K.pow(
        2.0,
        tf.math.round(K.log(quantization_scale + K.epsilon()) / K.log(2.0)))

    def loop_body(_, quantization_scale):
      """Loop body for least squares autoscaling"""

      scaled_xq = self._scale_clip_and_round(x, quantization_scale)
      new_quantization_scale = _get_least_squares_scale(
          alpha="auto_po2",
          x=x,
          q=scaled_xq,
          scale_axis=self.scale_axis,
      )
      return quantization_scale, new_quantization_scale

    def loop_cond(last_quantization_scale, quantization_scale):
      """Loop condition for least squares autoscaling- stop when the
      scale converges"""

      tensors_not_equal = tf.math.reduce_any(
          tf.not_equal(last_quantization_scale, quantization_scale))
      return tensors_not_equal

    # Need a tensor of the same shape as quantization_scale that
    # does not equal quantization_scale
    dummy_quantization_scale = -tf.ones_like(quantization_scale)

    # For 1-bit quantization, po2 autoscale loop is guaranteed to converge
    # after 1 iteration
    max_iterations = 1 if self.use_sign_function else 5

    _, quantization_scale = tf.while_loop(
        loop_cond,
        loop_body,
        (dummy_quantization_scale, quantization_scale),
        maximum_iterations=max_iterations,
    )

    return quantization_scale

  def _build(self):
    """Build if not done so already"""

    if not self.built:
      self.build(var_name=self.var_name, use_variables=self.use_variables)

  def max(self):
    """Get maximum value that quantized_linear class can represent."""
    _, clip_max = self.get_clip_bounds()
    return clip_max * self.quantization_scale

  def min(self):
    """Get minimum value that quantized_linear class can represent."""
    clip_min, _ = self.get_clip_bounds()
    return clip_min * self.quantization_scale

  def range(self):
    """Returns a list of all values that quantized_linear can represent
    }."""

    if self.use_sign_function:
      return K.cast_to_floatx([self.max(), self.min()])
    else:
      clip_min, clip_max = self.get_clip_bounds()
      clip_max = tf.cast(clip_max, tf.int32)
      clip_min = tf.cast(clip_min, tf.int32)
      pos_array = K.cast_to_floatx(tf.range(clip_max + 1))
      neg_array = K.cast_to_floatx(tf.range(clip_min, 0))

      return self.quantization_scale * tf.concat([pos_array, neg_array], axis=0)

  def __str__(self):

    # Main parameters always printed in string
    flags = [
      str(int(self.bits)),
      str(int(self.integer)),
      str(int(self.symmetric))]
    # Optional parameters only printed if not default
    if not self.keep_negative:
      flags.append("keep_negative=False")
    if self.auto_alpha:
      alpha = "'" + self.alpha + "'"
      flags.append("alpha=" + alpha)
    elif self.alpha is not None:
      alpha = np.array(alpha)
      flags.append("alpha=" + str(alpha))
    if self.use_stochastic_rounding:
      flags.append("use_stochastic_rounding=" +
                   str(int(self.use_stochastic_rounding)))
    return "quantized_linear(" + ",".join(flags) + ")"

  def _set_trainable_parameter(self):
    if self.alpha is None:
      self.alpha = "auto_po2"
      self.symmetric = True

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def get_config(self):

    config = {
        "bits": self.bits,
        "integer": self.integer,
        "symmetric": self.symmetric,
        "alpha": self.alpha,
        "keep_negative": self.keep_negative,
        "use_stochastic_rounding": self.use_stochastic_rounding,
        "qnoise_factor": self.qnoise_factor,
    }
    return config


@quantizer_registry.register_quantizer
class quantized_bits(base_quantizer.BaseQuantizer):  # pylint: disable=invalid-name
  """Legacy quantizer: Quantizes the number to a number of bits.

  In general, we want to use a quantization function like:

  a = (pow(2,bits) - 1 - 0) / (max(x) - min(x))
  b = -min(x) * a

  in the equation:

  xq = a x + b

  This requires multiplication, which is undesirable. So, we
  enforce weights to be between -1 and 1 (max(x) = 1 and min(x) = -1),
  and separating the sign from the rest of the number as we make this function
  symmetric, thus resulting in the following approximation.

  1) max(x) = +1, min(x) = -1
  2) max(x) = -min(x)

  a = pow(2,bits-1)
  b = 0

  Finally, just remember that to represent the number with sign, the
  largest representation is -pow(2,bits) to pow(2, bits-1)

  Symmetric and keep_negative allow us to generate numbers that are symmetric
  (same number of negative and positive representations), and numbers that
  are positive.

  Note:
    the behavior of quantized_bits is different than Catapult HLS ac_fixed
    or Vivado HLS ap_fixed. For ac_fixed<word_length, integer_lenth, signed>,
    when signed = true, it is equavlent to
    quantized_bits(word_length, integer_length-1, keep_negative=True)

  Attributes:
    bits: number of bits to perform quantization.
    integer: number of bits to the left of the decimal point.
    symmetric: if true, we will have the same number of values for positive
      and negative numbers.
    alpha: a tensor or None, the scaling factor per channel.
      If None, the scaling factor is 1 for all channels.
    keep_negative: if true, we do not clip negative numbers.
    use_stochastic_rounding: if true, we perform stochastic rounding.
    scale_axis: int or List[int] which axis/axes to calculate scale from.
    qnoise_factor: float. a scalar from 0 to 1 that represents the level of
      quantization noise to add. This controls the amount of the quantization
      noise to add to the outputs by changing the weighted sum of
      (1 - qnoise_factor)*unquantized_x + qnoise_factor*quantized_x.
    var_name: String or None. A variable name shared between the tf.Variables
      created in the build function. If None, it is generated automatically.
    use_ste: Bool. Whether to use "straight-through estimator" (STE) method or
        not.
    use_variables: Bool. Whether to make the quantizer variables to be dynamic
      tf.Variables or not.
    elements_per_scale: if set to an int or List[int], we create multiple scales
      per axis across scale_axis, where 'elements_per_scale' represents the
      number of elements/values associated with every separate scale value.
      It is only supported when using "auto_po2".
    min_po2_exponent: if set while using "auto_po2", it represents the minimum
      allowed power of two exponent.
    max_po2_exponent: if set while using "auto_po2", it represents the maximum
      allowed power of two exponent.
    post_training_scale: if set, it represents the scale value to be used for
      quantization.

  Returns:
    Function that computes fixed-point quantization with bits.
  """

  def __init__(self,
               bits=8,
               integer=0,
               symmetric=0,
               keep_negative=True,
               alpha=None,
               use_stochastic_rounding=False,
               scale_axis=None,
               qnoise_factor=1.0,
               var_name=None,
               use_ste=True,
               use_variables=False,
               elements_per_scale=None,
               min_po2_exponent=None,
               max_po2_exponent=None,
               post_training_scale=None):
    super().__init__()

    self.bits = bits
    self.integer = integer
    self.symmetric = symmetric
    self.keep_negative = keep_negative
    self.alpha = alpha
    self.use_stochastic_rounding = use_stochastic_rounding
    self.post_training_scale = post_training_scale
    # "auto*" |-> symmetric
    if isinstance(self.alpha, six.string_types):
      self.freeze_scale = False
      self.symmetric = True
      if post_training_scale is not None:
        self.scale = np.array(post_training_scale)
        self.freeze_scale = True
    else:
      if post_training_scale is not None:
        raise ValueError(f"alpha={alpha} doesn't support post_training_scale: "
                         f"{post_training_scale}")
      self.scale = None
      # If alpha is not "auto*", then scale is fixed and not trainable.
      self.freeze_scale = True

    self.scale_axis = scale_axis
    self.qnoise_factor = qnoise_factor
    self.use_ste = use_ste
    self.var_name = var_name
    self.use_variables = use_variables
    self.elements_per_scale = elements_per_scale
    self.min_po2_exponent = min_po2_exponent
    self.max_po2_exponent = max_po2_exponent

  def __str__(self):
    # Convert Tensors to printable strings by converting to a numpy array and
    # then using regex to remove brackets when there is only one integer bit
    integer_bits = re.sub(
        r"\[(\d)\]", r"\g<1>",
        str(self.integer.numpy() if isinstance(self.integer, tf.Variable
                                              ) else self.integer))

    flags = [str(self.bits), integer_bits, str(int(self.symmetric))]
    if not self.keep_negative:
      flags.append("keep_negative=False")
    if self.alpha:
      alpha = str(self.alpha)
      if isinstance(self.alpha, six.string_types):
        alpha = "'" + alpha + "'"
      flags.append("alpha=" + alpha)
    if self.use_stochastic_rounding:
      flags.append("use_stochastic_rounding=" +
                   str(int(self.use_stochastic_rounding)))
    return "quantized_bits(" + ",".join(flags) + ")"

  def __call__(self, x):
    """Computes fixedpoint quantization of x."""
    if not self.built:
      self.build(var_name=self.var_name, use_variables=self.use_variables)

    x = K.cast_to_floatx(x)

    # quantized_bits with "1" bit becomes a binary implementation.
    unsigned_bits = self.bits - self.keep_negative
    # In pow function, use float datatype instead of integer, so that
    # K.pow() results will use float32 instead of int32 as the default datatype.
    # float32 has a much larger value range (2^128) than int32 (2^32), this is
    # particularly important when quantizing very large values, and when integer
    # bits are set much larger than total bits.
    m = K.pow(2.0, K.cast_to_floatx(unsigned_bits))
    m_i = K.pow(2.0, K.cast_to_floatx(self.integer))

    # Verify that "elements_per_scale", "min_po2_exponent",
    # and "max_po2_exponent" are only set when alpha is "auto_po2"
    if self.alpha != "auto_po2":
      assert (
          self.elements_per_scale is None
      ), "elements_per_scale is only supported when using auto_po2"
      assert (
          self.min_po2_exponent is None
      ), "min_po2_exponent is only supported when using auto_po2"
      assert (
          self.max_po2_exponent is None
      ), "max_po2_exponent is only supported when using auto_po2"

    if self.alpha is None:
      scale = 1.0
    elif isinstance(self.alpha, six.string_types):
      # We only deal with the symmetric case right now.
      assert self.symmetric, "Only symmetric quantizers are implemented"
      len_axis = len(x.shape)
      if len_axis > 1:
        axis = _get_scaling_axis(self.scale_axis, len_axis)
      else:
        axis = [0]

      x = x / m_i

      # Using 2's complement, we can represent 2**(bits-1)-1 positive values
      # If we wish to maintain symmetry, we can double 2**(bits-1)-1 to get
      # the total number of possible values we can represent.
      # If symmetry is not enforced, then we can represent (2**bits)-1 values
      # using 2's complement.
      levels = (2**(self.bits-1)-1) * 2 if self.symmetric else (2**self.bits)-1

      if self.freeze_scale:
        # Scale is fixed value. In this case, scale is extracted from the
        # post-training quantizater scale. In order to retrain models with
        # this scale value, we need to divide it by m to make it in the same
        # value scale as x.
        scale = self.scale / m
      else:
        # Calculate the scale.
        scale = (K.max(abs(x), axis=axis, keepdims=True) * 2) / levels

        # If alpha is "auto_po2", then get the "best" po2 scale
        if "po2" in self.alpha:
          scale = K.pow(2.0,
                        tf.math.round(K.log(scale + K.epsilon()) / np.log(2.0)))
          for idx in range(5):
            v = tf.floor(tf.abs(x) / scale + 0.5)
            mask = v < levels / 2
            z = tf.sign(x) * tf.where(mask, v, tf.ones_like(v) * levels / 2)
            scale = _get_least_squares_scale(
                alpha="auto_po2", x=x, q=z,
                scale_axis=self.scale_axis,
                elements_per_scale=self.elements_per_scale,
                min_po2_exponent=self.min_po2_exponent,
                max_po2_exponent=self.max_po2_exponent)

        elif self.alpha != "auto":
          # If alpha is "auto", then directly uuse the "best"
          # floating point scale.
          raise ValueError(f"Invalid alpha '{self.alpha}'")

      # Even for trainable scale, we still need to quantize x with the best
      # scale. This extra step is needed to ensure that with the same input
      # and scale, the quantized output is identical between training and
      # inference.
      v = tf.floor(tf.abs(x) / scale + 0.5)
      mask = v < levels / 2
      z = tf.sign(x) * tf.where(mask, v, tf.ones_like(v) * levels / 2)

      # z is an integer number, so we must make the scale * m and z / m
      scale = scale * m

      # we will not use "z" right now because of stochastic_rounding
      # this is still under test.

      # if "new" in self.alpha:
      #  z = z / m
      #  self.scale = scale
      #  return x + tf.stop_gradient(-x + scale * z)
      x = m_i * x
      xq = m_i * z / m
      if not self.freeze_scale:
        self.scale = scale
      xq = scale * xq

      if self.use_ste:
        return x + tf.stop_gradient(self.qnoise_factor * (-x + xq))
      else:
        return (1 - self.qnoise_factor) * x + tf.stop_gradient(
            self.qnoise_factor * xq)

    else:
      scale = self.alpha

    # quantized_bits with "1" bit becomes a binary implementation.
    if unsigned_bits > 0:
      p = x * m / m_i
      xq = m_i * tf.keras.backend.clip(
          _round_through(p, self.use_stochastic_rounding, precision=1.0),
          self.keep_negative  * (-m + self.symmetric), m - 1) / m
    else:
      xq = tf.sign(x)
      xq += (1.0 - tf.abs(xq))
      if not self.keep_negative:
        xq = (xq + 1.0) / 2.0

    self.scale = scale
    xq = scale * xq

    if self.use_ste:
      return x + tf.stop_gradient(self.qnoise_factor * (-x + xq))
    else:
      return (1 - self.qnoise_factor) * x + tf.stop_gradient(
          self.qnoise_factor * xq)

  def _set_trainable_parameter(self):
    if self.alpha is None:
      self.alpha = "auto_po2"
      self.freeze_scale = False
      self.symmetric = True

  def max(self):
    """Get maximum value that quantized_bits class can represent."""
    unsigned_bits = self.bits - self.keep_negative
    if unsigned_bits > 0:
      return max(
          1.0,
          np.array(
              K.pow(2., K.cast(self.integer, dtype="float32")),
              dtype="float32"))
    else:
      return 1.0

  def min(self):
    """Get minimum value that quantized_bits class can represent."""
    if not self.keep_negative:
      return 0.0
    unsigned_bits = self.bits - self.keep_negative
    if unsigned_bits > 0:
      return -max(
          1.0,
          np.array(
              K.pow(2, K.cast(self.integer, dtype="float32")), dtype="float32"))
    else:
      return -1.0

  def range(self):
    """Returns a list of all values that quantized_bits can represent
    ordered by their binary representation ascending."""
    assert self.symmetric == 0
    assert self.keep_negative
    assert self.alpha is None or self.alpha == 1.0

    x = np.asarray(range(2**self.bits), dtype=np.float32)
    p_and_n = np.where(x >= 2**(self.bits - 1),
                       (x - 2**(self.bits - 1)) - 2**(self.bits - 1), x)
    return p_and_n * np.array(
        K.pow(2.0, -self.bits + K.cast(self.integer, dtype="float32") + 1),
        dtype="float32")

  @classmethod
  def from_config(cls, config):
    # Convert JSON-serializable lists back to NumPy arrays.
    if config.get("post_training_scale") is not None:
      config["post_training_scale"] = np.array(config["post_training_scale"])

    return cls(**config)

  def get_config(self):
    config = {
        "bits":
            self.bits,
        "integer":
            self.integer.numpy()
            if isinstance(self.integer, tf.Variable) else self.integer,
        "symmetric":
            self.symmetric,
        "alpha":
            self.alpha,
        "keep_negative":
            self.keep_negative,
        "use_stochastic_rounding":
            self.use_stochastic_rounding,
        "qnoise_factor":
            self.qnoise_factor.numpy() if isinstance(
                self.qnoise_factor, tf.Variable) else self.qnoise_factor,
        "post_training_scale":
            # Since NumPy arrays are not directly JSON-serializable,
            # we convert them to lists.
            (self.post_training_scale.tolist() if self.post_training_scale is
             not None else None)
    }
    return config


@quantizer_registry.register_quantizer
class bernoulli(base_quantizer.BaseQuantizer):  # pylint: disable=invalid-name
  """Computes a Bernoulli sample with probability sigmoid(x).

  This computation uses ST approximation.

  To do that, we compute sigmoid(x) and a random sample z ~ U[0,1]. As
  p in [0,1] and z in [0,1], p - z in [-1,1]. However, -1 will
  never appear because to get -1 we would need sigmoid(-inf) - z == 1.
  As a result, the range will be in practical terms [0,1].

  The noise introduced by z can be seen as a regularizer to the weights W of
  y = Wx as y = Wx + Wz for some noise z with mean mu(z) and var(z). As a
  result, W**2 var(z) to the variance of y, which has the same effect as a
  regularizer on L2 with lambda = var(z), as presented in Hinton"s Coursera
  Lecture 9c.

  Remember that E[dL/dy] = E[dL/dx] once we add stochastic sampling.

  Attributes:
    alpha: allows one to specify multiplicative factor for number generation
      of "auto" or "auto_po2".
    temperature: amplifier factor for sigmoid function, making stochastic
      less stochastic as it moves away from 0.
    use_real_sigmoid: use real sigmoid for probability.

  Returns:
    Computation of round with stochastic sampling with straight through
    gradient.
  """

  def __init__(self, alpha=None, temperature=6.0, use_real_sigmoid=True):
    super().__init__()
    self.alpha = alpha
    self.bits = 1
    self.temperature = temperature
    self.use_real_sigmoid = use_real_sigmoid
    self.default_alpha = 1.0
    self.scale = None

  def __str__(self):
    flags = []
    if self.alpha is not None:
      alpha = str(self.alpha)
      if isinstance(self.alpha, six.string_types):
        alpha = "'" + alpha + "'"
      flags.append("alpha=" + alpha)
    if self.temperature != 6.0:
      flags.append("temperature=" + str(self.temperature))
    if not self.use_real_sigmoid:
      flags.append("use_real_sigmoid=" + str(int(self.use_real_sigmoid)))
    return "bernoulli(" + ",".join(flags) + ")"

  def __call__(self, x):
    if isinstance(self.alpha, six.string_types):
      assert self.alpha in ["auto", "auto_po2"]

    if isinstance(self.alpha, six.string_types):
      len_axis = len(x.shape)

      if len_axis > 1:
        if K.image_data_format() == "channels_last":
          axis = list(range(len_axis - 1))
        else:
          axis = list(range(1, len_axis))
      else:
        axis = [0]

      std = K.std(x, axis=axis, keepdims=True) + K.epsilon()
    else:
      std = 1.0

    if self.use_real_sigmoid:
      p = tf.keras.backend.sigmoid(self.temperature * x / std)
    else:
      p = _sigmoid(self.temperature * x/std)
    r = tf.random.uniform(tf.shape(x))
    q = tf.sign(p - r)
    q += (1.0 - tf.abs(q))
    q = (q + 1.0) / 2.0

    q_non_stochastic = tf.sign(x)
    q_non_stochastic += (1.0 - tf.abs(q_non_stochastic))
    q_non_stochastic = (q_non_stochastic + 1.0) / 2.0

    # if we use non stochastic binary to compute alpha,
    # this function seems to behave better
    scale = _get_least_squares_scale(self.alpha, x, q_non_stochastic)
    self.scale = scale
    return x + tf.stop_gradient(-x + scale * q)

  def _set_trainable_parameter(self):
    if self.alpha is None:
      self.alpha = "auto_po2"

  def max(self):
    """Get the maximum value bernoulli class can represent."""
    if self.alpha is None or isinstance(self.alpha, six.string_types):
      return 1.0
    else:
      return max(1.0, self.alpha)

  def min(self):
    """Get the minimum value bernoulli class can represent."""
    return 0.0

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def get_config(self):
    config = {"alpha": self.alpha}
    return config


@quantizer_registry.register_quantizer
class ternary(base_quantizer.BaseQuantizer):  # pylint: disable=invalid-name
  """Computes an activation function returning -alpha, 0 or +alpha.

  Right now we assume two type of behavior. For parameters, we should
  have alpha, threshold and stochastic rounding on. For activations,
  alpha and threshold should be floating point numbers, and stochastic
  rounding should be off.

  Attributes:
    x: tensor to perform sign opertion with stochastic sampling.
    bits: number of bits to perform quantization.
    alpha: ternary is -alpha or +alpha. Alpha can be "auto" or "auto_po2".
    threshold: threshold to apply "dropout" or dead band (0 value). If "auto"
      is specified, we will compute it per output layer.
    use_stochastic_rounding: if true, we perform stochastic rounding.

  Returns:
    Computation of sign within the threshold.
  """

  def __init__(self, alpha=None, threshold=None, use_stochastic_rounding=False,
               number_of_unrolls=5):
    super().__init__()
    self.bits = 2
    self.alpha = alpha
    self.threshold = threshold
    self.use_stochastic_rounding = use_stochastic_rounding
    self.default_alpha = 1.0
    self.default_threshold = 0.33
    self.number_of_unrolls = number_of_unrolls
    self.scale = None

  def __str__(self):
    flags = []
    if self.alpha is not None:
      alpha = str(self.alpha)
      if isinstance(self.alpha, six.string_types):
        alpha = "'" + alpha + "'"
      flags.append("alpha=" + alpha)
    if self.threshold is not None:
      flags.append("threshold=" + str(self.threshold))
    if self.use_stochastic_rounding:
      flags.append(
          "use_stochastic_rounding=" + str(int(self.use_stochastic_rounding)))
    if self.number_of_unrolls != 5:
      flags.append(
          "number_of_unrolls=" + str(int(self.number_of_unrolls)))
    return "ternary(" + ",".join(flags) + ")"

  def __call__(self, x):
    if isinstance(self.alpha, six.string_types):
      # parameters
      assert self.alpha in ["auto", "auto_po2"]
      assert self.threshold is None
    else:
      # activations
      assert not self.use_stochastic_rounding
      assert not isinstance(self.threshold, six.string_types)

    if self.alpha is None or isinstance(self.alpha, six.string_types):
      scale = 1.0
    elif isinstance(self.alpha, np.ndarray):
      scale = self.alpha
    else:
      scale = float(self.alpha)

    # This is an approximiation from https://arxiv.org/abs/1605.04711
    # We consider channels_last only for now.
    if isinstance(self.alpha, six.string_types):
      # It is for parameters
      # first, compute which asix corresponds to the channels.
      # TODO(b/237833510): support channels_first
      try:
        len_axis = len(x.shape.as_list())
      except AttributeError:
        len_axis = len(list(x.shape))

      if len_axis == 1:
        axis = None
      elif K.image_data_format() == "channels_last":
        axis = list(range(len_axis - 1))
      else:
        axis = list(range(1, len_axis))

      # This approximation is exact if x ~ U[-m, m]. For x ~ N(0, m)
      # we need to iterate a few times before we can coverge
      m = K.max(tf.abs(x), axis=axis, keepdims=True)
      scale = 2 * m / 3.0
      if "po2" in self.alpha:
        scale = K.pow(2.0,
                      tf.math.round(K.log(scale + K.epsilon()) / np.log(2.0)))

      for _ in range(self.number_of_unrolls):
        thres = scale / 2.0
        # once we scale the number precision == 0.33 works
        # well for Uniform and Normal distribution of input
        v = scale * _round_through(
            x / scale,
            use_stochastic_rounding=self.use_stochastic_rounding,
            precision=1. / 3.)
        q = K.cast(tf.abs(v) >= thres, K.floatx()) * tf.sign(x)
        scale = _get_least_squares_scale(self.alpha, x, q)
    else:
      if self.threshold is None:
        thres = self.default_threshold
      else:
        thres = self.threshold
      q = K.cast(tf.abs(x) >= thres, K.floatx()) * tf.sign(x)

    # ternary ranges from -1 to +1, so we use tanh(x) to be a differentiable
    # version of that.
    if self.alpha is None:
      x = K.tanh(x)

    self.scale = scale
    return x + tf.stop_gradient(-x + scale * q)

  def _set_trainable_parameter(self):
    if self.alpha is None:
      self.alpha = "auto_po2"

  def max(self):
    """Get the maximum value that ternary can respresent."""
    if self.alpha is None or isinstance(self.alpha, six.string_types):
      return 1.0
    else:
      return max(1.0, self.alpha)

  def min(self):
    """Get the minimum value that ternary can respresent."""
    if self.alpha is None or isinstance(self.alpha, six.string_types):
      return -1.0
    else:
      return -max(1.0, self.alpha)

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def get_config(self):
    config = {
        "alpha": self.alpha,
        "threshold": self.threshold,
        "use_stochastic_rounding": self.use_stochastic_rounding,
        "number_of_unrolls": self.number_of_unrolls
    }
    return config


@quantizer_registry.register_quantizer
class stochastic_ternary(ternary):  # pylint: disable=invalid-name
  """Computes a stochastic activation function returning -alpha, 0 or +alpha.

  Computes straight-through approximation using random sampling to make
  E[dL/dy] = E[dL/dx], and computing the sign function. See explanation above.

  Attributes:
    x: tensor to perform sign opertion with stochastic sampling.
    bits: number of bits to perform quantization.
    alpha: ternary is -alpha or +alpha, or "auto" or "auto_po2".
    threshold: (1-threshold) specifies the spread of the +1 and -1 values.
    temperature: amplifier factor for sigmoid function, making stochastic less
      stochastic as it moves away from 0.
    use_real_sigmoid: use real sigmoid for probability.
    number_of_unrolls: number of times we iterate between scale and threshold.

  Returns:
    Computation of sign with stochastic sampling with straight through gradient.
  """

  def __init__(
      self,
      alpha=None,
      threshold=None,
      temperature=8.0,
      use_real_sigmoid=True,
      number_of_unrolls=5,
  ):
    super().__init__(
        alpha=alpha, threshold=threshold, number_of_unrolls=number_of_unrolls
    )

    self.bits = 2
    self.alpha = alpha
    self.threshold = threshold
    assert threshold != 1.0
    self.default_alpha = 1.0
    self.default_threshold = 0.33
    self.temperature = temperature
    self.use_real_sigmoid = use_real_sigmoid
    self.number_of_unrolls = number_of_unrolls
    self.scale = None

  def __str__(self):
    flags = []
    if self.alpha is not None:
      alpha = str(self.alpha)
      if isinstance(self.alpha, six.string_types):
        alpha = "'" + alpha + "'"
      flags.append("alpha=" + alpha)
    if self.threshold is not None:
      flags.append("threshold=" + str(self.threshold))
    if self.temperature != 8.0:
      flags.append("temperature=" + str(self.temperature))
    if not self.use_real_sigmoid:
      flags.append("use_real_sigmoid=0")
    if self.number_of_unrolls != 5:
      flags.append("number_of_unrolls=" + str(self.number_of_unrolls))
    return "stochastic_ternary(" + ",".join(flags) + ")"

  def __call__(self, x):
    def stochastic_output():
      # right now we only accept alpha = "auto" or "auto_po2"

      assert isinstance(self.alpha, six.string_types)
      assert self.alpha in ["auto", "auto_po2"]

      if self.alpha is None:
        scale = self.default_alpha
      elif isinstance(self.alpha, six.string_types):
        scale = 1.0
        assert self.alpha in ["auto", "auto_po2"]
      else:
        assert self.alpha >= 0.0
        scale = float(self.alpha)

      len_axis = len(x.shape)
      if len_axis > 1:
        if K.image_data_format() == "channels_last":
          axis = list(range(len_axis - 1))
        else:
          axis = list(range(1, len_axis))
      else:
        axis = [0]

      x_std = K.std(x, axis=axis, keepdims=True)

      m = K.max(tf.abs(x), axis=axis, keepdims=True)
      scale = 2.0 * m / 3.0
      if self.alpha == "auto_po2":
        scale = K.pow(
            2.0, tf.math.round(K.log(scale + K.epsilon()) / np.log(2.0))
        )
      for _ in range(self.number_of_unrolls):
        T = scale / 2.0
        q_ns = K.cast(tf.abs(x) >= T, K.floatx()) * K.sign(x)
        scale = _get_least_squares_scale(self.alpha, x, q_ns)

      x_norm = x / (x_std + K.epsilon())
      T = scale / (2.0 * (x_std + K.epsilon()))

      if self.use_real_sigmoid:
        p0 = tf.keras.backend.sigmoid(self.temperature * (x_norm - T))
        p1 = tf.keras.backend.sigmoid(self.temperature * (x_norm + T))
      else:
        p0 = _sigmoid(self.temperature * (x_norm - T))
        p1 = _sigmoid(self.temperature * (x_norm + T))
      r0 = tf.random.uniform(tf.shape(p0))
      r1 = tf.random.uniform(tf.shape(p1))
      q0 = tf.sign(p0 - r0)
      q0 += 1.0 - tf.abs(q0)
      q1 = tf.sign(p1 - r1)
      q1 += 1.0 - tf.abs(q1)

      q = (q0 + q1) / 2.0
      self.scale = scale
      return x + tf.stop_gradient(-x + scale * q)

    output = tf_utils.smart_cond(
        K.learning_phase(), stochastic_output, lambda: ternary.__call__(self, x)
    )
    return output

  def _set_trainable_parameter(self):
    if self.alpha is None:
      self.alpha = "auto_po2"

  def max(self):
    """Get the maximum value that stochastic_ternary can respresent."""
    if self.alpha is None or isinstance(self.alpha, six.string_types):
      return 1.0
    else:
      return max(1.0, self.alpha)

  def min(self):
    """Get the minimum value that stochastic_ternary can respresent."""
    if self.alpha is None or isinstance(self.alpha, six.string_types):
      return -1.0
    else:
      return -max(1.0, self.alpha)

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def get_config(self):
    config = {
        "alpha": self.alpha,
        "threshold": self.threshold,
        "temperature": self.temperature,
        "use_real_sigmoid": self.use_real_sigmoid,
        "number_of_unrolls": self.number_of_unrolls,
    }
    return config


@quantizer_registry.register_quantizer
class binary(base_quantizer.BaseQuantizer):  # pylint: disable=invalid-name
  """Computes the sign(x) returning a value between -alpha and alpha.

  Although we cannot guarantee E[dL/dy] = E[dL/dx] if we do not use the
  stochastic sampling, we still use the ST approximation.

  Modified from original binary to match QNN implementation.

  The binary qunatizer supports multiple-scales per tensor where:
  - alpha: It can be set to "auto" or "auto_po2" to enable auto-scaling. "auto"
           allows arbitrary scale while "auto_po2" allows power-of-two scales
           only. It can also be set to a fixed value or None (i.e., no scaling).
  - scale_axis: It determines the axis/axes to calculate the auto-scale at.
  - elements_per_scale: It enables fine-grained scaling where it determines
    the number of elements across scale axis/axes that should be grouped into
    one scale.

  Examples:

  1. Input shape = [1, 8, 8, 16] alpha="auto", scale_axis=None,
     elements_per_scale=None --> Number of separate scales = 16

  2. Input shape = [1, 8, 8, 16] alpha="auto", scale_axis=1,
     elements_per_scale=None --> Number of separate scales = 8

  3. Input shape = [1, 8, 8, 16] alpha="auto", scale_axis=1,
     elements_per_scale=2 --> Number of separate scales = 4

  4. Input shape = [1, 8, 8, 16] alpha="auto", scale_axis=[2, 3],
     elements_per_scale=2 --> Number of separate scales = 4*8 = 32

  5. Input shape = [1, 8, 8, 16] alpha="auto", scale_axis=[2, 3],
     elements_per_scale=[2, 4] --> Number of separate scales = 4*4 = 16

  Attributes:
    x: tensor to perform sign_through.
    bits: number of bits to perform quantization.
    use_01: if True, return {0,1} instead of {-1,+1}.
    alpha: binary is -alpha or +alpha, or "auto", "auto_po2" to compute
      automatically.
    use_stochastic_rounding: if true, we perform stochastic rounding.
    elements_per_scale: if set to an int or List[int], we create multiple scales
      per axis across scale_axis, where 'elements_per_scale' represents the
      number of elements/values associated with every separate scale value.
    scale_axis: int or List[int] which axis/axes to calculate scale from.
    min_po2_exponent: if set while using "auto_po2", it represents the minimum
      allowed power of two exponent.
    max_po2_exponent: if set while using "auto_po2", it represents the maximum
      allowed power of two exponent.

  Returns:
    Computation of sign operation with straight through gradient.
  """

  def __init__(self, use_01=False, alpha=None, use_stochastic_rounding=False,
               scale_axis=None, elements_per_scale=None, min_po2_exponent=None,
               max_po2_exponent=None):
    super().__init__()
    self.use_01 = use_01
    self.bits = 1
    self.alpha = alpha
    self.use_stochastic_rounding = use_stochastic_rounding
    self.default_alpha = 1.0
    self.scale = None
    self.scale_axis = scale_axis
    self.elements_per_scale = elements_per_scale
    self.min_po2_exponent = min_po2_exponent
    self.max_po2_exponent = max_po2_exponent

  def __str__(self):
    def list_to_str(l):
      return ",".join([str(x) for x in l])

    flags = []
    if self.use_01:
      flags.append("use_01=" + str(int(self.use_01)))
    if self.alpha is not None:
      alpha = str(self.alpha)
      if isinstance(self.alpha, six.string_types):
        alpha = "'" + alpha + "'"
      flags.append("alpha=" + alpha)
    if self.elements_per_scale is not None:
      if isinstance(self.elements_per_scale, list):
        flags.append("elements_per_scale=[" +
                     list_to_str(self.elements_per_scale) + "]")
      else:
        flags.append("elements_per_scale=" + str(self.elements_per_scale))
    if self.scale_axis is not None:
      if isinstance(self.scale_axis, list):
        flags.append("scale_axis=[" + list_to_str(self.scale_axis) + "]")
      else:
        flags.append("scale_axis=" + str(self.scale_axis))
    if self.min_po2_exponent is not None:
      flags.append("min_po2_exponent=" + str(self.min_po2_exponent))
    if self.max_po2_exponent is not None:
      flags.append("max_po2_exponent=" + str(self.max_po2_exponent))
    if self.use_stochastic_rounding:
      flags.append(
          "use_stochastic_rounding=" + str(self.use_stochastic_rounding))
    return "binary(" + ",".join(flags) + ")"

  def __call__(self, x):
    if isinstance(self.alpha, six.string_types):
      assert self.alpha in ["auto", "auto_po2"]
    if self.alpha is None:
      scale = self.default_alpha
    elif isinstance(self.alpha, six.string_types):
      scale = 1.0
    elif isinstance(self.alpha, np.ndarray):
      scale = self.alpha
    else:
      scale = float(self.alpha)

    if self.use_stochastic_rounding:
      try:
        len_axis = len(x.shape.as_list())
      except AttributeError:
        len_axis = len(list(x.shape))
      if len_axis == 1:
        axis = None
      elif K.image_data_format() == "channels_last":
        axis = list(range(len_axis - 1))
      else:
        axis = list(range(1, len_axis))

      # if stochastic_round is through, we need to scale
      # number so that the precision is small enough.
      # This is especially important if range of x is very
      # small, which occurs during initialization of weights.
      m = K.max(tf.abs(x), axis=axis, keepdims=True)
      m = tf.where(m > 1.0, tf.ones_like(m), m)
      f = 2 * m

      x = tf_utils.smart_cond(
          K.learning_phase(),
          lambda: f * _round_through(
              x / f, use_stochastic_rounding=True, precision=0.125),
          lambda: x)

    k_sign = tf.sign(x)
    if self.use_stochastic_rounding:
      # in inference, we use a biased "1" for stochastic rounding right now
      k_sign += (1.0 - tf.abs(k_sign)) * tf_utils.smart_cond(
          K.learning_phase(),
          lambda: 2.0 * tf.round(tf.random.uniform(tf.shape(x))) - 1.0,
          lambda: tf.ones_like(tf.shape(x), dtype=K.floatx()))
      # if something still remains, just make it positive for now.
    k_sign += (1.0 - tf.abs(k_sign))
    if self.use_01:
      k_sign = (k_sign + 1.0) / 2.0

    # approximate binary by tanh(x) as it has limited range between -1 and +1.
    if self.alpha is None:
      x = K.tanh(x)

    self.scale = _get_least_squares_scale(
        self.alpha,
        x,
        k_sign,
        elements_per_scale=self.elements_per_scale,
        scale_axis=self.scale_axis,
        min_po2_exponent=self.min_po2_exponent,
        max_po2_exponent=self.max_po2_exponent,
    )
    return x + tf.stop_gradient(-x + self.scale * k_sign)

  def _set_trainable_parameter(self):
    if self.alpha is None:
      self.alpha = "auto_po2"

  def max(self):
    """Get maximum value that binary class can respresent."""
    if self.alpha is None or isinstance(self.alpha, six.string_types):
      return 1.0
    else:
      return max(1.0, self.alpha)

  def min(self):
    """Get minimum value that binary class can respresent."""
    if self.use_01:
      return 0.0
    elif self.alpha is None or isinstance(self.alpha, six.string_types):
      return -1.0
    else:
      return -max(1.0, self.alpha)

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def get_config(self):
    config = {
        "use_01": self.use_01,
        "alpha": self.alpha,
        "use_stochastic_rounding": self.use_stochastic_rounding
    }
    return config


@quantizer_registry.register_quantizer
class stochastic_binary(binary):  # pylint: disable=invalid-name
  """Computes a stochastic activation function returning -alpha or +alpha.

  Computes straight-through approximation using random sampling to make
  E[dL/dy] = E[dL/dx], and computing the sign function. See explanation above.

  Attributes:
    x: tensor to perform sign opertion with stochastic sampling.
    alpha: binary is -alpha or +alpha, or "auto" or "auto_po2".
    bits: number of bits to perform quantization.
    temperature: amplifier factor for sigmoid function, making stochastic
      behavior less stochastic as it moves away from 0.
    use_real_sigmoid: use real sigmoid from tensorflow for probablity.

  Returns:
    Computation of sign with stochastic sampling with straight through gradient.
  """

  def __init__(self, alpha=None, temperature=6.0, use_real_sigmoid=True):
    super().__init__(alpha=alpha)
    self.alpha = alpha
    self.bits = 1
    self.temperature = temperature
    self.use_real_sigmoid = use_real_sigmoid
    self.default_alpha = 1.0
    self.scale = None

  def __str__(self):
    flags = []
    if self.alpha is not None:
      alpha = str(self.alpha)
      if isinstance(self.alpha, six.string_types):
        alpha = "'" + alpha + "'"
      flags.append("alpha=" + alpha)
    if self.temperature != 6.0:
      flags.append("temperature=" + str(self.temperature))
    if not self.use_real_sigmoid:
      flags.append("use_real_sigmoid=" + str(int(self.use_real_sigmoid)))
    return "stochastic_binary(" + ",".join(flags) + ")"

  def __call__(self, x):
    def stochastic_output():
      if isinstance(self.alpha, six.string_types):
        assert self.alpha in ["auto", "auto_po2"]
        len_axis = len(x.shape)
        if len_axis > 1:
          if K.image_data_format() == "channels_last":
            axis = list(range(len_axis - 1))
          else:
            axis = list(range(1, len_axis))
        else:
          axis = [0]
        std = K.std(x, axis=axis, keepdims=True) + K.epsilon()
      else:
        std = 1.0

      if self.use_real_sigmoid:
        p = tf.keras.backend.sigmoid(self.temperature * x / std)
      else:
        p = _sigmoid(self.temperature * x / std)

      r = tf.random.uniform(tf.shape(x))
      q = tf.sign(p - r)
      q += 1.0 - tf.abs(q)
      q_non_stochastic = tf.sign(x)
      q_non_stochastic += 1.0 - tf.abs(q_non_stochastic)
      scale = _get_least_squares_scale(self.alpha, x, q_non_stochastic)
      self.scale = scale
      return x + tf.stop_gradient(-x + scale * q)

    output = tf_utils.smart_cond(
        K.learning_phase(), stochastic_output, lambda: binary.__call__(self, x)
    )
    return output

  def _set_trainable_parameter(self):
    if self.alpha is None:
      self.alpha = "auto_po2"

  def max(self):
    """Get the maximum value that stochastic_binary can respresent."""
    if self.alpha is None or isinstance(self.alpha, six.string_types):
      return 1.0
    else:
      return max(1.0, self.alpha)

  def min(self):
    """Get the minimum value that stochastic_binary can respresent."""
    if self.alpha is None or isinstance(self.alpha, six.string_types):
      return -1.0
    else:
      return -max(1.0, self.alpha)

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def get_config(self):
    config = {
        "alpha": self.alpha,
        "temperature": self.temperature,
        "use_real_sigmoid": self.use_real_sigmoid,
    }
    return config


@quantizer_registry.register_quantizer
class quantized_relu(base_quantizer.BaseQuantizer):  # pylint: disable=invalid-name
  """Computes a quantized relu to a number of bits.

  Modified from:

  [https://github.com/BertMoons/QuantizedNeuralNetworks-Keras-Tensorflow]

  Assume h(x) = +1 with p = sigmoid(x), -1 otherwise, the expected value of
  h(x) is:

  E[h(x)] = +1 P(p <= sigmoid(x)) - 1 P(p > sigmoid(x))
          = +1 P(p <= sigmoid(x)) - 1 ( 1 - P(p <= sigmoid(x)) )
          = 2 P(p <= sigmoid(x)) - 1
          = 2 sigmoid(x) - 1, if p is sampled from a uniform distribution U[0,1]

  If use_sigmoid is 0, we just keep the positive numbers up to
  2**integer * (1 - 2**(-bits)) instead of normalizing them, which is easier
  to implement in hardware.

  Attributes:
    bits: number of bits to perform quantization.
    integer: number of bits to the left of the decimal point.
    use_sigmoid: if true, we apply sigmoid to input to normalize it.
    negative_slope: slope when activation < 0, needs to be power of 2.
    use_stochastic_rounding: if true, we perform stochastic rounding.
    relu_upper_bound: A float representing an upper bound of the unquantized
      relu. If None, we apply relu without the upper bound when
      "is_quantized_clip" is set to false (true by default).
      Note: The quantized relu uses the quantization parameters (bits and
      integer) to upper bound. So it is important to set relu_upper_bound
      appropriately to the quantization parameters. "is_quantized_clip"
      has precedence over "relu_upper_bound" for backward compatibility.
    is_quantized_clip: A boolean representing whether the inputs are clipped to
      the maximum value represented by the quantization parameters. This
      parameter is deprecated, and the default is set to True for backwards
      compatibility. Users are encouraged to use "relu_upper_bound" instead.
    qnoise_factor: float. a scalar from 0 to 1 that represents the level of
      quantization noise to add. This controls the amount of the quantization
      noise to add to the outputs by changing the weighted sum of
      (1 - qnoise_factor)*unquantized_x + qnoise_factor*quantized_x.
    var_name: String or None. A variable name shared between the tf.Variables
      created in the build function. If None, it is generated automatically.
    use_ste: Bool. Whether to use "straight-through estimator" (STE) method or
        not.
    use_variables: Bool. Whether to make the quantizer variables to be dynamic
      tf.Variables or not.

  Returns:
    Function that performs relu + quantization to bits >= 0.
  """

  def __init__(self,
               bits=8,
               integer=0,
               use_sigmoid=0,
               negative_slope=0.0,
               use_stochastic_rounding=False,
               relu_upper_bound=None,
               is_quantized_clip=True,
               qnoise_factor=1.0,
               var_name=None,
               use_ste=True,
               use_variables=False):
    super().__init__()
    self.bits = bits
    self.integer = integer
    self.use_sigmoid = use_sigmoid
    self.negative_slope = negative_slope
    self.use_stochastic_rounding = use_stochastic_rounding
    self.relu_upper_bound = relu_upper_bound
    self.is_quantized_clip = is_quantized_clip
    self.qnoise_factor = qnoise_factor
    self.use_ste = use_ste
    assert negative_slope >= 0.0
    if negative_slope != 0.0:
      assert np.mod(np.log2(negative_slope), 1) == 0
    self.var_name = var_name
    self.use_variables = use_variables

  def __str__(self):
    # Converts Tensors to printable strings by converting to a numpy array and
    # then using regex to remove brackets when there is only one integer bit
    integer_bits = re.sub(
        r"\[(\d)\]", r"\g<1>",
        str(self.integer.numpy() if isinstance(self.integer, tf.Variable
                                              ) else self.integer))

    flags = [str(self.bits), integer_bits]
    if self.use_sigmoid or self.use_stochastic_rounding:
      flags.append(str(int(self.use_sigmoid)))
    if self.negative_slope:
      flags.append(str(self.negative_slope))
    if self.use_stochastic_rounding:
      flags.append(str(int(self.use_stochastic_rounding)))
    return "quantized_relu(" + ",".join(flags) + ")"

  def __call__(self, x):
    if not self.built:
      self.build(var_name=self.var_name, use_variables=self.use_variables)

    non_sign_bits = self.bits - (self.negative_slope != 0.0)
    x = K.cast(x, dtype="float32")
    m = K.cast(K.pow(2, non_sign_bits), dtype="float32")
    m_i = K.cast(K.pow(2, self.integer), dtype="float32")

    # is_quantized_clip has precedence over relu_upper_bound for backward
    # compatibility.
    m_f = K.cast(
        K.pow(
            tf.constant(2., tf.float32),
            K.cast(self.integer, dtype="float32") - non_sign_bits),
        dtype="float32")
    if self.is_quantized_clip:
      x_u = tf.where(x <= m_i - m_f, K.relu(x, alpha=self.negative_slope),
                     tf.ones_like(x) * (m_i - m_f))
    elif self.relu_upper_bound is not None:
      x_u = tf.where(x <= self.relu_upper_bound,
                     K.relu(x, alpha=self.negative_slope),
                     tf.ones_like(x) * self.relu_upper_bound)
    else:
      x_u = K.relu(x, alpha=self.negative_slope)

    if self.use_sigmoid:
      p = _sigmoid(x / m_i) * m
      xq = m_i * tf.keras.backend.clip(
          2.0 * (_round_through(p, self.use_stochastic_rounding) / m) - 1.0,
          0.0, 1.0 - 1.0 / m)
      if self.negative_slope > 0:
        neg_factor = 1 / (self.negative_slope * m)
        xq = xq + m_i * self.negative_slope * tf.keras.backend.clip(
            2.0 * (_round_through(p * self.negative_slope,
                                  self.use_stochastic_rounding) * neg_factor) -
            1.0, -1.0, 0.0)
    else:
      p = x * m / m_i
      xq = m_i * tf.keras.backend.clip(
          _round_through(p, self.use_stochastic_rounding) / m, 0.0,
          1.0 - 1.0 / m)
      if self.negative_slope > 0:
        neg_factor = 1 / (self.negative_slope * m)
        xq = xq + m_i * self.negative_slope * (
            tf.keras.backend.clip(
                _round_through(p * self.negative_slope,
                               self.use_stochastic_rounding) * neg_factor, -1.0,
                0.0))

    if self.relu_upper_bound and not self.is_quantized_clip:
      xq = tf.where(xq <= self.relu_upper_bound, xq,
                    tf.ones_like(xq) * self.relu_upper_bound)

    if self.use_ste:
      return x_u + tf.stop_gradient(self.qnoise_factor * (-x_u + xq))
    else:
      return (1 - self.qnoise_factor) * x_u + tf.stop_gradient(
          self.qnoise_factor * xq)

  def max(self):
    """Get the maximum value that quantized_relu can represent."""
    unsigned_bits = self.bits - (self.negative_slope != 0.0)

    if unsigned_bits > 0:
      return max(
          1.0,
          np.array(
              K.pow(2.0, K.cast(self.integer, dtype="float32")),
              dtype="float32"))
    else:
      return 1.0

  def min(self):
    """Get the minimum value that quantized_relu can represent."""
    if self.negative_slope == 0.0:
      return 0.0

    unsigned_bits = self.bits - 1
    if unsigned_bits > 0:
      return min(
          -0.0, -self.negative_slope * np.array(
              K.pow(2.0, K.cast(self.integer, dtype="float32")),
              dtype="float32"))
    else:
      return -1.0

  def range(self):
    """Returns a list of all values that quantized_relu can represent

      ordered by their binary representation ascending.
    """
    assert self.use_sigmoid == 0  # current unsupported
    assert self.negative_slope == 0  # # unsupported unsupported
    x = np.asarray(range(2**self.bits))
    return x * np.array(
        K.pow(2.0, -self.bits + K.cast(self.integer, dtype="float32")),
        dtype="float32")

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def get_config(self):
    config = {
        "bits":
            self.bits,
        "integer":
            self.integer.numpy() if isinstance(self.integer, tf.Variable) else
            self.integer,
        "use_sigmoid":
            self.use_sigmoid,
        "negative_slope":
            self.negative_slope,
        "use_stochastic_rounding":
            self.use_stochastic_rounding,
        "relu_upper_bound":
            self.relu_upper_bound,
        "qnoise_factor":
            self.qnoise_factor.numpy() if isinstance(
                self.qnoise_factor, tf.Variable) else self.qnoise_factor
    }
    return config


@quantizer_registry.register_quantizer
class quantized_ulaw(base_quantizer.BaseQuantizer):  # pylint: disable=invalid-name
  """Computes a u-law quantization.

  Attributes:
    bits: number of bits to perform quantization.
    integer: number of bits to the left of the decimal point.
    symmetric: if true, we will have the same number of values for positive
      and negative numbers.
    u: parameter of u-law

  Returns:
    Function that performs ulaw + quantization to bits in the range -1.0 to 1.0.
  """

  def __init__(self, bits=8, integer=0, symmetric=0, u=255.0):
    super().__init__()
    self.bits = bits
    self.integer = integer
    self.symmetric = symmetric
    self.u = u

  def __str__(self):
    flags = [str(self.bits), str(self.integer)]
    if self.symmetric or self.u != 255.0:
      flags.append(str(int(self.symmetric)))
    if self.u != 255.0:
      flags.append(str(self.u))
    return "quantized_ulaw(" + ",".join(flags) + ")"

  def __call__(self, x):
    non_sign_bits = self.bits - 1
    m = pow(2, non_sign_bits)
    m_i = pow(2, self.integer)
    p = _sigmoid(x / m_i) * m
    rp = 2.0 * (_round_through(p) / m) - 1.0
    u_law_p = tf.sign(rp) * tf.keras.backend.log(
        1 + self.u * tf.abs(rp)) / tf.keras.backend.log(1 + self.u)
    xq = m_i * tf.keras.backend.clip(u_law_p, -1.0 +
                                     (1.0 * self.symmetric) / m, 1.0 - 1.0 / m)
    return xq

  def max(self):
    """Get the maximum value that quantized_ulaw can represent."""
    unsigned_bits = self.bits - 1

    if unsigned_bits > 0:
      return max(1.0, np.power(2.0, self.integer))
    else:
      return 1.0

  def min(self):
    """Get the minimum value that quantized_ulaw can represent."""
    unsigned_bits = self.bits - 1

    if unsigned_bits > 0:
      return -max(1.0, np.power(2.0, self.integer))
    else:
      return -1.0

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def get_config(self):
    config = {
        "bits": self.bits,
        "integer": self.integer,
        "symmetric": self.symmetric,
        "u": self.u
    }
    return config


@quantizer_registry.register_quantizer
class quantized_tanh(base_quantizer.BaseQuantizer):  # pylint: disable=invalid-name
  """Computes a quantized tanh to a number of bits.

  Modified from:

  [https://github.com/BertMoons/QuantizedNeuralNetworks-Keras-Tensorflow]

  Attributes:
    bits: number of bits to perform quantization.
    use_stochastic_rounding: if true, we perform stochastic rounding.
    symmetric: if true, we will have the same number of values for positive
      and negative numbers.
    use_real_tanh: if true, use the tanh function from Keras backend,
      if false, use tanh that is defined as 2 * sigmoid(x) - 1

  Returns:
    Function that performs tanh + quantization to bits in the range -1.0 to 1.0.
  """

  def __init__(self, bits=8, use_stochastic_rounding=False,
               symmetric=False, use_real_tanh=False):
    super().__init__()
    self.bits = bits
    self.symmetric = symmetric
    self.use_stochastic_rounding = use_stochastic_rounding
    self.use_real_tanh = use_real_tanh

  def __str__(self):
    flags = [str(self.bits)]
    if self.use_stochastic_rounding:
      flags.append(str(int(self.use_stochastic_rounding)))
    if self.symmetric:
      flags.append(str(int(self.symmetric)))
    if self.use_real_tanh:
      flags.append(str(int(self.use_real_tanh)))
    return "quantized_tanh(" + ",".join(flags) + ")"

  def __call__(self, x):
    non_sign_bits = self.bits - 1
    x = K.cast_to_floatx(x)
    m = K.cast_to_floatx(K.pow(2, non_sign_bits))
    p = K.tanh(x) if self.use_real_tanh else 2.0 * _sigmoid(x) - 1.0
    return tf.keras.backend.clip(
                                 (_round_through(p * m, self.use_stochastic_rounding) / m),
                                 -1.0 + (1.0 * self.symmetric) / m,
                                 1.0 - 1.0 / m)

  def max(self):
    """Get the maximum value that quantized_tanh can represent."""
    return 1.0 - 1.0 / pow(2, self.bits - 1)

  def min(self):
    """Get the minimum value that quantized_tanh can represent."""
    return -1.0 + (1.0 * self.symmetric) / pow(2, self.bits - 1)

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def get_config(self):
    config = {
        "bits": self.bits,
        "symmetric": self.symmetric,
        "use_stochastic_rounding": self.use_stochastic_rounding,
        "use_real_tanh": self.use_real_tanh
    }
    return config


@quantizer_registry.register_quantizer
class quantized_sigmoid(base_quantizer.BaseQuantizer):  # pylint: disable=invalid-name
  """Computes a quantized sigmoid to a number of bits.

  Attributes:
    bits: number of bits to perform quantization.
    symmetric: if true, we will have the same number of values for positive
      and negative numbers.
    use_real_sigmoid: if true, will use the sigmoid from Keras backend
    use_stochastic_rounding: if true, we perform stochastic rounding.

  Returns:
    Function that performs sigmoid + quantization to bits in the range 0.0 to 1.0.
  """

  def __init__(self, bits=8, symmetric=False,
               use_real_sigmoid=False,
               use_stochastic_rounding=False):
    super().__init__()
    self.bits = bits
    self.symmetric = symmetric
    self.use_real_sigmoid = use_real_sigmoid
    self.use_stochastic_rounding = use_stochastic_rounding

  def __str__(self):
    flags = [str(self.bits)]
    if self.symmetric:
      flags.append(str(int(self.symmetric)))
    if self.use_real_sigmoid:
      flags.append(str(int(self.use_real_sigmoid)))
    if self.use_stochastic_rounding:
      flags.append(str(int(self.use_stochastic_rounding)))
    return "quantized_sigmoid(" + ",".join(flags) + ")"

  def __call__(self, x):
    x = K.cast_to_floatx(x)
    m = K.cast_to_floatx(K.pow(2, self.bits))

    p = K.sigmoid(x) if self.use_real_sigmoid else _sigmoid(x)

    return tf.keras.backend.clip((_round_through(p*m, self.use_stochastic_rounding) / m),
                                 (1.0 * self.symmetric) / m,
                                 1.0 - 1.0 / m)

  def max(self):
    """Get the maximum value that quantized_sigmoid can represent."""
    return 1.0 - 1.0 / pow(2, self.bits)

  def min(self):
    """Get the minimum value that quantized_sigmoid can represent."""
    return (1.0 * self.symmetric) / pow(2, self.bits)

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def get_config(self):
    config = {
        "bits": self.bits,
        "symmetric": self.symmetric,
        "use_real_sigmoid": self.use_real_sigmoid,
        "use_stochastic_rounding": self.use_stochastic_rounding
    }
    return config


def _clip_power_of_two(x_abs,
                       min_exp,
                       max_exp,
                       max_value,
                       quadratic_approximation=False,
                       use_stochastic_rounding=False,
                       log2_rounding="rnd"):
  """Clips a tensor using power-of-two quantizer.


  Args:
    x_abs: A tensor object. Its elements should be non-negative.
    min_exp: An integer representing the smallest exponent.
    max_exp: An integer representing the largest exponent.
    max_value: A float or None. If it is None, we clip the value to max_value.
    quadratic_approximation: An boolean representing whether the quadratic
      approximation is applied.
    use_stochastic_rounding: An boolean representing whether the stochastic
      rounding method is applied.
    log2_rounding: log2 rounding mode. "rnd" and "floor" currently
      supported, corresponding to tf.round and tf.floor respectively.

  Returns:
    A tensor object, the values are clipped by min_exp and max_exp.
  """

  # if quadratic_approximation is True, round to the exponent for sqrt(x),
  # so that the return value can be divided by two without remainder.
  log2 = np.log(2.0)

  # When the elements of x_abs are small than the keras epsilon,
  # we just overwrite x_abs with eps
  eps = tf.keras.backend.epsilon()
  x_filter = tf.where(x_abs < eps, eps, x_abs)
  if max_value is not None:
    # If the elements of x_filter has value larger than x_value, clip it.
    x_filter = tf.where(x_filter >= max_value,
                        tf.ones_like(x_filter) * max_value, x_filter)

  def power_of_two_clip(x_abs, min_exp, max_exp, quadratic_approximation,
                        use_stochastic_rounding, log2_rounding):
    assert log2_rounding in ["rnd", "floor"]

    if quadratic_approximation:
      q_factor = 2.0
      x_input = tf.sqrt(x_abs)
    else:
      q_factor = 1.0
      x_input = x_abs

    if log2_rounding == "floor":
      x_log2 = _floor_through(tf.keras.backend.log(x_input) / log2)
    elif use_stochastic_rounding:
      x_log2 = tf_utils.smart_cond(
          K.learning_phase(),
          lambda: stochastic_round_po2(x_input),
          lambda: _round_through(tf.keras.backend.log(x_input) / log2))
    else:
      x_log2 = _round_through(tf.keras.backend.log(x_input) / log2)

    x_clipped = q_factor * tf.keras.backend.clip(x_log2, min_exp, max_exp)
    return x_clipped

  x_clipped = tf.where(
      x_abs < eps,
      tf.ones_like(x_abs) * min_exp,
      power_of_two_clip(x_filter, min_exp, max_exp, quadratic_approximation,
                        use_stochastic_rounding, log2_rounding))

  return x_clipped


def _need_exponent_sign_bit_check(max_value):
  """Checks whether the sign bit of exponent is needed.

  This is used by quantized_po2 and quantized_relu_po2.

  Args:
    max_value: the maximum value allowed.

  Returns:
    An integer. 1: sign_bit is needed. 0: sign_bit is not needed.
  """

  if max_value is not None:
    if max_value < 0:
      raise ValueError("po2 max_value should be non-negative.")
    if max_value > 1:
      # if max_value is larger than 1,
      #   the exponent could be positive and negative.
      #   e.g., log(max_value) > 0 when max_value > 1
      need_exponent_sign_bit = 1
    else:
      need_exponent_sign_bit = 0
  else:
    # max_value is not specified, so we cannot decide the range.
    # Then we need to put sign_bit for exponent to be safe
    need_exponent_sign_bit = 1
  return need_exponent_sign_bit


def _get_min_max_exponents(non_sign_bits, need_exponent_sign_bit,
                           quadratic_approximation):
  """Given a bitwidth, gets min and max exponents that it can represent.

  Args:
    non_sign_bits: An integer representing the bitwidth of the exponent.
    need_exponent_sign_bit: An integer representing whether it needs sign bit
      in exponent. (1: need sign bit. 0: sign bit is not needed.)
    quadratic_approximation: A boolean representing whether the quadratic
      approximiation method is enforced.

  Returns:
    A tuple of integers: min_exp, max_exp
  """
  effect_bits = non_sign_bits - need_exponent_sign_bit
  min_exp = -2**(effect_bits)
  max_exp = 2**(effect_bits) - 1
  if quadratic_approximation:
    max_exp = 2 * (max_exp // 2)
  return min_exp, max_exp


@quantizer_registry.register_quantizer
class quantized_po2(base_quantizer.BaseQuantizer):  # pylint: disable=invalid-name
  """Quantizes to the closest power of 2.

  Attributes:
    bits: An integer, the bits allocated for the exponent, its sign and the sign
      of x.
    max_value: An float or None. If None, no max_value is specified.
      Otherwise, the maximum value of quantized_po2 <= max_value
    use_stochastic_rounding: A boolean, default is False, if True, it uses
      stochastic rounding and forces the mean of x to be x statstically.
    quadratic_approximation: A boolean, default is False if True, it forces the
      exponent to be even number that closted to x.
    log2_rounding: A string, log2 rounding mode. "rnd" and "floor" currently
      supported, corresponding to tf.round and tf.floor respectively.
    qnoise_factor: float. a scalar from 0 to 1 that represents the level of
      quantization noise to add. This controls the amount of the quantization
      noise to add to the outputs by changing the weighted sum of
      (1 - qnoise_factor)*unquantized_x + qnoise_factor*quantized_x.
    var_name: String or None. A variable name shared between the tf.Variables
      created in the build function. If None, it is generated automatically.
    use_ste: Bool. Whether to use "straight-through estimator" (STE) method or
        not.
    use_variables: Bool. Whether to make the quantizer variables to be dynamic
      tf.Variables or not.
  """

  def __init__(self,
               bits=8,
               max_value=None,
               use_stochastic_rounding=False,
               quadratic_approximation=False,
               log2_rounding="rnd",
               qnoise_factor=1.0,
               var_name=None,
               use_ste=True,
               use_variables=False):
    super().__init__()
    self.bits = bits
    self.max_value = max_value
    self.use_stochastic_rounding = use_stochastic_rounding
    self.log2_rounding = log2_rounding
    # if True, round to the exponent for sqrt(x),
    # so that the return value can be divided by two without remainder.
    self.quadratic_approximation = quadratic_approximation
    need_exponent_sign_bit = _need_exponent_sign_bit_check(self.max_value)
    non_sign_bits = self.bits - 1
    self._min_exp, self._max_exp = _get_min_max_exponents(
        non_sign_bits, need_exponent_sign_bit, self.quadratic_approximation)
    # qnoise_factor related attributes
    self.qnoise_factor = qnoise_factor
    self.use_ste = use_ste
    self.var_name = var_name
    self.use_variables = use_variables

  def __str__(self):
    flags = [str(self.bits)]
    if self.max_value is not None or self.use_stochastic_rounding:
      flags.append(str(int(self.max_value)))
    if self.use_stochastic_rounding:
      flags.append(str(int(self.use_stochastic_rounding)))
    if self.quadratic_approximation:
      flags.append(
          "quadratic_approximation=" + str(int(self.quadratic_approximation)))
    return "quantized_po2(" + ",".join(flags) + ")"

  def __call__(self, x):
    if not self.built:
      self.build(var_name=self.var_name, use_variables=self.use_variables)

    x_sign = tf.sign(x)
    x_sign += (1.0 - tf.abs(x_sign))
    x_abs = tf.abs(x)
    x_clipped = _clip_power_of_two(x_abs, self._min_exp, self._max_exp,
                                   self.max_value,
                                   self.quadratic_approximation,
                                   self.use_stochastic_rounding,
                                   self.log2_rounding)
    xq = x_sign * pow(2.0, x_clipped)

    if self.use_ste:
      return x + tf.stop_gradient(self.qnoise_factor * (-x + xq))
    else:
      return (1 - self.qnoise_factor) * x + tf.stop_gradient(
          self.qnoise_factor * xq)

  def max(self):
    """Get the maximum value that quantized_po2 can represent."""
    if self.max_value:
      return max(1.0, self.max_value)
    else:
      return max(1.0, 2**self._max_exp)

  def min(self):
    """Get the minimum value that quantized_po2 can represent."""
    if self.max_value:
      return -max(1.0, self.max_value)
    else:
      return -max(1.0, 2**self._max_exp)

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def get_config(self):
    """Gets configugration of the quantizer.

    Returns:
      A dict mapping quantization configuration, including
        bits: bitwidth for exponents.
        max_value: the maximum value of this quantized_po2 can represent.
        use_stochastic_rounding:
          if True, stochastic rounding is used.
        quadratic_approximation:
          if True, the exponent is enforced to be even number, which is
          the closest one to x.
        log2_rounding:
          A string, Log2 rounding mode
    """
    config = {
        "bits":
            self.bits,
        "max_value":
            self.max_value,
        "use_stochastic_rounding":
            self.use_stochastic_rounding,
        "quadratic_approximation":
            self.quadratic_approximation,
        "qnoise_factor":
            self.qnoise_factor.numpy() if isinstance(
                self.qnoise_factor, tf.Variable) else self.qnoise_factor,
        "log2_rounding":
            self.log2_rounding
    }
    return config


@quantizer_registry.register_quantizer
class quantized_relu_po2(base_quantizer.BaseQuantizer):  # pylint: disable=invalid-name
  """Quantizes x to the closest power of 2 when x > 0

  Attributes:
    bits: An integer, the bits allocated for the exponent and its sign.
    max_value: default is None, or a non-negative value to put a constraint for
      the max value.
    negative_slope: slope when activation < 0, needs to be power of 2.
    use_stochastic_rounding: A boolean, default is False, if True, it uses
      stochastic rounding and forces the mean of x to be x statstically.
    quadratic_approximation: A boolean, default is False if True, it forces the
      exponent to be even number that is closest to x.
    log2_rounding: A string, log2 rounding mode. "rnd" and "floor" currently
      supported, corresponding to tf.round and tf.floor respectively.
    qnoise_factor: float. a scalar from 0 to 1 that represents the level of
      quantization noise to add. This controls the amount of the quantization
      noise to add to the outputs by changing the weighted sum of
      (1 - qnoise_factor)*unquantized_x + qnoise_factor*quantized_x.
    var_name: String or None. A variable name shared between the tf.Variables
      created in the build function. If None, it is generated automatically.
    use_ste: Bool. Whether to use "straight-through estimator" (STE) method or
        not.
    use_variables: Bool. Whether to make the quantizer variables to be dynamic
      tf.Variables or not.
  """

  def __init__(self,
               bits=8,
               max_value=None,
               negative_slope=0,
               use_stochastic_rounding=False,
               quadratic_approximation=False,
               log2_rounding="rnd",
               qnoise_factor=1.0,
               var_name=None,
               use_ste=True,
               use_variables=False):
    super().__init__()
    self.bits = bits
    self.max_value = max_value
    self.negative_slope = negative_slope
    self.use_stochastic_rounding = use_stochastic_rounding
    self.log2_rounding = log2_rounding
    # if True, round to the exponent for sqrt(x),
    # so that the return value can be divided by two without remainder.
    self.quadratic_approximation = quadratic_approximation
    need_exponent_sign_bit = _need_exponent_sign_bit_check(self.max_value)
    self._min_exp = -2**(self.bits - need_exponent_sign_bit)
    self._max_exp = 2**(self.bits - need_exponent_sign_bit) - 1
    if self.quadratic_approximation:
      self._max_exp = 2 * (self._max_exp // 2)

    assert negative_slope >= 0.0
    if negative_slope != 0:
      assert np.mod(np.log2(negative_slope), 1) == 0
    # qnoise_factor related attributes
    self.qnoise_factor = qnoise_factor
    self.use_ste = use_ste
    self.var_name = var_name
    self.use_variables = use_variables

  def __str__(self):
    flags = [str(self.bits)]
    if self.max_value is not None or self.use_stochastic_rounding:
      flags.append(str(int(self.max_value)))
    if self.negative_slope:
      flags.append(str(self.negative_slope))
    if self.use_stochastic_rounding:
      flags.append(str(int(self.use_stochastic_rounding)))
    if self.quadratic_approximation:
      flags.append(
          "quadratic_approximation=" + str(int(self.quadratic_approximation)))
    return "quantized_relu_po2(" + ",".join(flags) + ")"

  def __call__(self, x):
    if not self.built:
      self.build(var_name=self.var_name, use_variables=self.use_variables)

    x_original = x

    if self.max_value is None:
      x = K.relu(x, self.negative_slope)
    else:
      x = tf.where(
          x <= self.max_value,
          K.relu(x, self.negative_slope),
          tf.ones_like(x) * self.max_value)

    x_pos_clipped = _clip_power_of_two(
        K.relu(x_original),
        self._min_exp, self._max_exp,
        self.max_value,
        self.quadratic_approximation,
        self.use_stochastic_rounding,
        self.log2_rounding)

    x_neg_clipped = _clip_power_of_two(
        K.relu(-x_original) * self.negative_slope,
        self._min_exp, self._max_exp,
        self.max_value,
        self.quadratic_approximation,
        self.use_stochastic_rounding,
        self.log2_rounding)

    xq = tf.where(
        tf.logical_or(x_original >= 0.0, self.negative_slope == 0.0),
        pow(2.0, x_pos_clipped), -pow(2.0, x_neg_clipped))

    if self.use_ste:
      return x + tf.stop_gradient(self.qnoise_factor * (-x + xq))
    else:
      return (1 - self.qnoise_factor) * x + tf.stop_gradient(
          self.qnoise_factor * xq)

  def max(self):
    """Get the maximum value that quantized_relu_po2 can represent."""
    if self.max_value:
      return max(1.0, self.max_value)
    else:
      return max(1.0, 2**self._max_exp)

  def min(self):
    """Get the minimum value that quantized_relu_po2 can represent."""
    if self.negative_slope == 0.0:
      return 2**self._min_exp

    unsigned_bits = self.bits - 1
    if unsigned_bits > 0:
      return min(2**self._min_exp, - self.negative_slope * np.power(2.0, unsigned_bits))
    else:
      return 2**self._min_exp

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def get_config(self):
    """Gets configugration of the quantizer.

    Returns:
      A dict mapping quantization configuration, including
        bits: bitwidth for exponents.
        max_value: the maximum value of this quantized_relu_po2 can represent.
        use_stochastic_rounding:
          if True, stochastic rounding is used.
        quadratic_approximation:
          if True, the exponent is enforced to be even number, which is
          the closest one to x.
        log2_rounding:
          A string, Log2 rounding mode

    """

    config = {
        "bits":
            self.bits,
        "max_value":
            self.max_value,
        "negative_slope":
            self.negative_slope,
        "use_stochastic_rounding":
            self.use_stochastic_rounding,
        "quadratic_approximation":
            self.quadratic_approximation,
        "qnoise_factor":
            self.qnoise_factor.numpy() if isinstance(
                self.qnoise_factor, tf.Variable) else self.qnoise_factor,
        "log2_rounding":
            self.log2_rounding
    }
    return config


@quantizer_registry.register_quantizer
class quantized_hswish(quantized_bits):  # pylint: disable=invalid-name
  """Computes a quantized hard swish to a number of bits.

  # TODO(mschoenb97): Update to inherit from quantized_linear.

  Equation of h-swisth function in mobilenet v3:
  hswish(x) = x * ReluY(x + relu_shift) / Y
  Y is relu_upper_bound

  Attributes:
    bits: number of bits to perform quantization, also known as word length.
    integer: number of integer bits.
    symmetric: if True,  the quantization is in symmetric mode, which puts
      restricted range for the quantizer. Otherwise, it is in asymmetric mode,
      which uses the full range.
    alpha: a tensor or None, the scaling factor per channel. If None, the
      scaling factor is 1 for all channels.
    use_stochastic_rounding: if true, we perform stochastic rounding. This
      parameter is passed on to the underlying quantizer quantized_bits which is
      used to quantize h_swish.
    scale_axis: which axis to calculate scale from
    qnoise_factor: float. a scalar from 0 to 1 that represents the level of
      quantization noise to add. This controls the amount of the quantization
      noise to add to the outputs by changing the weighted sum of (1 -
      qnoise_factor)*unquantized_x + qnoise_factor*quantized_x.
    var_name: String or None. A variable name shared between the tf.Variables
      created in the build function. If None, it is generated automatically.
    use_ste: Bool. Whether to use "straight-through estimator" (STE) method or
      not.
    use_variables: Bool. Whether to make the quantizer variables to be dynamic
      tf.Variables or not.
    relu_shift: integer type, representing the shift amount of the unquantized
      relu.
    relu_upper_bound: integer type, representing an upper bound of the
      unquantized relu. If None, we apply relu without the upper bound when
      "is_quantized_clip" is set to false (true by default).
      Note: The quantized relu uses the quantization parameters (bits and
        integer) to upper bound. So it is important to set relu_upper_bound
        appropriately to the quantization parameters. "is_quantized_clip" has
        precedence over "relu_upper_bound" for backward compatibility.
  """

  def __init__(
      self,
      bits=8,
      integer=0,
      symmetric=0,
      alpha=None,
      use_stochastic_rounding=False,
      scale_axis=None,
      qnoise_factor=1.0,
      var_name=None,
      use_variables=False,
      relu_shift: int = 3,
      relu_upper_bound: int = 6,
  ):
    super().__init__(
        bits=bits,
        integer=integer,
        symmetric=symmetric,
        keep_negative=True,
        alpha=alpha,
        use_stochastic_rounding=use_stochastic_rounding,
        scale_axis=scale_axis,
        qnoise_factor=qnoise_factor,
        var_name=var_name,
        use_variables=use_variables,
    )

    self.relu_shift = relu_shift
    self.relu_upper_bound = relu_upper_bound

  def __str__(self):
    """Converts Tensors to printable strings."""

    integer_bits = re.sub(
        r"\[(\d)\]",
        r"\g<1>",
        str(
            self.integer.numpy()
            if isinstance(self.integer, tf.Variable)
            else self.integer
        ),
    )
    assert isinstance(integer_bits, int)

    flags = [
        str(self.bits),
        integer_bits,
        str(int(self.symmetric)),
        "relu_shift=" + str(self.relu_shift),
        "relu_upper_bound=" + str(self.relu_upper_bound),
    ]

    if not self.keep_negative:
      flags.append("keep_negative=False")
    if self.alpha:
      alpha = str(self.alpha)
      if isinstance(self.alpha, six.string_types):
        alpha = "'" + alpha + "'"
      flags.append("alpha=" + alpha)
    if self.use_stochastic_rounding:
      flags.append(
          "use_stochastic_rounding=" + str(int(self.use_stochastic_rounding))
      )
    return "quantized_hswish(" + ",".join(flags) + ")"

  def __call__(self, x):
    assert self.relu_upper_bound > 0, (
        "relu_upper_bound must be a positive value, "
        f"found {self.relu_upper_bound} instead"
    )
    assert (
        self.relu_shift > 0
    ), f"relu_shift must be a positive value, found {self.relu_shift} instead"
    x = K.cast_to_floatx(x)
    shift_x = x + self.relu_shift
    relu_x = tf.where(
        shift_x <= self.relu_upper_bound,
        K.relu(shift_x, alpha=False),
        tf.ones_like(shift_x) * self.relu_upper_bound,
    )

    hswish_x = tf.math.multiply(x, relu_x) / self.relu_upper_bound
    return super(quantized_hswish, self).__call__(hswish_x)

  def min(self):
    """Gets the minimum value that quantized_hswish can represent."""

    # get the minimum value that the number of bits can represent
    min_quant = super(quantized_hswish, self).min()
    # In the negative end, the hswish function becomes
    # x * (x + relu_shift) / relu_upper_bound
    # the min value of this parabolic function is
    # - relu_shift^2 / (4 * relu_upper_bound)
    denom = 4 * self.relu_upper_bound
    min_parabolic = -self.relu_shift * self.relu_shift / denom

    if min_quant >= min_parabolic:
      return min_quant

    # get the quantized value of min_parabolic
    return super(quantized_hswish, self).call(min_parabolic)

  def get_config(self):
    """Add relu_shift and relu_upper_bound to the config file."""

    base_config = super(quantized_hswish, self).get_config()

    config = {
        "relu_shift": self.relu_shift,
        "relu_upper_bound": self.relu_upper_bound,
    }

    out_config = dict(list(base_config.items()) + list(config.items()))
    return out_config


# TODO(akshayap): Update to use registry for quantizers instead of globals().
def get_quantizer(identifier):
  """Gets the quantizer.

  Args:
    identifier: An quantizer, which could be dict, string, or callable function.

  Returns:
    A quantizer class or quantization function from this file. For example,
      Quantizer classes: quantized_bits, quantized_po2, quantized_relu_po2,
      binary, stochastic_binary, ternary, stochastic_ternary, etc.

      Quantization functions: binary_sigmoid, hard_sigmoid, soft_sigmoid, etc.

  Raises:
    ValueError: An error occurred when quantizer cannot be interpreted.
  """

  if identifier is None:
    return None
  if isinstance(identifier, dict):
    return deserialize_keras_object(
        identifier, module_objects=globals(), printable_module_name="quantizer")
  elif isinstance(identifier, six.string_types):
    return safe_eval(identifier, globals())
  elif callable(identifier):
    return identifier
  else:
    raise ValueError("Could not interpret quantizer identifier: " +
                     str(identifier))


def get_quantized_initializer(w_initializer, w_range):
  """Gets the initializer and scales it by the range."""

  if isinstance(w_initializer, six.string_types):

    if w_initializer == "he_normal":
      return initializers.VarianceScaling(
          scale=2 * w_range, mode="fan_in", distribution="normal", seed=None)
    if w_initializer == "he_uniform":
      return initializers.VarianceScaling(
          scale=2 * w_range, mode="fan_in", distribution="uniform", seed=None)
    elif w_initializer == "glorot_normal":
      return initializers.VarianceScaling(
          scale=w_range, mode="fan_avg", distribution="normal", seed=None)
    elif w_initializer == "glorot_uniform":
      return initializers.VarianceScaling(
          scale=w_range, mode="fan_avg", distribution="uniform", seed=None)
    elif w_initializer == "random_uniform":
      return initializers.RandomUniform(-w_range, w_range)

  return w_initializer
