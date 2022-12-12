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
"""Power-of-2 quantizers based on https://arxiv.org/pdf/2210.03671.pdf.

  Example usages:
  < MSQE-based quantizer >
    Default (using the second moments MSQE optimization and the outlier mask):
      quantized_bits_msqe_po2(bits=4)
    Per-channel quantization:
      quantized_bits_msqe_po2(bits=4, scale_axis=3, per_channel_scale=True)

  < Gradient-based (learnable) quantizer >
    Default (using the MSQE round (Round-to-Lower-MSQE)):
      quantized_bits_learnable_po2(bits=4)
    Per-channel quantization:
      quantized_bits_learnable_po2(bits=4, scale_axis=3, per_channel_scale=True)
    Relu activation (the MSQE round is not supported for non-variable tensors):
      quantized_bits_learnable_po2(bits=4, keep_negative=False,
      use_second_moments_msqe_opt=False, use_po2_scale_msqe_round=False)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import re
import numpy as np
from six.moves import range
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer


def _update_ema_variable(variable, new_val, ema_decay, is_initialized,
                         should_update):
  """Updates exponentially moving average (EMA) of a tf.Variable.

    This function directly updates the variable.

  Args:
    variable: A tf.Variable to be updated.
    new_val: A tensor with a new value to update 'variable'. Its shape is same
      as 'variable'.
    ema_decay: A scalar python float or tensor. EMA decay factor.
    is_initialized: A scalar tensor indicating whether 'variable' has been
      initialized or not.
    should_update: A scalar python bool or tensor indicating whether to update
      'variable' or not.
  """
  if not tf.is_tensor(should_update):
    should_update = tf.convert_to_tensor(should_update)

  val_to_update = ema_decay * variable + (1.0 - ema_decay) * new_val
  val_to_update = tf.cond(is_initialized, lambda: val_to_update,
                          lambda: new_val)
  val_to_update = tf.cond(should_update, lambda: val_to_update,
                          lambda: variable)
  variable.assign(val_to_update)


def _get_scaling_axis(scale_axis, len_axis):
  """Gets the axis to perform scaling with.

  Args:
    scale_axis: an integer scalar tensor or None to get which axis to calculate
      scale from. If None, the scaling axis is set based on the image data
      format.
    len_axis: an integer scalar tensor of the dimension of the tensor to be
      quantized.

  Returns:
    A list of axes to be quantized together.
  """

  if scale_axis is not None:
    axis = list(range(scale_axis))
    axis += list(range(scale_axis + 1, len_axis))
  else:
    if K.image_data_format() == "channels_last":
      axis = list(range(len_axis - 1))
    else:
      axis = list(range(1, len_axis))
  return axis


def _get_msqe_scale(x,
                    q,
                    scale_axis=None,
                    per_channel_scale=True,
                    msqe_weight=None):
  """Gets scaling factor for scaling the tensor per channel.

  It uses a linear least squares method to find the scaling factor.
  (https://en.wikipedia.org/wiki/Linear_least_squares)

  Args:
     x: A tensor object. Its elements are in float.
     q: A tensor object. Its elements are in quantized format of x.
     scale_axis: which axis to calculate scale from
     per_channel_scale: A bool. Whether to perform per-channel scaling or not.
     msqe_weight: A tensor object or None. Its elements are in float, which are
     used to perform weighted least squares optimization. If None, it performs
     non-weighted least squares optimization.

  Returns:
    A scaling factor tensor or scalar for scaling tensor per channel or per
    layer.
  """
  # in different tensorflow version (e.g., 2.4)
  # x.shape is a tuple which doesn't have as_list() method
  try:
    x_shape = x.shape.as_list()
  except AttributeError:
    x_shape = list(x.shape)

  len_axis = len(x_shape)

  if msqe_weight is not None:
    sqrt_msqe_weight = tf.math.sqrt(msqe_weight)
    x = tf.math.multiply(x, sqrt_msqe_weight)
    q = tf.math.multiply(q, sqrt_msqe_weight)

  if not per_channel_scale:
    qx = K.mean(q * x, keepdims=True)
    qq = K.mean(q * q, keepdims=True)
  else:
    if len_axis > 1:
      axis = _get_scaling_axis(scale_axis, len_axis)
      qx = K.mean(tf.math.multiply(q, x), axis=axis, keepdims=True)
      qq = K.mean(tf.math.multiply(q, q), axis=axis, keepdims=True)
    else:
      # No summing (averaging) along the channel axis to get per-channel
      # scales.
      qx = tf.math.multiply(q, x)
      qq = tf.math.multiply(q, q)

  scale = qx / (qq + K.epsilon())

  # Rounds the exponent to the nearest integer for power-of-2 scale.
  return K.pow(2.0, tf.math.rint(K.log(scale + K.epsilon()) / np.log(2.0)))


class BaseQuantizerPO2(Layer):  # pylint: disable=invalid-name
  """This is the base class from which all power-of-2 quantizers inherit, which
  is based on the reference paper (https://arxiv.org/pdf/2210.03671.pdf).

  Attributes:
    bits: Integer, number of bits to perform quantization.
    keep_negative: Boolean, if true, it keeps negative values and sets the
      quantization levels symmetrically around 0. If false, negative numbers is
      clipped to 0.
    scale_axis: Integer, which axis to calculate scale from.
    per_channel_scale: Boolean, whether to perform per-channel (true) or
      per-layer (false) quantization.
    init_scale: Float or None, initial scale factor to initialize the scale with
      (if None, it will be initialized based on the first inputs.).
    use_second_moments_msqe_opt: Bool, whether to use the second moments based
      MSQE optimization or not. The second moments is used as a weighting factor
      to calculate the quantization error.
    second_moments_ema_decay: Float, EMA decay factor for the second moments
      update.
    use_sqrt_of_msqe_weight: Bool, whether to use square root of MSQE weight.
    use_outlier_mask_msqe_weight: Bool, whether to apply outlier mask.
    use_stable_scale_exponent: Bool, whether to use exponentially moving
      averaged ("stable") scale exponent or not. Note: there is a tf.Variable
        (self.switch_to_stable_scale) that controls when to apply the stable
        scale exponent (i.e., if use_stable_scale_exponent is true and
        self.switch_to_stable_scale is false, the stable scale exponent is
        updated but not used.).
    stable_scale_ema_decay: Float, EMA decay factor for the stable scale update.
    is_gradient_based: Bool, whether to optimize the scale_exponent from the
      gradients or not (i.e, if true, self.scale_exponent is set to be
      "trainable".)
  """

  def __init__(self,
               bits=4,
               keep_negative=True,
               scale_axis=None,
               per_channel_scale=False,
               init_scale=None,
               use_second_moments_msqe_opt=False,
               second_moments_ema_decay=0.999,
               use_sqrt_of_msqe_weight=True,
               use_outlier_mask_msqe_weight=True,
               use_stable_scale_exponent=False,
               stable_scale_ema_decay=0.99,
               is_gradient_based=True,
               **kwargs):

    self.bits = bits
    self.keep_negative = keep_negative
    self.scale_axis = scale_axis
    self.per_channel_scale = per_channel_scale
    self.init_scale = init_scale
    self.use_second_moments_msqe_opt = use_second_moments_msqe_opt
    self.second_moments_ema_decay = second_moments_ema_decay
    self.use_sqrt_of_msqe_weight = use_sqrt_of_msqe_weight
    self.use_outlier_mask_msqe_weight = use_outlier_mask_msqe_weight
    self.use_stable_scale_exponent = use_stable_scale_exponent
    self.stable_scale_ema_decay = stable_scale_ema_decay
    self.is_gradient_based = is_gradient_based
    self.alpha = "auto_po2"

    # scale exponent to be learned.
    self.scale_exponent = None
    # Stores the power-of-2 scale factor used for quantization.
    self.scale = None
    # Axes to perform reduce sum (mean) operation.
    self.reduce_axes = None
    # Running averaged gradient variances of the input
    self.msqe_weight = None
    # A knob to switch to "stable_scale_exponent".
    self.switch_to_stable_scale = None
    # variable holding the running averaged scale exponent
    self.stable_scale_exponent = None
    # Indicator variable whether to update stable_scale_exponent or not. This
    # can be used as an indicator whether it is in training or not.
    self.should_update_stable_scale_exponent = None
    # The assignments from "kwargs" are to restore from the config.
    # The maximum quantization level of negative numbers.
    self.qn = kwargs.pop("qn") if "qn" in kwargs else None
    # The maximum quantization level of positive numbers.
    self.qp = kwargs.pop("qp") if "qp" in kwargs else None
    # Axes scaled together.
    self.scaled_axes = kwargs.pop(
        "scaled_axes") if "scaled_axes" in kwargs else None

    super().__init__(**kwargs)

  def build(self, input_shape):
    """Creates and initializes variables."""
    # Number of quantization levels.
    levels = tf.math.pow(2.0, tf.cast(self.bits, dtype=tf.float32)) - 1

    # Sets the number of quantization levels for the negative and positive
    # ranges.
    if self.keep_negative:
      # Sets them symmetric about 0 to reduce the quantization induced bias.
      self.qn = float((levels - 1.0) / 2.0)
      self.qp = float((levels - 1.0) / 2.0)
    else:
      self.qn = 0.0
      self.qp = float(levels)

    if self.init_scale is None:
      init_scale_exponent = 0.0
      init_scale = 1.0
    else:
      init_scale = self.init_scale + K.epsilon()
      init_scale_exponent = tf.math.log(init_scale) / tf.math.log(2.0)

    if self.scale_axis is None:
      self.scale_axis = self._get_scale_axis(input_shape)

    self.scaled_axes = self._get_scaled_axes(self.scale_axis, input_shape)

    if self.per_channel_scale:
      scale_exponent_shape = tf.TensorShape([
          input_shape[i] if i == self.scale_axis else 1
          for i in range(len(input_shape))
      ])
    else:
      scale_exponent_shape = [1 for i in range(len(input_shape))]

    # Creates the scale exponent variable to be learned.
    self.scale_exponent = tf.Variable(
        lambda: tf.constant(
            init_scale_exponent, shape=scale_exponent_shape, dtype=tf.float32),
        trainable=self.is_gradient_based,
        synchronization=tf.VariableSynchronization.ON_READ,
        aggregation=tf.compat.v1.VariableAggregation.MEAN,
        name="scale_exponent")

    # "self.scale" is not a trainable variable which gets assigned not learned.
    self.scale = tf.Variable(
        lambda: tf.constant(
            init_scale, shape=scale_exponent_shape, dtype=tf.float32),
        trainable=False,
        synchronization=tf.VariableSynchronization.ON_READ,
        aggregation=tf.compat.v1.VariableAggregation.MEAN,
        name="scale")

    self.reduce_axes = [
        i for i in range(len(self.scale_exponent.shape))
        if self.scale_exponent.shape[i] == 1
    ]

    if self.use_second_moments_msqe_opt:
      msqe_weight_shape = tf.TensorShape(
          [1 if s is None else s for s in input_shape])
      self.msqe_weight = tf.Variable(
          lambda: tf.ones(shape=msqe_weight_shape),
          trainable=False,
          dtype=tf.float32,
          name="msqe_weight")

    if self.use_stable_scale_exponent:
      self.stable_scale_exponent = tf.Variable(
          lambda: tf.zeros_like(self.scale_exponent),
          dtype=tf.float32,
          trainable=False,
          synchronization=tf.VariableSynchronization.ON_READ,
          aggregation=tf.compat.v1.VariableAggregation.MEAN,
          name="stable_scale_exponent")
      self.switch_to_stable_scale = tf.Variable(
          False, trainable=False, name="switch_to_stable_scale")
      self.should_update_stable_scale_exponent = tf.Variable(
          False, trainable=False, name="should_update_stable_scale_exponent")

    # Inidicator variable for initializing variables (e.g, the scale exponent
    # etc.).
    self.is_initialized = tf.Variable(
        False, trainable=False, name="is_initialized")

  def call(self, inputs, msqe_weight=None):
    """Returns a fake quantized tensor of 'inputs'.

    Args:
      inputs: A tensor to be fake quantized.
      msqe_weight: A tensor which is used in the scale optimization to weight
        the MSQE (Mean Squared Quantization Error) of individual input elements.
        Its shape is same as 'inputs' and its dtype is `float32` If None, it
        will be set by "self._get_msqe_weight" (this should be left as None
        unless you explicitly assign its value in a different way.).

    Returns:
      A tensor of fake quantized input. Its shape is same as 'inputs' and its
      dtype is `float32`.
    """
    if not self.keep_negative:
      # Quantize only positive values (e.g. relu activation).
      inputs = tf.keras.activations.relu(inputs)

    if self.use_second_moments_msqe_opt:
      return self._update_second_moments_msqe_weight(
          self._quantize(inputs, msqe_weight=msqe_weight), inputs)

    return self._quantize(inputs, msqe_weight=msqe_weight)

  def _quantize(self, inputs, msqe_weight=None):
    """Returns (fake) quantized inputs and optimizes the scaling factor.

    Args:
      inputs: A tensor to be fake quantized and used in optimizing the scaling
        factor.
      msqe_weight: A tensor or None, which is used in the MSQE optimizations.

    Returns:
      A tensor of fake quantized inputs.
    """
    # Initialize self.scale_exponent (it is initialized only once).
    self._initialize_scale_exponent(inputs)

    scale = self._get_scale(inputs, msqe_weight=msqe_weight)

    if self.use_stable_scale_exponent:
      # Only outputs the stable scale when 'self.switch_to_stable_scale' is set
      # to true, which is false by default.
      scale = self._get_stable_scale(scale)

    # Stores the scaling factors used for quantization.
    self.scale.assign(scale)

    # Perform rounding.
    inputs_rounded = self._round_quant(inputs / scale)

    # Perform clipping.
    inputs_clipped = self._clip_quant(inputs_rounded)
    inputs_quant = scale * inputs_clipped

    # Update initialization indicator.
    self.is_initialized.assign(True)

    return inputs_quant

  @tf.custom_gradient
  def _update_second_moments_msqe_weight(self, input_quantized, inputs):
    """Updates the second moments of the gradients respect to the inputs.

    Args:
      input_quantized: A tensor which is the output from 'self._quantize' method
        (fake quantized input).
      inputs: A tensor which is the input to 'self._quantize' method.

    Returns:
      'input_quantized', the upstream gradient of 'input_quantized', and the
      gradients (zeros) of 'inputs'
    """

    def grad(upstream_grad):
      """Calculates and updates the second moments of the gradients."""
      # Get a mask for clipped inputs (i.e., 1.0 for rounded inputs and
      # 0.0 for clipped inputs). self.scale is the previously used scaling
      # factors.
      clip_error_mask = self._get_clipped_inputs_mask(inputs, self.scale)
      # Calculate the second moments of the gradients respect to 'inputs' that
      # is clip_error_mask * upstream_grad.
      second_moments = clip_error_mask * upstream_grad * upstream_grad

      # Update the second moments
      _update_ema_variable(
          self.msqe_weight,
          second_moments,
          self.second_moments_ema_decay,
          self.is_initialized,
          should_update=True)
      return upstream_grad, tf.zeros_like(inputs)

    return input_quantized, grad

  @abc.abstractmethod
  def _get_scale(self, inputs=None, reduce_axes=None, msqe_weight=None):
    """Returns power-of-2 scaling factors for quantization.

    Args:
      inputs: A tensor to be used to optimize the scale value.
      reduce_axes: A list of axes to be summed (averaged) over.
      msqe_weight: A tensor which is used in scale optimization to weight the
        MSQE (Mean Squared Quantization Error) of individual input elements. Its
        shape is same as 'inputs' and its dtype is `float32`.

    Returns:
      A tensor of power-of-2 scaling factors. Its shape is same as
      'self.scale_exponent' and its dtype is `float32`.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def _get_init_scale_exponent(self, inputs):
    """Returns a scale exponent tensor to initialize "self.scale_exponent".

    Args:
      inputs: A tensor to be used to calculate initial scale exponent values.

    Returns:
      A tensor of scale exponent. Its shape is same as 'self.scale_exponent' and
      its dtype is `float32`.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def _get_outlier_mask(self, inputs):
    """Returns a tensor to suppress outliers in the input for MSQE optimizations.

    Args:
      inputs: A tensor to be used to generate the outlier mask.

    Returns:
      A tensor to mask out the outliers of the inputs. Its shape is same as
      'inputs' and its dtype is `float32`.
    """
    raise NotImplementedError

  def _get_msqe_weight(self, inputs=None):
    """Returns weighting factors for MSQE optimizations.

    Args:
      inputs: A tensor to be used to generate the outlier mask.

    Returns:
      A tensor to be used as weighting factors for MSQE optimizations or None.
      Note: it is assumed that when 'None' is returned, no weighting factors
      will be applied for MSQE optimizations.

    Raises:
      ValueError: if 'inputs' is None when self.use_outlier_mask_msqe_weight is
      True.
    """
    if self.use_outlier_mask_msqe_weight and inputs is None:
      raise ValueError(
          f"inputs must not be None if self.use_outlier_mask_msqe_weight is"
          f" True.")

    if self.msqe_weight is None:
      # Only returns the outlier mask
      return self._get_outlier_mask(
          inputs) if self.use_outlier_mask_msqe_weight else None

    msqe_weight = self.msqe_weight

    if self.use_sqrt_of_msqe_weight:
      # To use square rooted msqe_weight
      msqe_weight = tf.math.sqrt(msqe_weight)

    if self.use_outlier_mask_msqe_weight:
      # Returns the outlier mask modulated msqe_weight
      msqe_weight = msqe_weight * self._get_outlier_mask(inputs)

    return msqe_weight

  def _get_stable_scale(self, scale):
    """Updates and returns power-of-2 'stable' scaling factors.

    It updates the exponential moving average (EMA) of the scale exponent when
    self.should_update_stable_scale_exponent is true and
    self.switch_to_stable_scale is false, and returns scaling factor based on
    the stable (EMAed) scale exponent when self.switch_to_stable_scale is set
    true else returns passed-in 'scale'.

    Args:
      scale: A tensor of power-of-2 scaling factors.

    Returns:
      A tensor of power-of-2 scaling factors.
    """
    # Freezes updating exponential moving average of self.stable_scale_exponent
    # when self.should_update_stable_scale_exponent is false or
    # self.switch_to_stable_scale is set True.
    should_update = tf.logical_and(self.should_update_stable_scale_exponent,
                                   not self.switch_to_stable_scale)
    # Update the stable (EMAed) scale exponent.
    # Note: when 'self.is_initialized' is false, 'self.stable_scale_exponent' is
    # assigned with the scale exponent of the 'scale' input otherwise it is
    # updated with exponential moving average.
    stable_scale = self._update_stable_scale_exponent(scale, should_update,
                                                      self.is_initialized)
    # Use the stable scale only when self.switch_to_stable_scale is set True.
    scale = tf.cond(self.switch_to_stable_scale, lambda: stable_scale,
                    lambda: scale)
    return scale

  def _update_stable_scale_exponent(self, scale, should_update, is_initialized):
    """Updates and returns stable (EMAed) power-of-2 scaling factors.

    It performs exponential moving average on the scale exponent, not on the
    scale itself.

    Args:
      scale: a tensor to be used to update exponential moving average of scale
        exponents.
      should_update: A bool. Whether to update exponential moving average of
        scale exponents.
      is_initialized: A bool. Whether to initialize the stable scale exponent.

    Returns:
      A tensor of (stable) power-of-2 scaling factors
    """
    scale_exponent = self._get_po2_scale_exponent(scale)
    _update_ema_variable(
        self.stable_scale_exponent,
        scale_exponent,
        ema_decay=self.stable_scale_ema_decay,
        is_initialized=is_initialized,
        should_update=should_update)
    return tf.math.pow(2.0, tf.math.rint(self.stable_scale_exponent))

  def _initialize_scale_exponent(self, inputs):
    """Initializes the scale exponent only once.

    It only initializes 'self.scale_exponent' once when there is no preset
    initial scaling factor (i.e., self.init_scale is None).

    Args:
      inputs: A tensor, where the initial scale exponent is based on.
    """
    update_cond = tf.math.logical_and(not self.is_initialized,
                                      self.init_scale is None)
    scale_exponent_to_init = tf.cond(
        update_cond,
        lambda: tf.stop_gradient(self._get_init_scale_exponent(inputs)),
        lambda: self.scale_exponent)
    self.scale_exponent.assign(scale_exponent_to_init)

  def _get_clipped_inputs_mask(self, inputs, scale):
    """Returns a tensor to mask out the clipped inputs.

    The mask has 1.0 for the rounded inputs and 0.0 for the clipped inputs.

    Args:
      inputs: A tensor to get the clipping mask from.
      scale: A tensor of the scaling factor.

    Returns:
      A tensor to mask out the clipped inputs.
    """
    inputs_rounded = tf.math.rint(inputs / scale)
    clip_error_mask = tf.math.logical_and(
        tf.less_equal(inputs_rounded, self.qp),
        tf.greater_equal(inputs_rounded, -self.qn))
    return tf.cast(clip_error_mask, tf.float32)

  def _get_scale_axis(self, input_shape):
    """Returns the scaling axis based on the input shape.

    Args:
      input_shape: a tuple of integers which is the size of the input channels.

    Returns:
      A scalar value.
    """
    if K.image_data_format() == "channels_last":
      scale_axis = (len(input_shape) - 1) if len(input_shape) else 0
    else:
      scale_axis = 1 if input_shape[0] is None else 0
    return scale_axis

  def _get_scaled_axes(self, scale_axis, input_shape):
    """Returns the axes scaled together.

    Args:
      scale_axis: an integer of the scaling axis.
      input_shape: a tuple of integers which is the size of the input channels.

    Returns:
      A list of integers.
    """
    if self.per_channel_scale:
      scaled_axes = list(range(scale_axis))
    else:
      scaled_axes = list(range(len(input_shape)))
    return scaled_axes

  def _clip_quant(self, inputs):
    """Returns clipped inputs (scale-normalized) by the quantization levels.

    Args:
      inputs: A tensor (scale-normalized input value).

    Returns:
      A tensor clipped by the quantization levels.
    """
    return tf.minimum(tf.maximum(inputs, -self.qn), self.qp)

  def _round_quant(self, inputs):
    """Returns rounded inputs using a straight-through estimator (STE).

    Args:
      inputs: A tensor to be rounded.

    Returns:
      A tensor through a straight-through estimator.
    """
    return inputs + tf.stop_gradient(-inputs + tf.math.rint(inputs))

  def _simple_quantize(self, inputs, scale, should_return_q=False):
    """Returns quantized inputs without a straight-through estimator (STE).

    Args:
      inputs: A tensor to be quantized.
      scale: A tensor of the scaling factor.
      should_return_q: if true, quantized inputs in integer will be also
        returned.

    Returns:
      A tensor of fake quantized inputs (, a tensor of quantized inputs)
    """
    inputs_rounded = tf.math.rint(inputs / scale)
    inputs_clipped = self._clip_quant(inputs_rounded)
    if should_return_q:
      return scale * inputs_clipped, inputs_clipped
    else:
      return scale * inputs_clipped

  def _get_po2_scale(self, scale):
    """Returns power-of-2 constrained scaling factors.

    Args:
      scale: A tensor to be power-of-2 constrained.

    Returns:
      A tensor (power-of-2 constrained scaling factor).
    """
    return tf.math.pow(2.0, self._get_po2_scale_exponent(scale))

  def _get_po2_scale_exponent(self, scale):
    """Returns power-of-2 constrained scale exponent.

    Args:
      scale: A tensor to get power-of-2 scale exponent from.

    Returns:
      A tensor constrained to be in integer values.
    """
    scale_exponent = tf.math.log(scale + K.epsilon()) / tf.math.log(2.0)
    return tf.round(scale_exponent)

  def _calculate_msqe(self, x, xq, reduce_axes=None, msqe_weight=None):
    """Returns the mean squared quantization error (MSQE).

    Args:
      x: a tensor of the original inputs.
      xq: a tensor of the fake quantized inputs.
      reduce_axes: A list of axes to be summed (averaged) over or None. If None,
        self.reduce_axes will be used.
      msqe_weight: A tensor or None. If None, no weighting is applied in the
        MSQE calculation.

    Returns:
      A tensor of the MSQE
    """
    if reduce_axes is None:
      reduce_axes = self.reduce_axes
    msqe = tf.math.pow(x - xq, 2.0)
    if msqe_weight is not None:
      msqe *= msqe_weight
    return tf.reduce_sum(msqe, axis=reduce_axes, keepdims=True)

  def _calculate_msqe_inputs(self,
                             inputs,
                             scale,
                             reduce_axes=None,
                             msqe_weight=None):
    """Returns the mean squared quantization error (MSQE) of the inputs.

    Args:
      inputs: a tensor to calculate the MSQE from.
      scale: a tensor to scale (quantize) the input with.
      reduce_axes: A list of axes to be summed (averaged) over or None. If None,
        self.reduce_axes will be used.
      msqe_weight: A tensor or None. If None, no weighting is applied in the
        MSQE calculation.

    Returns:
      A tensor of the MSQE
    """
    inputs_quant = self._simple_quantize(inputs, scale)
    return self._calculate_msqe(
        inputs, inputs_quant, reduce_axes=reduce_axes, msqe_weight=msqe_weight)

  def _least_squares_msqe_scale(self,
                                inputs,
                                scale,
                                reduce_axes=None,
                                msqe_weight=None,
                                num_lls_iters=3,
                                should_return_msqe=False):
    """Returns power-of-2 scaling factors from linear least squares regression.

    Args:
      inputs: a tensor to optimize the scaling factor from.
      scale: a tensor to be used as initial  quantize the input with.
      reduce_axes: A list of axes to be summed (averaged) over or None. If None,
        self.reduce_axes will be used.
      msqe_weight: A tensor or None. If None, no weighting is applied in the
        linear least squares regression.
      num_lls_iters: An integer. Number of linear least squares regression
        iterations.
      should_return_msqe: A bool. Whether to return the MSQE of the inputs.

    Returns:
      A tensor of power-of-2 scaling factors (, a tensor of the MSQE)
    """
    if reduce_axes is None:
      reduce_axes = self.reduce_axes

    best_scale = tf.identity(scale)
    xq, q = self._simple_quantize(inputs, best_scale, should_return_q=True)
    best_msqe = self._calculate_msqe(inputs, xq, reduce_axes, msqe_weight)

    for _ in range(num_lls_iters):
      # performs linear least squares regression
      new_scale = _get_msqe_scale(
          x=inputs,
          q=q,
          scale_axis=self.scale_axis,
          per_channel_scale=self.per_channel_scale,
          msqe_weight=msqe_weight)
      xq, q = self._simple_quantize(inputs, new_scale, should_return_q=True)
      new_msqe = self._calculate_msqe(inputs, xq, reduce_axes, msqe_weight)

      # Update the best scale and the best msqe
      best_scale = tf.where(new_msqe < best_msqe, new_scale, best_scale)
      best_msqe = tf.where(new_msqe < best_msqe, new_msqe, best_msqe)

    if should_return_msqe:
      return best_scale, best_msqe
    else:
      return best_scale

  def _line_search_msqe_scale(self,
                              inputs,
                              scale,
                              reduce_axes=None,
                              msqe_weight=None,
                              line_search_range=6,
                              should_return_msqe=False):
    """Returns power-of-2 scaling factors from line search.

    Args:
      inputs: a tensor to optimize the scaling factor from.
      scale: a tensor to be used as initial  quantize the input with.
      reduce_axes: A list of axes to be summed (averaged) over or None. If None,
        self.reduce_axes will be used.
      msqe_weight: A tensor or None. If None, no weighting is applied in the
        line search.
      line_search_range: An integer. Search range of the line search.
      should_return_msqe: A bool. Whether to return the MSQE of the inputs.

    Returns:
      A tensor of power-of-2 scaling factors (, a tensor of the MSQE)
    """
    if reduce_axes is None:
      reduce_axes = self.reduce_axes

    best_scale = tf.identity(scale)
    xq = self._simple_quantize(inputs, best_scale)
    best_msqe = self._calculate_msqe(inputs, xq, reduce_axes, msqe_weight)
    best_scale_exponent = self._get_po2_scale_exponent(best_scale)

    # PO2 exponent search offsets
    end_range = line_search_range // 2 + 1
    po2_exponent_offsets = [i for i in range(-end_range+1,end_range) if i != 0]
    for exp_offset in po2_exponent_offsets:
      # Optimize scale
      new_scale = tf.math.pow(2.0, best_scale_exponent + exp_offset)
      xq = self._simple_quantize(inputs, new_scale)
      new_msqe = self._calculate_msqe(inputs, xq, reduce_axes, msqe_weight)
      # Update the best scale and msqe
      best_scale = tf.where(new_msqe < best_msqe, new_scale, best_scale)
      best_msqe = tf.where(new_msqe < best_msqe, new_msqe, best_msqe)

    if should_return_msqe:
      return best_scale, best_msqe
    else:
      return best_scale

  def _optimize_msqe_scale(self,
                           inputs,
                           scale,
                           reduce_axes=None,
                           msqe_weight=None,
                           num_lls_iters=None,
                           should_line_search=True,
                           line_search_range=None):
    """Returns optimized power-of-2 scaling factors.

    It performs an iterative linear least squares regression and an optional
    line search to find optimal power-of-2 scaling factors for the given inputs
    from the initial scaling factors ('scale').

    Args:
      inputs: a tensor to find optimal power-of-2 scaling factors for.
      scale: a tensor to be used as initial scaling factors.
      reduce_axes: A list of axes to be summed (averaged) over or None. If None,
        self.reduce_axes will be used.
      msqe_weight: A tensor or None. If None, no weighting is applied in the
        optimizations.
      num_lls_iters: An integer. Number of linear least squares regression
        iterations.
      should_line_search: A bool. Whether to perform a line search.
      line_search_range: An integer. Search range of the line search.

    Returns:
      A tensor of power-of-2 scaling factors, A tensor of the MSQE
    """
    if reduce_axes is None:
      reduce_axes = self.reduce_axes
    if num_lls_iters is None:
      num_lls_iters = self.num_lls_iters
    if line_search_range is None:
      line_search_range = self.line_search_range

    scale, msqe = self._least_squares_msqe_scale(
        inputs,
        scale,
        reduce_axes=self.reduce_axes,
        msqe_weight=msqe_weight,
        num_lls_iters=num_lls_iters,
        should_return_msqe=True)

    if should_line_search:
      scale, msqe = self._line_search_msqe_scale(
          inputs,
          scale,
          reduce_axes=self.reduce_axes,
          msqe_weight=msqe_weight,
          line_search_range=line_search_range,
          should_return_msqe=True)

    # Having an additional '_get_po2_scale' is just to make sure returning
    # scaling factors are in power-of-2.
    return self._get_po2_scale(scale), msqe

  def max(self):
    """Returns the maximum value that the quantizer can represent."""
    if hasattr(self, "is_initialized") and self.is_initialized.numpy():
      return self._get_scale() * self.qp
    else:
      return 1.0

  def min(self):
    """Returns the minimum value that the quantizer can represent."""
    if self.keep_negative:
      if hasattr(self, "is_initialized") and self.is_initialized.numpy():
        return self._get_scale() * (-self.qn)
      else:
        return -1.0
    else:
      return 0.0


class quantized_bits_learnable_po2(BaseQuantizerPO2):  # pylint: disable=invalid-name
  """Quantizes the number to a number of bits by learnable scale factors.
  For more details, see https://arxiv.org/abs/2210.03671.

  The implementation was inspired by "TRAINED QUANTIZATION THRESHOLDS FOR
  ACCURATE AND EFFICIENT FIXED-POINT INFERENCE OF DEEP NEURAL NETWORKS"
  (https://arxiv.org/pdf/1903.08066.pdf).

  Attributes:
    bits: Integer, number of bits to perform quantization.
    keep_negative: Boolean, if true, it keeps negative values and sets the
      quantization levels symmetrically around 0. If false, negative numbers is
      clipped to 0.
    scale_axis: Integer, which axis to calculate scale from.
    per_channel_scale: Boolean, whether to perform per-channel (true) or
      per-layer (false) quantization.
    init_scale: Float or None, initial scale factor to initialize the scale with
      (if None, it will be initialized based on the first inputs.).
    use_second_moments_msqe_opt: Bool, whether to use the second moments based
      MSQE optimization or not.
    second_moments_ema_decay: Float, EMA decay factor for the second moments
      update.
    use_sqrt_of_msqe_weight: Bool, whether to use square root of MSQE weight.
    use_outlier_mask_msqe_weight: Bool, whether to apply outlier mask.
    use_stable_scale_exponent: Bool, whether to use exponentially moving
      averaged ("stable") scale exponent or not. Note: there is a tf.Variable
        (self.switch_to_stable_scale) that controls when to apply the stable
        scale exponent (i.e., if use_stable_scale_exponent is true and
        self.switch_to_stable_scale is false, the stable scale exponent is
        updated but not used.).
    stable_scale_ema_decay: Float, EMA decay factor for the stable scale update.
    min_init_scale: float or None. minimum initial scale value. If None, the
      initial scale value is not bounded by a minimum value. It is useful to
      prevent zero initial scale value for inputs with all zeros (e.g., bias).
    use_po2_scale_ceil: Bool, whether to use ceil function for constraining
      power-of-2 scale exponents. If false, round function is used instead.
    use_po2_scale_msqe_round: Bool, whether to use MSQE rounding function for
      constraining power-of-2 scale exponents. Note: MSQE rounding has
        precedence over ceil and round function.
  """

  def __init__(self,
               bits=4,
               keep_negative=True,
               scale_axis=None,
               per_channel_scale=False,
               init_scale=None,
               use_second_moments_msqe_opt=True,
               second_moments_ema_decay=0.999,
               use_sqrt_of_msqe_weight=True,
               use_outlier_mask_msqe_weight=True,
               use_stable_scale_exponent=False,
               stable_scale_ema_decay=0.99,
               min_init_scale=0.00001,
               use_po2_scale_ceil=True,
               use_po2_scale_msqe_round=True,
               **kwargs):

    self.min_init_scale = min_init_scale
    self.use_po2_scale_ceil = use_po2_scale_ceil
    self.use_po2_scale_msqe_round = use_po2_scale_msqe_round

    # An indicator variable to control usage of MSQE rounding function, which is
    # set to true by default (i.e, if use_po2_scale_msqe_round is true, MSQE
    # rounding is used by default based on self.switch_to_msqe_round.). It can
    # be used to delay using MSQE rounding.
    self.switch_to_msqe_round = None

    super().__init__(
        bits=bits,
        keep_negative=keep_negative,
        scale_axis=scale_axis,
        per_channel_scale=per_channel_scale,
        init_scale=init_scale,
        use_second_moments_msqe_opt=use_second_moments_msqe_opt,
        second_moments_ema_decay=second_moments_ema_decay,
        use_sqrt_of_msqe_weight=use_sqrt_of_msqe_weight,
        use_outlier_mask_msqe_weight=use_outlier_mask_msqe_weight,
        use_stable_scale_exponent=use_stable_scale_exponent,
        stable_scale_ema_decay=stable_scale_ema_decay,
        is_gradient_based=True,
        **kwargs)

  def __str__(self):
    # Convert Tensors to printable strings by converting to a numpy array and
    # then using regex to remove brackets when there is only one integer bit.
    ptn, repl = r"\[(\d)\]", r"\g<1>"
    bits = re.sub(
        ptn, repl,
        str(self.bits.numpy() if isinstance(self.bits, tf.Variable) else self
            .bits))

    flags = []
    flags.append("bits=" + str(int(bits)))
    flags.append("keep_negative=" + str(self.keep_negative))
    flags.append("scale_axis=" + str(self.scale_axis))
    flags.append("per_channel_scale=" + str(self.per_channel_scale))
    flags.append("init_scale=" + str(self.init_scale))
    flags.append("use_second_moments_msqe_opt=" +
                 str(self.use_second_moments_msqe_opt))
    flags.append("second_moments_ema_decay=" +
                 str(self.second_moments_ema_decay))
    flags.append("use_outlier_mask_msqe_weight=" +
                 str(self.use_outlier_mask_msqe_weight))
    flags.append("use_sqrt_of_msqe_weight=" + str(self.use_sqrt_of_msqe_weight))
    flags.append("use_stable_scale_exponent=" +
                 str(self.use_stable_scale_exponent))
    flags.append("stable_scale_ema_decay=" + str(self.stable_scale_ema_decay))
    flags.append("min_init_scale=" + str(self.min_init_scale))
    flags.append("use_po2_scale_ceil=" + str(self.use_po2_scale_ceil))
    flags.append("use_po2_scale_msqe_round=" +
                 str(self.use_po2_scale_msqe_round))
    return "quantized_bits_learnable_po2(" + ",".join(flags) + ")"

  def build(self, input_shape):
    """Creates and initializes variables."""
    super().build(input_shape)

    if self.use_po2_scale_msqe_round:
      self.switch_to_msqe_round = tf.Variable(
          True, trainable=False, name="switch_to_msqe_round")

  def _get_init_scale_exponent(self, inputs):
    """Returns inputs distribution based initial scale exponent values.

    Args:
      inputs: A tensor to be used to calculate initial scale exponent values.

    Returns:
      A tensor of initial scale exponent values.
    """
    std = tf.math.reduce_std(inputs, axis=self.reduce_axes, keepdims=True)
    # Uses 3 sigma percentile to get scale
    scale = 3.0 * std / tf.cast(self.qp, dtype=tf.float32)

    # Prevents zero scale values for inputs with all zeros (e.g., bias).
    if self.min_init_scale is not None:
      scale = tf.math.maximum(scale, self.min_init_scale)

    # Returns scale exponent
    return tf.math.log(scale) / tf.math.log(2.0)

  def _get_outlier_mask(self, inputs):
    """Returns a tensor to mask outliers in the input for MSQE optimizations.

    The outlier threshold is based on the (unconstrained) output dynamic range
    of the quantizer.

    Args:
      inputs: A tensor to be used to generate the outlier mask.

    Returns:
      A tensor to mask out the outliers of the inputs. Its shape is same as
      'inputs' and its dtype is `float32`.
    """
    # Calculates the output (unconstrained) dynamic range of the quantizer (i.e.
    # , self.scale_exponent is not power-of-2 constrained.).
    outlier_threshold = tf.math.pow(2.0, self.scale_exponent) * (self.qp + 0.5)
    return tf.where(
        abs(inputs) <= outlier_threshold,
        tf.ones_like(inputs, dtype=tf.float32),
        tf.zeros_like(inputs, dtype=tf.float32))

  def _get_scale(self, inputs=None, reduce_axes=None, msqe_weight=None):
    """Returns power-of-2 scaling factors for quantization.

    Args:
      inputs: A tensor to be used for MSQE rounding. Note: ceil and round
        functions do not use the inputs.
      reduce_axes: A list of axes to be summed (averaged) over.
      msqe_weight: A tensor which is used in scale optimization to weight the
        MSQE (Mean Squared Quantization Error) of individual input elements. Its
        shape is same as 'inputs' and its dtype is `float32`.

    Returns:
      A tensor of power-of-2 scaling factors.
    """
    if self.use_po2_scale_ceil:
      scale_exponent = tf.math.ceil(self.scale_exponent)
    else:
      scale_exponent = tf.math.rint(self.scale_exponent)

    # MSQE rounding requires the inputs to optimize the scale exponent.
    if self.use_po2_scale_msqe_round and inputs is not None:
      scale_exponent_msqe = self.msqe_round(
          inputs=inputs,
          scale_exponent=self.scale_exponent,
          reduce_axes=reduce_axes,
          msqe_weight=msqe_weight)

      # Control when to use MSQE rounding. Note: self.switch_to_msqe_round is
      # set to true by default.
      scale_exponent = tf.cond(self.switch_to_msqe_round,
                               lambda: scale_exponent_msqe,
                               lambda: scale_exponent)

    # Apply STE
    scale_exponent = self.scale_exponent + tf.stop_gradient(scale_exponent -
                                                            self.scale_exponent)
    return tf.math.pow(2.0, scale_exponent)

  def msqe_round(self,
                 inputs,
                 scale_exponent,
                 reduce_axes=None,
                 msqe_weight=None):
    """Returns MSQE-wise optimum power-of-2 scale exponents.

    Args:
      inputs: A tensor, MSQE rounding is based on.
      scale_exponent: A tensor, learnable scale exponents which are not
        constrained in power-of-2.
      reduce_axes: A list of axes to be summed (averaged) over or None. If None,
        self.reduce_axes is used.
      msqe_weight: A tensor which is used to weight MSQE rounding or None. If
        None, a tensor (or None) from self._get_msqe_weight is used.

    Returns:
      A tensor of power-of-2 scale exponents.
    """
    if reduce_axes is None:
      reduce_axes = self.reduce_axes

    if msqe_weight is None:
      # Returned msqe_weight can be None.
      msqe_weight = self._get_msqe_weight(inputs)

    # floor
    scale_exponent_floor = tf.math.floor(scale_exponent)
    msqe_floor = self._calculate_msqe_inputs(
        inputs=inputs,
        scale=tf.math.pow(2.0, scale_exponent_floor),
        reduce_axes=reduce_axes,
        msqe_weight=msqe_weight)

    # ceil
    scale_exponent_ceil = tf.math.ceil(scale_exponent)
    msqe_ceil = self._calculate_msqe_inputs(
        inputs=inputs,
        scale=tf.math.pow(2.0, scale_exponent_ceil),
        reduce_axes=reduce_axes,
        msqe_weight=msqe_weight)

    return tf.where(msqe_floor < msqe_ceil, scale_exponent_floor,
                    scale_exponent_ceil)

  def get_config(self):
    config = {
        "bits": self.bits,
        "keep_negative": self.keep_negative,
        "scale_axis": self.scale_axis,
        "per_channel_scale": self.per_channel_scale,
        "init_scale": self.init_scale,
        "use_second_moments_msqe_opt": self.use_second_moments_msqe_opt,
        "second_moments_ema_decay": self.second_moments_ema_decay,
        "use_outlier_mask_msqe_weight": self.use_outlier_mask_msqe_weight,
        "use_sqrt_of_msqe_weight": self.use_sqrt_of_msqe_weight,
        "use_stable_scale_exponent": self.use_stable_scale_exponent,
        "stable_scale_ema_decay": self.stable_scale_ema_decay,
        "min_init_scale": self.min_init_scale,
        "use_po2_scale_ceil": self.use_po2_scale_ceil,
        "use_po2_scale_msqe_round": self.use_po2_scale_msqe_round,
        "qn": self.qn,
        "qp": self.qp,
        "scaled_axes": self.scaled_axes,
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


class quantized_bits_msqe_po2(BaseQuantizerPO2):  # pylint: disable=invalid-name
  """Quantizes the number to a number of bits by MSQE based scaling factors.
  For more details, see https://arxiv.org/abs/2210.03671.

  Attributes:
    bits: Integer, number of bits to perform quantization.
    keep_negative: Boolean, if true, it keeps negative values and sets the
      quantization levels symmetrically around 0. If false, negative numbers is
      clipped to 0.
    scale_axis: Integer, which axis to calculate scale from.
    per_channel_scale: Boolean, whether to perform per-channel (true) or
      per-layer (false) quantization.
    init_scale: Float or None, initial scale factor to initialize the scale with
      (if None, it will be initialized based on the first inputs.).
    use_second_moments_msqe_opt: Bool, whether to use the second moments based
      MSQE optimization or not.
    second_moments_ema_decay: Float, EMA decay factor for the second moments
      update.
    use_sqrt_of_msqe_weight: Bool, whether to use square root of MSQE weight.
    use_outlier_mask_msqe_weight: Bool, whether to apply outlier mask.
    use_stable_scale_exponent: Bool, whether to use exponentially moving
      averaged ("stable") scale exponent or not. Note: there is a tf.Variable
        (self.switch_to_stable_scale) that controls when to apply the stable
        scale exponent (i.e., if use_stable_scale_exponent is true and
        self.switch_to_stable_scale is false, the stable scale exponent is
        updated but not used.).
    stable_scale_ema_decay: Float, EMA decay factor for the stable scale update.
    outlier_mask_sigma: Float, sigma to apply for the outlier masking threshold.
    num_lls_iters: An integer. Number of linear least squares regression
      iterations.
    should_line_search: A bool. Whether to perform a line search.
    line_search_range: An integer. Search range of the line search.
  """

  def __init__(self,
               bits=4,
               keep_negative=True,
               scale_axis=None,
               per_channel_scale=False,
               init_scale=None,
               use_second_moments_msqe_opt=True,
               second_moments_ema_decay=0.999,
               use_sqrt_of_msqe_weight=True,
               use_outlier_mask_msqe_weight=True,
               use_stable_scale_exponent=False,
               stable_scale_ema_decay=0.99,
               outlier_mask_sigma=2.0,
               num_lls_iters=3,
               should_line_search=True,
               line_search_range=6,
               **kwargs):

    self.outlier_mask_sigma = outlier_mask_sigma
    self.num_lls_iters = num_lls_iters
    self.should_line_search = should_line_search
    self.line_search_range = line_search_range

    super().__init__(
        bits=bits,
        keep_negative=keep_negative,
        scale_axis=scale_axis,
        per_channel_scale=per_channel_scale,
        init_scale=init_scale,
        use_second_moments_msqe_opt=use_second_moments_msqe_opt,
        second_moments_ema_decay=second_moments_ema_decay,
        use_sqrt_of_msqe_weight=use_sqrt_of_msqe_weight,
        use_outlier_mask_msqe_weight=use_outlier_mask_msqe_weight,
        use_stable_scale_exponent=use_stable_scale_exponent,
        stable_scale_ema_decay=stable_scale_ema_decay,
        is_gradient_based=False,
        **kwargs)

  def __str__(self):
    # Convert Tensors to printable strings by converting to a numpy array and
    # then using regex to remove brackets when there is only one integer bit.
    ptn, repl = r"\[(\d)\]", r"\g<1>"
    bits = re.sub(
        ptn, repl,
        str(self.bits.numpy() if isinstance(self.bits, tf.Variable) else self
            .bits))

    flags = []
    flags.append("bits=" + str(int(bits)))
    flags.append("keep_negative=" + str(self.keep_negative))
    flags.append("scale_axis=" + str(self.scale_axis))
    flags.append("per_channel_scale=" + str(self.per_channel_scale))
    flags.append("init_scale=" + str(self.init_scale))
    flags.append("use_second_moments_msqe_opt=" +
                 str(self.use_second_moments_msqe_opt))
    flags.append("second_moments_ema_decay=" +
                 str(self.second_moments_ema_decay))
    flags.append("use_sqrt_of_msqe_weight=" + str(self.use_sqrt_of_msqe_weight))
    flags.append("use_outlier_mask_msqe_weight=" +
                 str(self.use_outlier_mask_msqe_weight))
    flags.append("use_stable_scale_exponent=" +
                 str(self.use_stable_scale_exponent))
    flags.append("stable_scale_ema_decay=" + str(self.stable_scale_ema_decay))
    flags.append("outlier_mask_sigma=" + str(self.outlier_mask_sigma))
    flags.append("num_lls_iters=" + str(self.num_lls_iters))
    flags.append("should_line_search=" + str(self.should_line_search))
    flags.append("line_search_range=" + str(self.line_search_range))
    return "quantized_bits_msqe_po2(" + ",".join(flags) + ")"

  def _get_init_scale_exponent(self, inputs):
    """Returns min and max of the inputs based initial scale exponent values.

    Args:
      inputs: A tensor to be used to calculate initial scale exponent values.

    Returns:
      A tensor of initial scale exponent values.
    """
    scale = K.max(
        abs(inputs), axis=self.scaled_axes, keepdims=True) / tf.cast(
            self.qp, dtype=tf.float32)
    return self._get_po2_scale_exponent(scale)

  def _get_outlier_mask(self, inputs):
    """Returns a tensor to mask outliers in the input for MSQE optimizations.

    The outlier threshold is based on the inputs distribution.

    Args:
      inputs: A tensor to be used to generate the outlier mask.

    Returns:
      A tensor to mask out the outliers of the inputs. Its shape is same as
      'inputs' and its dtype is `float32`.
    """
    std = tf.math.reduce_std(inputs, axis=self.reduce_axes, keepdims=True)
    outlier_threshold = self.outlier_mask_sigma * std
    return tf.where(
        abs(inputs) <= outlier_threshold, tf.ones_like(inputs),
        tf.zeros_like(inputs))

  def _get_scale(self, inputs=None, reduce_axes=None, msqe_weight=None):
    """Returns power-of-2 scaling factors for quantization.

    Args:
      inputs: A tensor to be used to optimize the scale value.
      reduce_axes: A list of axes to be summed (averaged) over.
      msqe_weight: A tensor which is used in scale optimization to weight the
        MSQE (Mean Squared Quantization Error) of individual input elements. Its
        shape is same as 'inputs' and its dtype is `float32`.

    Returns:
      A tensor of power-of-2 scaling factors. Its shape is same as
      'self.scale_exponent' and its dtype is `float32`.
    """
    if inputs is None:
      return self._get_po2_scale(self.scale)

    if reduce_axes is None:
      reduce_axes = self.reduce_axes

    if msqe_weight is None:
      msqe_weight = self._get_msqe_weight(inputs)

    scale, _ = self._optimize_msqe_scale(
        inputs,
        tf.math.pow(2.0, tf.round(self.scale_exponent)),
        reduce_axes=reduce_axes,
        msqe_weight=msqe_weight,
        num_lls_iters=self.num_lls_iters,
        should_line_search=self.should_line_search,
        line_search_range=self.line_search_range,
    )
    self.scale_exponent.assign(self._get_po2_scale_exponent(scale))
    return scale

  def get_config(self):
    config = {
        "bits": self.bits,
        "keep_negative": self.keep_negative,
        "scale_axis": self.scale_axis,
        "per_channel_scale": self.per_channel_scale,
        "init_scale": self.init_scale,
        "use_second_moments_msqe_opt": self.use_second_moments_msqe_opt,
        "second_moments_ema_decay": self.second_moments_ema_decay,
        "use_sqrt_of_msqe_weight": self.use_sqrt_of_msqe_weight,
        "use_outlier_mask_msqe_weight": self.use_outlier_mask_msqe_weight,
        "use_stable_scale_exponent": self.use_stable_scale_exponent,
        "stable_scale_ema_decay": self.stable_scale_ema_decay,
        "outlier_mask_sigma": self.outlier_mask_sigma,
        "num_lls_iters": self.num_lls_iters,
        "should_line_search": self.should_line_search,
        "line_search_range": self.line_search_range,
        "qn": self.qn,
        "qp": self.qp,
        "scaled_axes": self.scaled_axes,
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))
