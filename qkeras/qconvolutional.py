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
import warnings
import tensorflow as tf
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import SeparableConv1D
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import InputSpec
from tensorflow.python.eager import context
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import array_ops
from .qlayers import get_auto_range_constraint_initializer
from .qlayers import QActivation
from .quantizers import get_quantized_initializer
from .quantizers import get_quantizer

from tensorflow_model_optimization.python.core.sparsity.keras.prunable_layer import PrunableLayer


class QConv1D(Conv1D, PrunableLayer):
  """1D convolution layer (e.g. spatial convolution over images)."""

  # most of these parameters follow the implementation of Conv1D in Keras,
  # with the exception of kernel_range, bias_range, kernel_quantizer
  # and bias_quantizer, and kernel_initializer.
  #
  # kernel_quantizer: quantizer function/class for kernel
  # bias_quantizer: quantizer function/class for bias
  # kernel_range/bias_ranger: for quantizer functions whose values
  #   can go over [-1,+1], these values are used to set the clipping
  #   value of kernels and biases, respectively, instead of using the
  #   constraints specified by the user.
  #
  # we refer the reader to the documentation of Conv1D in Keras for the
  # other parameters.
  #

  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding="valid",
               dilation_rate=1,
               activation=None,
               use_bias=True,
               kernel_initializer="he_normal",
               bias_initializer="zeros",
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               kernel_quantizer=None,
               bias_quantizer=None,
               kernel_range=None,
               bias_range=None,
               **kwargs):

    if kernel_range is not None:
      warnings.warn("kernel_range is deprecated in QConv1D layer.")

    if bias_range is not None:
      warnings.warn("bias_range is deprecated in QConv1D layer.")

    self.kernel_range = kernel_range
    self.bias_range = bias_range

    self.kernel_quantizer = kernel_quantizer
    self.bias_quantizer = bias_quantizer

    self.kernel_quantizer_internal = get_quantizer(self.kernel_quantizer)
    self.bias_quantizer_internal = get_quantizer(self.bias_quantizer)

    # optimize parameter set to "auto" scaling mode if possible
    if hasattr(self.kernel_quantizer_internal, "_set_trainable_parameter"):
      self.kernel_quantizer_internal._set_trainable_parameter()

    self.quantizers = [
        self.kernel_quantizer_internal, self.bias_quantizer_internal
    ]

    kernel_constraint, kernel_initializer = (
        get_auto_range_constraint_initializer(self.kernel_quantizer_internal,
                                              kernel_constraint,
                                              kernel_initializer))

    if use_bias:
      bias_constraint, bias_initializer = (
          get_auto_range_constraint_initializer(self.bias_quantizer_internal,
                                                bias_constraint,
                                                bias_initializer))
    if activation is not None:
      activation = get_quantizer(activation)

    super(QConv1D, self).__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        **kwargs)

  def call(self, inputs):
    if self.kernel_quantizer:
      quantized_kernel = self.kernel_quantizer_internal(self.kernel)
    else:
      quantized_kernel = self.kernel

    outputs = tf.keras.backend.conv1d(
        inputs,
        quantized_kernel,
        strides=self.strides[0],
        padding=self.padding,
        data_format=self.data_format,
        dilation_rate=self.dilation_rate[0])

    if self.use_bias:
      if self.bias_quantizer:
        quantized_bias = self.bias_quantizer_internal(self.bias)
      else:
        quantized_bias = self.bias

      outputs = tf.keras.backend.bias_add(
          outputs, quantized_bias, data_format=self.data_format)

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def get_config(self):
    config = {
        "kernel_quantizer":
            constraints.serialize(self.kernel_quantizer_internal),
        "bias_quantizer":
            constraints.serialize(self.bias_quantizer_internal),
        "kernel_range": self.kernel_range,
        "bias_range": self.bias_range
    }
    base_config = super(QConv1D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_quantization_config(self):
    return {
        "kernel_quantizer":
            str(self.kernel_quantizer_internal),
        "bias_quantizer":
            str(self.bias_quantizer_internal),
        "activation":
            str(self.activation),
        "filters" : str(self.filters)
    }

  def get_quantizers(self):
    return self.quantizers

  def get_prunable_weights(self):
    return [self.kernel]


class QConv2D(Conv2D, PrunableLayer):
  """2D convolution layer (e.g. spatial convolution over images)."""

  # most of these parameters follow the implementation of Conv2D in Keras,
  # with the exception of kernel_range, bias_range, kernel_quantizer
  # and bias_quantizer, and kernel_initializer.
  #
  # kernel_quantizer: quantizer function/class for kernel
  # bias_quantizer: quantizer function/class for bias
  # kernel_range/bias_ranger: for quantizer functions whose values
  #   can go over [-1,+1], these values are used to set the clipping
  #   value of kernels and biases, respectively, instead of using the
  #   constraints specified by the user.
  #
  # we refer the reader to the documentation of Conv2D in Keras for the
  # other parameters.
  #

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding="valid",
               data_format="channels_last",
               dilation_rate=(1, 1),
               activation=None,
               use_bias=True,
               kernel_initializer="he_normal",
               bias_initializer="zeros",
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               kernel_range=None,
               bias_range=None,
               kernel_quantizer=None,
               bias_quantizer=None,
               **kwargs):

    if kernel_range is not None:
      warnings.warn("kernel_range is deprecated in QConv2D layer.")

    if bias_range is not None:
      warnings.warn("bias_range is deprecated in QConv2D layer.")

    self.kernel_range = kernel_range
    self.bias_range = bias_range

    self.kernel_quantizer = kernel_quantizer
    self.bias_quantizer = bias_quantizer

    self.kernel_quantizer_internal = get_quantizer(self.kernel_quantizer)
    self.bias_quantizer_internal = get_quantizer(self.bias_quantizer)

    # optimize parameter set to "auto" scaling mode if possible
    if hasattr(self.kernel_quantizer_internal, "_set_trainable_parameter"):
      self.kernel_quantizer_internal._set_trainable_parameter()

    self.quantizers = [
        self.kernel_quantizer_internal, self.bias_quantizer_internal
    ]

    kernel_constraint, kernel_initializer = (
        get_auto_range_constraint_initializer(self.kernel_quantizer_internal,
                                              kernel_constraint,
                                              kernel_initializer))

    if use_bias:
      bias_constraint, bias_initializer = (
          get_auto_range_constraint_initializer(self.bias_quantizer_internal,
                                                bias_constraint,
                                                bias_initializer))

    if activation is not None:
      activation = get_quantizer(activation)

    super(QConv2D, self).__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        **kwargs)

  def call(self, inputs):
    if self.kernel_quantizer:
      quantized_kernel = self.kernel_quantizer_internal(self.kernel)
    else:
      quantized_kernel = self.kernel

    outputs = tf.keras.backend.conv2d(
        inputs,
        quantized_kernel,
        strides=self.strides,
        padding=self.padding,
        data_format=self.data_format,
        dilation_rate=self.dilation_rate)

    if self.use_bias:
      if self.bias_quantizer:
        quantized_bias = self.bias_quantizer_internal(self.bias)
      else:
        quantized_bias = self.bias

      outputs = tf.keras.backend.bias_add(
          outputs, quantized_bias, data_format=self.data_format)

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def get_config(self):
    config = {
        "kernel_quantizer":
            constraints.serialize(self.kernel_quantizer_internal),
        "bias_quantizer":
            constraints.serialize(self.bias_quantizer_internal),
        "kernel_range": self.kernel_range,
        "bias_range": self.bias_range
    }
    base_config = super(QConv2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_quantization_config(self):
    return {
        "kernel_quantizer":
            str(self.kernel_quantizer_internal),
        "bias_quantizer":
            str(self.bias_quantizer_internal),
        "activation":
            str(self.activation),
        "filters" : str(self.filters)
    }

  def get_quantizers(self):
    return self.quantizers

  def get_prunable_weights(self):
    return [self.kernel]


class QConv2DTranspose(Conv2DTranspose, PrunableLayer):
  """2D convolution layer (e.g. spatial convolution over images)."""

  # most of these parameters follow the implementation of Conv2DTranspose
  # in Keras, with the exception of kernel_quantizer and bias_quantizer
  # and kernel_initializer.
  #
  # kernel_quantizer: quantizer function/class for kernel
  # bias_quantizer: quantizer function/class for bias
  #
  # we refer the reader to the documentation of Conv2DTranspose in Keras for
  # the other parameters.
  #

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               output_padding=None,
               data_format=None,
               dilation_rate=(1, 1),
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               kernel_quantizer=None,
               bias_quantizer=None,
               **kwargs):

    self.kernel_quantizer = kernel_quantizer
    self.bias_quantizer = bias_quantizer

    self.kernel_quantizer_internal = get_quantizer(self.kernel_quantizer)
    self.bias_quantizer_internal = get_quantizer(self.bias_quantizer)

    # optimize parameter set to "auto" scaling mode if possible
    if hasattr(self.kernel_quantizer_internal, "_set_trainable_parameter"):
      self.kernel_quantizer_internal._set_trainable_parameter()

    self.quantizers = [
        self.kernel_quantizer_internal, self.bias_quantizer_internal
    ]

    kernel_constraint, kernel_initializer = (
        get_auto_range_constraint_initializer(self.kernel_quantizer_internal,
                                              kernel_constraint,
                                              kernel_initializer))

    if use_bias:
      bias_constraint, bias_initializer = (
          get_auto_range_constraint_initializer(self.bias_quantizer_internal,
                                                bias_constraint,
                                                bias_initializer))

    if activation is not None:
      activation = get_quantizer(activation)

    super(QConv2DTranspose, self).__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        output_padding=None,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        **kwargs)

  def call(self, inputs):
    inputs_shape = array_ops.shape(inputs)
    batch_size = inputs_shape[0]
    if self.data_format == 'channels_first':
      h_axis, w_axis = 2, 3
    else:
      h_axis, w_axis = 1, 2

    height, width = inputs_shape[h_axis], inputs_shape[w_axis]
    kernel_h, kernel_w = self.kernel_size
    stride_h, stride_w = self.strides

    if self.output_padding is None:
      out_pad_h = out_pad_w = None
    else:
      out_pad_h, out_pad_w = self.output_padding

    # Infer the dynamic output shape:
    out_height = conv_utils.deconv_output_length(height,
                                                 kernel_h,
                                                 padding=self.padding,
                                                 output_padding=out_pad_h,
                                                 stride=stride_h,
                                                 dilation=self.dilation_rate[0])
    out_width = conv_utils.deconv_output_length(width,
                                                kernel_w,
                                                padding=self.padding,
                                                output_padding=out_pad_w,
                                                stride=stride_w,
                                                dilation=self.dilation_rate[1])
    if self.data_format == 'channels_first':
      output_shape = (batch_size, self.filters, out_height, out_width)
    else:
      output_shape = (batch_size, out_height, out_width, self.filters)

    if self.kernel_quantizer:
      quantized_kernel = self.kernel_quantizer_internal(self.kernel)
    else:
      quantized_kernel = self.kernel

    output_shape_tensor = array_ops.stack(output_shape)
    outputs = tf.keras.backend.conv2d_transpose(
        inputs,
        quantized_kernel,
        output_shape_tensor,
        strides=self.strides,
        padding=self.padding,
        data_format=self.data_format,
        dilation_rate=self.dilation_rate)

    if not context.executing_eagerly():
      # Infer the static output shape:
      out_shape = self.compute_output_shape(inputs.shape)
      outputs.set_shape(out_shape)

    if self.use_bias:
      if self.bias_quantizer:
        quantized_bias = self.bias_quantizer_internal(self.bias)
      else:
        quantized_bias = self.bias

      outputs = tf.keras.backend.bias_add(
          outputs,
          quantized_bias,
          data_format=self.data_format)

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def get_config(self):
    config = {
        "kernel_quantizer":
            constraints.serialize(self.kernel_quantizer_internal),
        "bias_quantizer":
            constraints.serialize(self.bias_quantizer_internal)
    }
    base_config = super(QConv2DTranspose, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_quantizers(self):
    return self.quantizers

  def get_prunable_weights(self):
    return [self.kernel]


class QSeparableConv1D(SeparableConv1D, PrunableLayer):
  """Depthwise separable 1D convolution."""

  # most of these parameters follow the implementation of SeparableConv1D
  # in Keras, with the exception of depthwise_quantizer, pointwise_quantizer
  # and bias_quantizer.
  #
  # depthwise_quantizer: quantizer function/class for depthwise spatial kernel
  # pointwise_quantizer: quantizer function/class for pointwise kernel
  # bias_quantizer: quantizer function/class for bias
  #
  # we refer the reader to the documentation of SeparableConv1D in Keras for
  # the other parameters.
  #

  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format=None,
               dilation_rate=1,
               depth_multiplier=1,
               activation=None,
               use_bias=True,
               depthwise_initializer='glorot_uniform',
               pointwise_initializer='glorot_uniform',
               bias_initializer='zeros',
               depthwise_regularizer=None,
               pointwise_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               depthwise_constraint=None,
               pointwise_constraint=None,
               bias_constraint=None,
               depthwise_quantizer=None,
               pointwise_quantizer=None,
               bias_quantizer=None,
               **kwargs):

    self.depthwise_quantizer = depthwise_quantizer
    self.pointwise_quantizer = pointwise_quantizer
    self.bias_quantizer = bias_quantizer

    self.depthwise_quantizer_internal = get_quantizer(self.depthwise_quantizer)
    self.pointwise_quantizer_internal = get_quantizer(self.pointwise_quantizer)
    self.bias_quantizer_internal = get_quantizer(self.bias_quantizer)

    # optimize parameter set to "auto" scaling mode if possible
    if hasattr(self.depthwise_quantizer_internal, "_set_trainable_parameter"):
      self.depthwise_quantizer_internal._set_trainable_parameter()

    if hasattr(self.pointwise_quantizer_internal, "_set_trainable_parameter"):
      self.pointwise_quantizer_internal._set_trainable_parameter()

    self.quantizers = [
        self.depthwise_quantizer_internal, self.pointwise_quantizer_internal,
        self.bias_quantizer_internal
    ]

    depthwise_constraint, depthwise_initializer = (
        get_auto_range_constraint_initializer(self.depthwise_quantizer_internal,
                                              depthwise_constraint,
                                              depthwise_initializer))

    pointwise_constraint, pointwise_initializer = (
        get_auto_range_constraint_initializer(self.pointwise_quantizer_internal,
                                              pointwise_constraint,
                                              pointwise_initializer))

    if use_bias:
      bias_constraint, bias_initializer = (
          get_auto_range_constraint_initializer(self.bias_quantizer_internal,
                                                bias_constraint,
                                                bias_initializer))

    if activation is not None:
      activation = get_quantizer(activation)

    super(QSeparableConv1D, self).__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        depth_multiplier=depth_multiplier,
        activation=activation,
        use_bias=use_bias,
        depthwise_initializer=initializers.get(depthwise_initializer),
        pointwise_initializer=initializers.get(pointwise_initializer),
        bias_initializer=initializers.get(bias_initializer),
        depthwise_regularizer=regularizers.get(depthwise_regularizer),
        pointwise_regularizer=regularizers.get(pointwise_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        depthwise_constraint=constraints.get(depthwise_constraint),
        pointwise_constraint=constraints.get(pointwise_constraint),
        bias_constraint=constraints.get(bias_constraint),
        **kwargs)

  def call(self, inputs):
    if self.padding == 'causal':
      inputs = array_ops.pad(inputs, self._compute_causal_padding())
    
    spatial_start_dim = 1 if self.data_format == 'channels_last' else 2

    # Explicitly broadcast inputs and kernels to 4D.
    inputs = array_ops.expand_dims(inputs, spatial_start_dim)
    depthwise_kernel = array_ops.expand_dims(self.depthwise_kernel, 0)
    pointwise_kernel = array_ops.expand_dims(self.pointwise_kernel, 0)
    dilation_rate = (1,) + self.dilation_rate

    if self.padding == 'causal':
      op_padding = 'valid'
    else:
      op_padding = self.padding

    if self.depthwise_quantizer:
      quantized_depthwise_kernel = self.depthwise_quantizer_internal(
          depthwise_kernel)
    else:
      quantized_depthwise_kernel = depthwise_kernel

    if self.pointwise_quantizer:
      quantized_pointwise_kernel = self.pointwise_quantizer_internal(
          pointwise_kernel)
    else:
      quantized_pointwise_kernel = pointwise_kernel

    outputs = tf.keras.backend.separable_conv2d(
        inputs,
        quantized_depthwise_kernel,
        quantized_pointwise_kernel,
        strides=self.strides * 2,
        padding=op_padding,
        dilation_rate=dilation_rate,
        data_format=self.data_format)

    if self.use_bias:
      if self.bias_quantizer:
        quantized_bias = self.bias_quantizer_internal(self.bias)
      else:
        quantized_bias = self.bias

      outputs = tf.keras.backend.bias_add(
          outputs,
          quantized_bias,
          data_format=self.data_format)

    outputs = array_ops.squeeze(outputs, [spatial_start_dim])

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def get_config(self):
    config = {
        "depthwise_quantizer":
            constraints.serialize(self.depthwise_quantizer_internal),
        "pointwise_quantizer":
            constraints.serialize(self.pointwise_quantizer_internal),
        "bias_quantizer":
            constraints.serialize(self.bias_quantizer_internal)
    }
    base_config = super(QSeparableConv1D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_quantizers(self):
    return self.quantizers

  def get_prunable_weights(self):
    return [self.depthwise_kernel, self.pointwise_kernel]


class QSeparableConv2D(SeparableConv2D, PrunableLayer):
  """Depthwise separable 2D convolution."""

  # most of these parameters follow the implementation of SeparableConv2D
  # in Keras, with the exception of depthwise_quantizer, pointwise_quantizer
  # and bias_quantizer.
  #
  # depthwise_quantizer: quantizer function/class for depthwise spatial kernel
  # pointwise_quantizer: quantizer function/class for pointwise kernel
  # bias_quantizer: quantizer function/class for bias
  #
  # we refer the reader to the documentation of SeparableConv2D in Keras for
  # the other parameters.
  #

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               depth_multiplier=1,
               activation=None,
               use_bias=True,
               depthwise_initializer='glorot_uniform',
               pointwise_initializer='glorot_uniform',
               bias_initializer='zeros',
               depthwise_regularizer=None,
               pointwise_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               depthwise_constraint=None,
               pointwise_constraint=None,
               bias_constraint=None,
               depthwise_quantizer=None,
               pointwise_quantizer=None,
               bias_quantizer=None,
               **kwargs):

    self.depthwise_quantizer = depthwise_quantizer
    self.pointwise_quantizer = pointwise_quantizer
    self.bias_quantizer = bias_quantizer

    self.depthwise_quantizer_internal = get_quantizer(self.depthwise_quantizer)
    self.pointwise_quantizer_internal = get_quantizer(self.pointwise_quantizer)
    self.bias_quantizer_internal = get_quantizer(self.bias_quantizer)

    # optimize parameter set to "auto" scaling mode if possible
    if hasattr(self.depthwise_quantizer_internal, "_set_trainable_parameter"):
      self.depthwise_quantizer_internal._set_trainable_parameter()

    if hasattr(self.pointwise_quantizer_internal, "_set_trainable_parameter"):
      self.pointwise_quantizer_internal._set_trainable_parameter()

    self.quantizers = [
        self.depthwise_quantizer_internal, self.pointwise_quantizer_internal,
        self.bias_quantizer_internal
    ]

    depthwise_constraint, depthwise_initializer = (
        get_auto_range_constraint_initializer(self.depthwise_quantizer_internal,
                                              depthwise_constraint,
                                              depthwise_initializer))

    pointwise_constraint, pointwise_initializer = (
        get_auto_range_constraint_initializer(self.pointwise_quantizer_internal,
                                              pointwise_constraint,
                                              pointwise_initializer))

    if use_bias:
      bias_constraint, bias_initializer = (
          get_auto_range_constraint_initializer(self.bias_quantizer_internal,
                                                bias_constraint,
                                                bias_initializer))

    if activation is not None:
      activation = get_quantizer(activation)

    super(QSeparableConv2D, self).__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        depth_multiplier=depth_multiplier,
        activation=activation,
        use_bias=use_bias,
        depthwise_initializer=initializers.get(depthwise_initializer),
        pointwise_initializer=initializers.get(pointwise_initializer),
        bias_initializer=initializers.get(bias_initializer),
        depthwise_regularizer=regularizers.get(depthwise_regularizer),
        pointwise_regularizer=regularizers.get(pointwise_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        depthwise_constraint=constraints.get(depthwise_constraint),
        pointwise_constraint=constraints.get(pointwise_constraint),
        bias_constraint=constraints.get(bias_constraint),
        **kwargs)

  def call(self, inputs):
    # Apply the actual ops.
    if self.depthwise_quantizer:
      quantized_depthwise_kernel = self.depthwise_quantizer_internal(
          self.depthwise_kernel)
    else:
      quantized_depthwise_kernel = self.depthwise_kernel

    if self.pointwise_quantizer:
      quantized_pointwise_kernel = self.pointwise_quantizer_internal(
          self.pointwise_kernel)
    else:
      quantized_pointwise_kernel = self.pointwise_kernel

    outputs = tf.keras.backend.separable_conv2d(
        inputs,
        quantized_depthwise_kernel,
        quantized_pointwise_kernel,
        strides=self.strides,
        padding=self.padding,
        dilation_rate=self.dilation_rate,
        data_format=self.data_format)

    if self.use_bias:
      if self.bias_quantizer:
        quantized_bias = self.bias_quantizer_internal(self.bias)
      else:
        quantized_bias = self.bias

      outputs = tf.keras.backend.bias_add(
          outputs,
          quantized_bias,
          data_format=self.data_format)

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def get_config(self):
    config = {
        "depthwise_quantizer":
            constraints.serialize(self.depthwise_quantizer_internal),
        "pointwise_quantizer":
            constraints.serialize(self.pointwise_quantizer_internal),
        "bias_quantizer":
            constraints.serialize(self.bias_quantizer_internal)
    }
    base_config = super(QSeparableConv2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_quantizers(self):
    return self.quantizers

  def get_prunable_weights(self):
    return [self.depthwise_kernel, self.pointwise_kernel]


class QDepthwiseConv2D(DepthwiseConv2D, PrunableLayer):
  """Creates quantized depthwise conv2d. Copied from mobilenet."""

  # most of these parameters follow the implementation of DepthwiseConv2D
  # in Keras, # with the exception of depthwise_range, bias_range,
  # depthwise_quantizer # and bias_quantizer, and kernel_initializer.
  #
  # depthwise_quantizer: quantizer function/class for kernel
  # bias_quantizer: quantizer function/class for bias
  # depthwise_range/bias_ranger: for quantizer functions whose values
  #   can go over [-1,+1], these values are used to set the clipping
  #   value of kernels and biases, respectively, instead of using the
  #   constraints specified by the user.
  #
  # we refer the reader to the documentation of DepthwiseConv2D in Keras for the
  # other parameters.
  #

  def __init__(self,
               kernel_size,
               strides=(1, 1),
               padding="VALID",
               depth_multiplier=1,
               data_format=None,
               activation=None,
               use_bias=True,
               depthwise_initializer="he_normal",
               bias_initializer="zeros",
               depthwise_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               depthwise_constraint=None,
               bias_constraint=None,
               dilation_rate=(1, 1),
               depthwise_quantizer=None,
               bias_quantizer=None,
               depthwise_range=None,
               bias_range=None,
               **kwargs):

    if depthwise_range is not None:
      warnings.warn("depthwise_range is deprecated in QDepthwiseConv2D layer.")

    if bias_range is not None:
      warnings.warn("bias_range is deprecated in QDepthwiseConv2D layer.")

    self.depthwise_range = depthwise_range
    self.bias_range = bias_range

    self.depthwise_quantizer = depthwise_quantizer
    self.bias_quantizer = bias_quantizer

    self.depthwise_quantizer_internal = get_quantizer(self.depthwise_quantizer)
    self.bias_quantizer_internal = get_quantizer(self.bias_quantizer)

    # optimize parameter set to "auto" scaling mode if possible
    if hasattr(self.depthwise_quantizer_internal, "_set_trainable_parameter"):
      self.depthwise_quantizer_internal._set_trainable_parameter()

    self.quantizers = [
        self.depthwise_quantizer_internal, self.bias_quantizer_internal
    ]

    depthwise_constraint, depthwise_initializer = (
        get_auto_range_constraint_initializer(self.depthwise_quantizer_internal,
                                              depthwise_constraint,
                                              depthwise_initializer))

    if use_bias:
      bias_constraint, bias_initializer = (
          get_auto_range_constraint_initializer(self.bias_quantizer_internal,
                                                bias_constraint,
                                                bias_initializer))
    if activation is not None:
      activation = get_quantizer(activation)

    super(QDepthwiseConv2D, self).__init__(
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        activation=activation,
        use_bias=use_bias,
        depthwise_regularizer=depthwise_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        depth_multiplier=depth_multiplier,
        depthwise_initializer=depthwise_initializer,
        bias_initializer=bias_initializer,
        depthwise_constraint=depthwise_constraint,
        bias_constraint=bias_constraint,
        dilation_rate=dilation_rate,
        **kwargs)

  def build(self, input_shape):
    if len(input_shape) < 4:
      raise ValueError(
          "Inputs to `QDepthwiseConv2D` should have rank 4. "
          "Received input shape:", str(input_shape))
    if self.data_format == "channels_first":
      channel_axis = 1
    else:
      channel_axis = 3
    if input_shape[channel_axis] is None:
      raise ValueError("The channel dimension of the inputs to "
                       "`QDepthwiseConv2D` "
                       "should be defined. Found `None`.")
    input_dim = int(input_shape[channel_axis])
    depthwise_kernel_shape = (self.kernel_size[0], self.kernel_size[1],
                              input_dim, self.depth_multiplier)

    self.depthwise_kernel = self.add_weight(
        shape=depthwise_kernel_shape,
        initializer=self.depthwise_initializer,
        name="depthwise_kernel",
        regularizer=self.depthwise_regularizer,
        constraint=self.depthwise_constraint)

    if self.use_bias:
      self.bias = self.add_weight(
          shape=(input_dim * self.depth_multiplier,),
          initializer=self.bias_initializer,
          name="bias",
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint)
    else:
      self.bias = None
    # Set input spec.
    self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
    self.built = True

  def call(self, inputs, training=None):
    if self.depthwise_quantizer:
      quantized_depthwise_kernel = (
          self.depthwise_quantizer_internal(self.depthwise_kernel))
    else:
      quantized_depthwise_kernel = self.depthwise_kernel
    outputs = tf.keras.backend.depthwise_conv2d(
        inputs,
        quantized_depthwise_kernel,
        strides=self.strides,
        padding=self.padding,
        dilation_rate=self.dilation_rate,
        data_format=self.data_format)

    if self.use_bias:
      if self.bias_quantizer:
        quantized_bias = self.bias_quantizer_internal(self.bias)
      else:
        quantized_bias = self.bias
      outputs = tf.keras.backend.bias_add(
          outputs, quantized_bias, data_format=self.data_format)

    if self.activation is not None:
      return self.activation(outputs)

    return outputs

  def get_config(self):
    config = super(QDepthwiseConv2D, self).get_config()
    config.pop("filters", None)
    config.pop("kernel_initializer", None)
    config.pop("kernel_regularizer", None)
    config.pop("kernel_constraint", None)
    config["depth_multiplier"] = self.depth_multiplier
    config["depthwise_initializer"] = initializers.serialize(
        self.depthwise_initializer)
    config["depthwise_regularizer"] = regularizers.serialize(
        self.depthwise_regularizer)
    config["depthwise_constraint"] = constraints.serialize(
        self.depthwise_constraint)
    config["depthwise_quantizer"] = constraints.serialize(
        self.depthwise_quantizer_internal)
    config["bias_quantizer"] = constraints.serialize(
        self.bias_quantizer_internal)
    config["depthwise_range"] = self.depthwise_range
    config["bias_range"] = self.bias_range
    return config

  def get_quantization_config(self):
    return {
        "depthwise_quantizer_internal":
            str(self.depthwise_quantizer_internal),
        "bias_quantizer":
            str(self.bias_quantizer_internal),
        "activation":
            str(self.activation),
        "filters" : str(self.filters)
    }

  def get_quantizers(self):
    return self.quantizers

  def get_prunable_weights(self):
    return [self.depthwise_kernel]


def QMobileNetSeparableConv2D(
    filters,  # pylint: disable=invalid-name
    kernel_size,
    strides=(1, 1),
    padding="VALID",
    dilation_rate=(1, 1),
    depth_multiplier=1,
    activation=None,
    use_bias=True,
    depthwise_initializer="he_normal",
    pointwise_initializer="he_normal",
    bias_initializer="zeros",
    depthwise_regularizer=None,
    pointwise_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    depthwise_constraint=None,
    pointwise_constraint=None,
    bias_constraint=None,
    depthwise_quantizer=None,
    pointwise_quantizer=None,
    bias_quantizer=None,
    depthwise_activation=None,
    depthwise_range=None,
    pointwise_range=None,
    bias_range=None,
    depthwise_dropout_rate=0.0,
    pw_first=False,
    name=""):
  """Adds a quantized separableconv2d."""

  # we use here a modified version that appeared in mobilenet that adds
  # quantization to the network, and possibly an intermediate activation
  # layer that acts as a quantizer and possible dropout layer between
  # the depthwise and pointwise convolutions.
  #
  # since this implementation expands into depthwise -> pointwise
  # convolutions, the users will not see a separable convolution operation
  # in model.summary(), but rather a depthwise convolution followed by a
  # pointwise convolution.
  #
  # depthwise_quantizer: depthwise quantization function
  # pointwise_quantizer: pointwise quantization function
  # bias_quantizer: bias quantization function for the pointwise convolution
  # depthwise_range/pointwise_range/bias_range: ranges to be used if
  # quantization values can become greater than -1 and +1.
  # depthwise_dropout_rate: dropout between depthwise and pointwise is added
  #   if rate > 0.0
  # pw_first: this may disappear in the future, but as deep quantized networks
  #   sometimes behave in different ways, if we are using binary or ternary
  #   quantization, it may be better to apply pointwise before depthwise.
  #
  # For the remaining parameters, please refer to Keras implementation of
  # SeparableConv2D.
  #

  def _call(inputs):  # pylint: disable=invalid-name
    """Internally builds qseparableconv2d."""

    x = inputs

    if pw_first:
      x = QConv2D(
          filters, (1, 1),
          strides=(1, 1),
          padding="same",
          use_bias=use_bias,
          kernel_constraint=pointwise_constraint,
          kernel_initializer=pointwise_initializer,
          kernel_regularizer=pointwise_regularizer,
          kernel_quantizer=pointwise_quantizer,
          bias_quantizer=bias_quantizer,
          bias_regularizer=bias_regularizer,
          bias_initializer=bias_initializer,
          bias_constraint=bias_constraint,
          activity_regularizer=activity_regularizer,
          kernel_range=pointwise_range,
          bias_range=bias_range,
          name=name + "_pw")(
              x)

      if depthwise_activation:
        if isinstance(depthwise_activation, QActivation):
          x = depthwise_activation(x)
        else:
          x = QActivation(depthwise_activation, name=name + "_dw_act")(x)

      if depthwise_dropout_rate > 0.0:
        x = Dropout(rate=depthwise_dropout_rate, name=name + "_dw_dropout")(x)

    x = QDepthwiseConv2D(
        kernel_size,
        strides=strides,
        dilation_rate=dilation_rate,
        padding=padding,
        depth_multiplier=depth_multiplier,
        use_bias=False,
        depthwise_regularizer=depthwise_regularizer,
        depthwise_initializer=depthwise_initializer,
        depthwise_constraint=depthwise_constraint,
        depthwise_quantizer=depthwise_quantizer,
        depthwise_range=depthwise_range,
        name=name + "_dw")(
            x)

    if not pw_first:
      if depthwise_activation:
        if isinstance(depthwise_activation, QActivation):
          x = depthwise_activation(x)
        else:
          x = QActivation(depthwise_activation, name=name + "_dw_act")(x)

      if depthwise_dropout_rate > 0.0:
        x = Dropout(rate=depthwise_dropout_rate, name=name + "_dw_dropout")(x)

      x = QConv2D(
          filters, (1, 1),
          strides=(1, 1),
          padding="same",
          use_bias=use_bias,
          kernel_constraint=pointwise_constraint,
          kernel_initializer=pointwise_initializer,
          kernel_regularizer=pointwise_regularizer,
          kernel_quantizer=pointwise_quantizer,
          bias_quantizer=bias_quantizer,
          bias_regularizer=bias_regularizer,
          bias_initializer=bias_initializer,
          bias_constraint=bias_constraint,
          activity_regularizer=activity_regularizer,
          kernel_range=pointwise_range,
          bias_range=bias_range,
          name=name + "_pw")(
              x)

    if activation:
      if isinstance(activation, QActivation):
        x = activation(x)
      else:
        x = Activation(activation, name=name + "_pw_act")(x)
    return x

  return _call
