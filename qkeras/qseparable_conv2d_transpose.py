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


import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import InputSpec
from tf_keras.utils import conv_utils

from .quantizers import get_quantizer
from tensorflow.python.eager import context
from tensorflow.python.keras import constraints
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops


class QSeparableConv2DTransposeTPU(Conv2DTranspose):
  """Quantized Separable Conv2DTranspose layer for TPU and GPU."""

  # Most of these parameters follow the implementation of Conv2DTranspose
  # in Keras, with the exception of following parameters.
  #
  # depthwise_activation: activation quantizer for depthwise convolution
  # pointwise_activation: activation quantizer for pointwise convolution
  # depthwise_kernel_quantizer: quantizer function/class for depthwise kernel
  # pointwise_kernel_quantizers: quantizer function/class for pointwise kernel
  # bias_quantizer: quantizer function/class for bias
  #
  # we refer the reader to the documentation of Conv2DTranspose in Keras for
  # the other parameters.

  # Important Notes:
  # This implementation requies the use of grouped convolution, which is only
  # supported in TPU/GPU, not in CPU.
  # When running in CPU, it gives the following error:
  # "Gradients for grouped convolutions are not supported on CPU.
  # Please file a feature request if you run into this issue."
  # For now we can train with this implmentation in TPU/GPU,
  # for inference in CPU, we will convert the layer to an equivalent
  # QSeparableConv2DTransposeCPU layer, which is slow in training,
  # but should suffice in inference.

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding="valid",
               output_padding=None,
               depth_multiplier=1,
               depthwise_activation=None,
               pointwise_activation=None,
               use_bias=True,
               depthwise_kernel_quantizer=None,
               pointwise_kernel_quantizer=None,
               bias_quantizer=None,
               **kwargs):

    self.filters = filters
    self.kernel_size = kernel_size
    self.strides = strides
    self.padding = padding
    self.output_padding = output_padding
    self.depth_multiplier = depth_multiplier
    self.depthwise_activation = depthwise_activation
    self.pointwise_activation = pointwise_activation
    self.use_bias = use_bias

    self.depthwise_kernel_quantizer = depthwise_kernel_quantizer
    self.pointwise_kernel_quantizer = pointwise_kernel_quantizer
    self.bias_quantizer = bias_quantizer

    self.depthwise_kernel_quantizer_internal = get_quantizer(
        self.depthwise_kernel_quantizer)
    self.pointwise_kernel_quantizer_internal = get_quantizer(
        self.pointwise_kernel_quantizer)
    self.bias_quantizer_internal = get_quantizer(self.bias_quantizer)

    # optimize parameter set to "auto" scaling mode if possible
    for q in [self.depthwise_kernel_quantizer_internal,
              self.pointwise_kernel_quantizer_internal]:
      if hasattr(q, "_set_trainable_parameter"):
        q._set_trainable_parameter()

    if depthwise_activation is not None:
      self.depthwise_activation = get_quantizer(depthwise_activation)

    if pointwise_activation is not None:
      self.pointwise_activation = get_quantizer(pointwise_activation)

    super().__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        **kwargs)

  def _get_input_axis(self):
    if self.data_format == "channels_first":
      b_axis, c_axis, h_axis, w_axis = 0, 1, 2, 3
    else:
      b_axis, c_axis, h_axis, w_axis = 0, 3, 1, 2

    return b_axis, c_axis, h_axis, w_axis

  def _get_input_dims(self, input_shape):
    b_axis, c_axis, h_axis, w_axis = self._get_input_axis()

    return (
        input_shape[b_axis], input_shape[c_axis],
        input_shape[h_axis], input_shape[w_axis])

  def _get_output_size(self, inputs, output_padding, padding, strides,
                       dilation_rate, kernel_weights):
    input_shape = array_ops.shape(inputs)
    batch_size, _, height, width = self._get_input_dims(input_shape)
    kernel_h, kernel_w = kernel_weights.shape[:2]
    stride_h, stride_w = strides

    dilation_h, dilation_w = dilation_rate[0], dilation_rate[1]

    if output_padding is None:
      out_pad_h = out_pad_w = None
    else:
      out_pad_h, out_pad_w = output_padding

    # Infer the dynamic output shape:
    out_height = conv_utils.deconv_output_length(
        height,
        kernel_h,
        padding=padding,
        output_padding=out_pad_h,
        stride=stride_h,
        dilation=dilation_h)

    out_width = conv_utils.deconv_output_length(
        width,
        kernel_w,
        padding=padding,
        output_padding=out_pad_w,
        stride=stride_w,
        dilation=dilation_w)

    return (batch_size, out_height, out_width, kernel_h, kernel_w)

  def build(self, input_shape):
    self._input_shape = input_shape

    _, input_channel, _, _ = self._get_input_dims(input_shape)
    channel_axis = self._get_input_axis()[1]

    self.input_spec = InputSpec(
        min_ndim=self.rank + 2, axes={channel_axis: input_channel}
    )
    # By enforcing the kernel shape, we can control how convolution is
    # done in depthwise or pointwise.
    # When setting kernel shape=(kw, kh, 1, input_channel), it does depthwise
    # convolution.
    depthwise_kernel_shape = self.kernel_size + (1, input_channel)

    self.depthwise_kernel = self.add_weight(
        name="depthwise_kernel",
        shape=depthwise_kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=True,
        dtype=self.dtype,
    )

    # When setting kernel shape=(1, 1, output_channel, input_channel), it does
    # pointwise convolution.
    pointwise_kernel_shape = (1, 1, self.filters, input_channel)
    self.pointwise_kernel = self.add_weight(
        name="pointwise_kernel",
        shape=pointwise_kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=True,
        dtype=self.dtype,
    )

    if self.use_bias:
      # This bias term is usally add at the end of the pointwise convolution.
      self.bias = self.add_weight(
          name="bias",
          shape=(self.filters,),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype,
      )
    else:
      self.bias = None

    self.built = True

  def compute_final_output_shape(
      self, input_shape, kernel_size, strides, is_depthwise=True):
    input_shape = tf.TensorShape(input_shape).as_list()
    # By using list(), output_shape is a copy of input_shape, instead of a
    # reference to input_shape.
    output_shape = list(input_shape)
    _, c_axis, h_axis, w_axis = self._get_input_axis()

    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = strides

    if self.output_padding is None:
      out_pad_h = out_pad_w = None
    else:
      out_pad_h, out_pad_w = self.output_padding

    if is_depthwise:
      # Convolution is performed separately on each spatial domain.
      output_shape[c_axis] = input_shape[c_axis]
    else:
      # Pointwise convolution maps input channels to output filters.
      output_shape[c_axis] = self.filters

    output_shape[h_axis] = conv_utils.deconv_output_length(
        output_shape[h_axis],
        kernel_h,
        padding=self.padding,
        output_padding=out_pad_h,
        stride=stride_h,
        dilation=self.dilation_rate[0],
    )
    output_shape[w_axis] = conv_utils.deconv_output_length(
        output_shape[w_axis],
        kernel_w,
        padding=self.padding,
        output_padding=out_pad_w,
        stride=stride_w,
        dilation=self.dilation_rate[1],
    )
    return tf.TensorShape(output_shape)

  def conv_transpose_op(self, inputs, filters, strides, padding,
                        output_padding, dilation_rate,
                        kernel_quantizer, kernel_weights, use_bias,
                        bias_quantizer, bias, activation, is_depthwise):
    """Transpose convolution op that shared by both depthwise and pointwise."""

    batch_size, out_height, out_width, kernel_h, kernel_w = (
        self._get_output_size(inputs, output_padding, padding, strides,
                              dilation_rate, kernel_weights))

    if kernel_quantizer:
      quantized_kernel = kernel_quantizer(kernel_weights)
    else:
      quantized_kernel = kernel_weights

    if self.data_format == "channels_first":
      output_shape = (batch_size, filters, out_height, out_width)
    else:
      output_shape = (batch_size, out_height, out_width, filters)

    output_shape_tensor = array_ops.stack(output_shape)

    outputs = tf.keras.backend.conv2d_transpose(
        inputs,
        quantized_kernel,
        output_shape_tensor,
        strides=strides,
        padding=padding,
        data_format=self.data_format,
        dilation_rate=dilation_rate)

    if not context.executing_eagerly():
      # Infer the static output shape:
      out_shape = self.compute_final_output_shape(
          input_shape=inputs.shape,
          kernel_size=(kernel_h, kernel_w),
          strides=strides,
          is_depthwise=is_depthwise)
      outputs.set_shape(out_shape)

    if use_bias:
      quantized_bias = bias_quantizer(bias) if bias_quantizer else bias
      outputs = tf.keras.backend.bias_add(
          outputs,
          quantized_bias,
          data_format=self.data_format)

    if activation is not None:
      return activation(outputs)

    return outputs

  def call(self, inputs):
    input_shape = array_ops.shape(inputs)
    _, input_channel, _, _ = self._get_input_dims(input_shape)

    # First apply depthwise transposed convolution.
    x = self.conv_transpose_op(
        inputs=inputs,
        # Depthwise convolution doesn't operate across channels. Thereofore its
        # output channels is the same as input channels.
        filters=input_channel,
        strides=self.strides,
        padding=self.padding,
        output_padding=self.output_padding,
        dilation_rate=self.dilation_rate,
        kernel_quantizer=self.depthwise_kernel_quantizer_internal,
        kernel_weights=self.depthwise_kernel,
        use_bias=False,  # Usually set bias=False for depthwise conv.
        bias_quantizer=None,
        bias=None,
        activation=self.depthwise_activation,
        is_depthwise=True)

    # Then apply pointwise transposed convolution
    x = self.conv_transpose_op(
        inputs=x,
        # Pointwise convolution maps input channels to output filters.
        filters=self.filters,
        strides=(1, 1),   # strides is set to (1, 1) for pointwise conv.
        # Though it will not applied in pointwise conv, we need to set
        # padding here to pass value checking in keras utility functions.
        padding=self.padding,
        output_padding=None,  # Prevent output_padding from adding twice.
        dilation_rate=self.dilation_rate,
        kernel_quantizer=self.pointwise_kernel_quantizer_internal,
        kernel_weights=self.pointwise_kernel,
        use_bias=self.use_bias,
        bias_quantizer=self.bias_quantizer_internal,
        bias=self.bias,
        activation=self.pointwise_activation,
        is_depthwise=False)

    return x

  def get_config(self):
    config = super().get_config()
    config.update({
        "filters": self.filters,
        "kernel_size": self.kernel_size,
        "strides": self.strides,
        "padding": self.padding,
        "output_padding": self.output_padding,
        "dilation_rate": self.dilation_rate,
        "data_format": self.data_format,
        "depth_multiplier": self.depth_multiplier,
        "activation": self.activation,
        "use_bias": self.use_bias,
        "depthwise_kernel_quantizer": constraints.serialize(
            self.depthwise_kernel_quantizer_internal),
        "pointwise_kernel_quantizer": constraints.serialize(
            self.pointwise_kernel_quantizer_internal),
        "bias_quantizer": constraints.serialize(
            self.bias_quantizer_internal,
            ),
    })
    return config

  def get_quantizers(self):
    return [
        self.depthwise_kernel_quantizer_internal,
        self.pointwise_kernel_quantizer_internal,
        self.bias_quantizer_internal,
        self.depthwise_activation,
        self.pointwise_activation,
    ]

  def get_prunable_weights(self):
    w = [self.depthwise_kernel, self.pointwise_kernel]
    if self.use_bias:
      w.append(self.bias)

    return w


class QSeparableConv2DTransposeCPU(QSeparableConv2DTransposeTPU):
  """CPU version of Quantized Separable Conv2DTranspose layer.

  Important Notes:
  * This implementation can run on TPU, GPU and CPU. But the training speed can
  be significantly slower than the TPU/GPU version.

  * QSeparableConv2DTransposeCPU and QSeparableConv2DTransposeTPU layer have
  the same shape on kernel and bias variables. With the same input and the same
  weights, the output of the two layers are the same.

  """

  def conv_transpose_op(self, inputs, filters, strides, padding,
                        output_padding, dilation_rate,
                        kernel_quantizer, kernel_weights, use_bias,
                        bias_quantizer, bias, activation, is_depthwise):
    """Transpose convolution op that shared by both depthwise and pointwise."""

    batch_size, out_height, out_width, kernel_h, kernel_w = (
        self._get_output_size(inputs, output_padding, padding, strides,
                              dilation_rate, kernel_weights))

    if kernel_quantizer:
      quantized_kernel = kernel_quantizer(kernel_weights)
    else:
      quantized_kernel = kernel_weights

    output_filters = 1 if is_depthwise else filters

    if self.data_format == "channels_first":
      output_shape = (batch_size, output_filters, out_height, out_width)
    else:
      output_shape = (batch_size, out_height, out_width, output_filters)

    output_shape_tensor = array_ops.stack(output_shape)

    # Split the input channels into groups.
    x = tf.split(inputs, self._input_shape[-1], axis=-1)

    if is_depthwise:
      # For depthwise convolution, since CPU doesn't support grouped
      # convolution, we run convolution on each slice of inputs and concat
      # the results.
      outputs = [
          tf.keras.backend.conv2d_transpose(
              x=x[i],
              kernel=quantized_kernel[:, :, :, i : i + 1],
              output_shape=output_shape_tensor,
              strides=strides,
              padding=padding,
              data_format=self.data_format,
              dilation_rate=dilation_rate) for i in range(len(x))]

      # Concat the channels.
      outputs = tf.concat(outputs, axis=-1)

    else:
      outputs = tf.keras.backend.conv2d_transpose(
          inputs,
          quantized_kernel,
          output_shape_tensor,
          strides=strides,
          padding=padding,
          data_format=self.data_format,
          dilation_rate=dilation_rate)

    if not context.executing_eagerly():
      # Infer the static output shape:
      out_shape = self.compute_final_output_shape(
          input_shape=inputs.shape,
          kernel_size=(kernel_h, kernel_w),
          strides=strides,
          is_depthwise=is_depthwise)
      outputs.set_shape(out_shape)

    if use_bias:
      quantized_bias = bias_quantizer(bias) if bias_quantizer else bias
      outputs = tf.keras.backend.bias_add(
          outputs,
          quantized_bias,
          data_format=self.data_format)

    if activation is not None:
      return activation(outputs)

    return outputs
