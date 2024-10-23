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

from .qconvolutional import deconv_output_length
from .quantizers import get_quantizer
from tensorflow.python.eager import context
from tensorflow.python.keras import constraints
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops


# TODO(akshayap): Commonized functionality with QSeparableConv2DTranspose.
class QDepthwiseConv2DTranspose(Conv2DTranspose):
  """Quantized Depthwise Conv2DTranspose layer."""

  # Most of these parameters follow the implementation of Conv2DTranspose
  # in Keras, with the exception of following parameters.
  #
  # depthwise_activation: activation quantizer for depthwise convolution
  # depthwise_kernel_quantizer: quantizer function/class for depthwise kernel
  # bias_quantizer: quantizer function/class for bias
  #
  # we refer the reader to the documentation of Conv2DTranspose in Keras for
  # the other parameters.

  def __init__(
      self,
      filters,
      kernel_size,
      group_size=1,
      strides=(1, 1),
      padding="valid",
      output_padding=None,
      depth_multiplier=1,
      depthwise_activation=None,
      use_bias=True,
      depthwise_kernel_quantizer=None,
      bias_quantizer=None,
      **kwargs,
  ):

    self.filters = filters
    self.kernel_size = kernel_size
    self.strides = strides
    self.padding = padding
    self.output_padding = output_padding
    self.depth_multiplier = depth_multiplier
    self.depthwise_activation = depthwise_activation
    self.use_bias = use_bias
    self.group_size = group_size

    self.depthwise_kernel_quantizer = depthwise_kernel_quantizer
    self.bias_quantizer = bias_quantizer

    self.depthwise_kernel_quantizer_internal = get_quantizer(
        self.depthwise_kernel_quantizer
    )
    self.bias_quantizer_internal = get_quantizer(self.bias_quantizer)

    # optimize parameter set to "auto" scaling mode if possible
    for q in [
        self.depthwise_kernel_quantizer_internal,
    ]:
      if hasattr(q, "_set_trainable_parameter"):
        q._set_trainable_parameter()

    if depthwise_activation is not None:
      self.depthwise_activation = get_quantizer(depthwise_activation)

    super().__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        **kwargs,
    )

  def _get_input_axis(self):
    if self.data_format == "channels_first":
      b_axis, c_axis, h_axis, w_axis = 0, 1, 2, 3
    else:
      b_axis, c_axis, h_axis, w_axis = 0, 3, 1, 2

    return b_axis, c_axis, h_axis, w_axis

  def _get_input_dims(self, input_shape):
    b_axis, c_axis, h_axis, w_axis = self._get_input_axis()

    return (
        input_shape[b_axis],
        input_shape[c_axis],
        input_shape[h_axis],
        input_shape[w_axis],
    )

  def _get_output_size(
      self,
      inputs,
      output_padding,
      padding,
      strides,
      dilation_rate,
      kernel_h,
      kernel_w,
  ):
    input_shape = array_ops.shape(inputs)
    batch_size, _, height, width = self._get_input_dims(input_shape)
    stride_h, stride_w = strides

    dilation_h, dilation_w = dilation_rate[0], dilation_rate[1]

    if output_padding is None:
      out_pad_h = out_pad_w = None
    else:
      out_pad_h, out_pad_w = output_padding

    # Infer the dynamic output shape:
    out_height = deconv_output_length(
        height,
        kernel_h,
        padding=padding,
        output_padding=out_pad_h,
        stride=stride_h,
        dilation=dilation_h,
    )

    out_width = deconv_output_length(
        width,
        kernel_w,
        padding=padding,
        output_padding=out_pad_w,
        stride=stride_w,
        dilation=dilation_w,
    )

    return (batch_size, out_height, out_width)

  def build(self, input_shape):
    self._input_shape = input_shape

    _, input_channel, _, _ = self._get_input_dims(input_shape)
    channel_axis = self._get_input_axis()[1]

    self.input_spec = InputSpec(
        min_ndim=self.rank + 2, axes={channel_axis: input_channel}
    )
    # When setting kernel shape=(kw, kh, 1, input_channel), it does depthwise
    # convolution.
    depthwise_kernel_shape = self.kernel_size + (
        input_channel,
        self.group_size,
    )

    self.depthwise_kernel = self.add_weight(
        name=f"depthwise_kernel",
        shape=depthwise_kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=True,
        dtype=self.dtype,
    )

    if self.use_bias:
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

  def compute_final_output_shape(self, input_shape, kernel_size, strides):
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

    # Convolution is performed separately on each spatial domain.
    output_shape[c_axis] = input_shape[c_axis]

    output_shape[h_axis] = deconv_output_length(
        output_shape[h_axis],
        kernel_h,
        padding=self.padding,
        output_padding=out_pad_h,
        stride=stride_h,
        dilation=self.dilation_rate[0],
    )
    output_shape[w_axis] = deconv_output_length(
        output_shape[w_axis],
        kernel_w,
        padding=self.padding,
        output_padding=out_pad_w,
        stride=stride_w,
        dilation=self.dilation_rate[1],
    )
    return tf.TensorShape(output_shape)

  def conv_transpose_op(
      self,
      inputs,
      filters,
      strides,
      padding,
      output_padding,
      dilation_rate,
      kernel_quantizer,
      kernel_weights,
      use_bias,
      bias_quantizer,
      bias,
      activation,
  ):
    """Transpose convolution operation."""

    kernel_h, kernel_w = self.kernel_size
    batch_size, out_height, out_width = self._get_output_size(
        inputs,
        output_padding,
        padding,
        strides,
        dilation_rate,
        kernel_h,
        kernel_w,
    )

    if kernel_quantizer:
      quantized_kernel = kernel_quantizer(kernel_weights)
    else:
      quantized_kernel = kernel_weights

    output_filters = self.group_size

    if self.data_format == "channels_first":
      output_shape = (batch_size, output_filters, out_height, out_width)
    else:
      output_shape = (batch_size, out_height, out_width, output_filters)

    output_shape_tensor = array_ops.stack(output_shape)

    num_input_channels = self._input_shape[-1]
    if num_input_channels % self.group_size:
      raise ValueError(
          "Input channels should be exactly divisible by group_size."
      )
    num_output_groups = num_input_channels // self.group_size

    # Split the input channels into groups.
    x = tf.split(inputs, num_output_groups, axis=-1)

    # For depthwise convolution, since CPU doesn't support grouped
    # convolution, we run convolution on each slice of inputs and concat
    # the results.
    outputs = [
        tf.keras.backend.conv2d_transpose(
            x=x[i],
            kernel=quantized_kernel[
                :,
                :,
                self.group_size * i : self.group_size * (i + 1),
                :,
            ],
            output_shape=output_shape_tensor,
            strides=strides,
            padding=padding,
            data_format=self.data_format,
            dilation_rate=dilation_rate,
        )
        for i in range(num_output_groups)
    ]

    # Concat the channels.
    outputs = tf.concat(outputs, axis=-1)

    if not context.executing_eagerly():
      # Infer the static output shape:
      out_shape = self.compute_final_output_shape(
          input_shape=inputs.shape,
          kernel_size=(kernel_h, kernel_w),
          strides=strides,
      )
      outputs.set_shape(out_shape)

    if use_bias:
      quantized_bias = bias_quantizer(bias) if bias_quantizer else bias
      outputs = tf.keras.backend.bias_add(
          outputs, quantized_bias, data_format=self.data_format
      )

    if activation is not None:
      return activation(outputs)

    return outputs

  def call(self, inputs):
    input_shape = array_ops.shape(inputs)
    _, input_channel, _, _ = self._get_input_dims(input_shape)

    return self.conv_transpose_op(
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
    )

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
            self.depthwise_kernel_quantizer_internal
        ),
        "bias_quantizer": constraints.serialize(
            self.bias_quantizer_internal,
        ),
        "group_size": self.group_size,
    })
    return config

  def get_quantizers(self):
    return [
        self.depthwise_kernel_quantizer_internal,
        self.bias_quantizer_internal,
        self.depthwise_activation,
    ]

  def get_prunable_weights(self):
    w = [self.depthwise_kernel]
    if self.use_bias:
      w.append(self.bias)

    return w
