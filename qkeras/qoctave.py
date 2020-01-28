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
"""Octave Convolution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import UpSampling2D
from .qlayers import QActivation
from .qconvolutional import QConv2D
from .qconvolutional import QSeparableConv2D
from .qpooling import QAveragePooling2D


def GetActivationSuffix(activation):
  """Returns suffix for layer name to facilitate debugging."""
  if not activation:
    return "linear"

  if "po2" in activation:
    return "q2"
  elif "quantized_relu" in activation:
    suffix = "qr"
  elif "quantized_tanh" in activation:
    suffix = "qt"
  else:
    suffix = "qb"

  numbers = re.findall(r"[0-9]+", activation)

  numbers = [n + "_" if len(n) > 1 else n for n in numbers]

  return suffix + "".join(numbers)


def QOctaveConv2D(
    filters,
    kernel_size,
    alpha,
    strides=(1, 1),
    padding="valid",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    # NOTE: kernel_regularizer not used with separable convolution
    kernel_regularizer=None,
    bias_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    use_separable=True,
    name="",
    **kwargs):
  """Implements quantized QOctaveConv2D."""

  def _QOctaveConv2DInternal(x):
    """Computes QOctaveConv2D on a tensor."""

    x_h, x_l = x

    bias_quantizer = kwargs.get("bias_quantizer", None)
    kernel_quantizer = kwargs.get("kernel_quantizer", None)
    depthwise_quantizer = kwargs.get("depthwise_quantizer", None)
    pointwise_quantizer = kwargs.get("pointwise_quantizer", None)
    acc_quantizer = kwargs.get("acc_quantizer", None)
    pooling_quantizer = kwargs.get("pooling_quantizer", None)
    depthwise_activation = kwargs.get("depthwise_activation", None)
    activation = kwargs.get("activation", None)

    bias_range = kwargs.get("bias_range", 1.0)
    kernel_range = kwargs.get("kernel_range", 1.0)
    depthwise_range = kwargs.get("depthwise_range", 1.0)
    pointwise_range = kwargs.get("pointwise_range", 1.0)

    if activation:
      act_suffix = "_" + GetActivationSuffix(activation)
    acc_suffix = "_" + GetActivationSuffix(acc_quantizer)

    if alpha == -1.0:
      if use_separable:
        x_h = QSeparableConv2D(
            filters, kernel_size, strides=strides, padding=padding,
            depthwise_regularizer=kernel_regularizer,
            depthwise_constraint=kernel_constraint,
            depthwise_initializer=kernel_initializer,
            pointwise_regularizer=kernel_regularizer,
            pointwise_constraint=kernel_constraint,
            pointwise_initializer=kernel_initializer,
            bias_regularizer=bias_regularizer,
            bias_constraint=bias_constraint,
            bias_initializer=bias_initializer,
            depthwise_quantizer=depthwise_quantizer,
            pointwise_quantizer=pointwise_quantizer,
            bias_quantizer=bias_quantizer,
            depthwise_activation=depthwise_activation,
            pointwise_range=pointwise_range,
            depthwise_range=depthwise_range,
            bias_range=bias_range,
            name=name + "_c_h_to_h")(x_h)
      else:
        x_h = QConv2D(
            filters, kernel_size, strides=strides, padding=padding,
            kernel_regularizer=kernel_regularizer,
            kernel_constraint=kernel_constraint,
            kernel_initializer=kernel_initializer,
            bias_regularizer=bias_regularizer,
            bias_constraint=bias_constraint,
            bias_initializer=bias_initializer,
            kernel_quantizer=kernel_quantizer,
            bias_quantizer=bias_quantizer,
            kernel_range=kernel_range,
            bias_range=bias_range,
            name=name + "_c_h_to_h")(x_h)

      if activation:
        x_h = QActivation(
            activation, name=name + "_c_h_to_h_act" + act_suffix)(
                x_h)

      return [x_h, None]

    co_h = int(filters * (1 - alpha))
    co_l = filters - co_h

    x_h_to_h = None
    x_h_to_l = None
    x_l_to_l = None
    x_l_to_h = None

    if co_h > 0:
      if x_h is not None:
        if use_separable:
          x_h_to_h = QSeparableConv2D(
              co_h, kernel_size, strides=strides, padding=padding,
              depthwise_regularizer=kernel_regularizer,
              depthwise_constraint=kernel_constraint,
              depthwise_initializer=kernel_initializer,
              pointwise_regularizer=kernel_regularizer,
              pointwise_constraint=kernel_constraint,
              pointwise_initializer=kernel_initializer,
              bias_regularizer=bias_regularizer,
              bias_constraint=bias_constraint,
              bias_initializer=bias_initializer,
              depthwise_quantizer=depthwise_quantizer,
              pointwise_quantizer=pointwise_quantizer,
              bias_quantizer=bias_quantizer,
              depthwise_activation=depthwise_activation,
              pointwise_range=pointwise_range,
              depthwise_range=depthwise_range,
              bias_range=bias_range,
              name=name + "_c_h_to_h")(x_h)
        else:
          x_h_to_h = QConv2D(
              co_h, kernel_size, strides=strides, padding=padding,
              kernel_regularizer=kernel_regularizer,
              kernel_constraint=kernel_constraint,
              kernel_initializer=kernel_initializer,
              bias_regularizer=bias_regularizer,
              bias_constraint=bias_constraint,
              bias_initializer=bias_initializer,
              kernel_quantizer=kernel_quantizer,
              bias_quantizer=bias_quantizer,
              kernel_range=kernel_range,
              bias_range=bias_range,
              name=name + "_c_h_to_h")(x_h)

        if acc_quantizer:
          x_h_to_h = QActivation(
              acc_quantizer,
              name=name + "_c_h_to_h_act" + acc_suffix)(x_h_to_h)

    if co_l > 0:
      if x_h is not None:
        x_h_to_l = QAveragePooling2D(
            pool_size=2, strides=2,
            quantizer=pooling_quantizer,
            name=name + "_avg_h_to_l")(x_h)

        if use_separable:
          x_h_to_l = QSeparableConv2D(
              co_l, kernel_size, strides=strides, padding=padding,
              depthwise_regularizer=kernel_regularizer,
              depthwise_constraint=kernel_constraint,
              depthwise_initializer=kernel_initializer,
              pointwise_regularizer=kernel_regularizer,
              pointwise_constraint=kernel_constraint,
              pointwise_initializer=kernel_initializer,
              bias_regularizer=bias_regularizer,
              bias_constraint=bias_constraint,
              bias_initializer=bias_initializer,
              depthwise_quantizer=depthwise_quantizer,
              pointwise_quantizer=pointwise_quantizer,
              bias_quantizer=bias_quantizer,
              depthwise_activation=depthwise_activation,
              pointwise_range=pointwise_range,
              depthwise_range=depthwise_range,
              bias_range=bias_range,
              name=name + "_c_h_to_l")(x_h_to_l)
        else:
          x_h_to_l = QConv2D(
              co_l, kernel_size, strides=strides, padding=padding,
              kernel_regularizer=kernel_regularizer,
              kernel_constraint=kernel_constraint,
              kernel_initializer=kernel_initializer,
              bias_regularizer=bias_regularizer,
              bias_constraint=bias_constraint,
              bias_initializer=bias_initializer,
              kernel_quantizer=kernel_quantizer,
              bias_quantizer=bias_quantizer,
              kernel_range=kernel_range,
              bias_range=bias_range,
              name=name + "_c_h_to_l")(x_h_to_l)

        if acc_quantizer:
          x_h_to_l = QActivation(
              acc_quantizer,
              name=name + "_c_h_to_l_act" + acc_suffix)(x_h_to_l)

    if co_h > 0:
      if x_l is not None:
        _, height, width, _ = x_l.shape.as_list()
        if height == 1 and width == 1:
          local_kernel = 1
          local_strides = 1
          local_padding = "same"
          upsampling = False
        else:
          local_kernel = kernel_size
          local_strides = strides
          local_padding = padding
          upsampling = True

        if use_separable and upsampling:
          x_l_to_h = QSeparableConv2D(
              co_h, kernel_size, strides=strides, padding=padding,
              depthwise_regularizer=kernel_regularizer,
              depthwise_constraint=kernel_constraint,
              depthwise_initializer=kernel_initializer,
              pointwise_regularizer=kernel_regularizer,
              pointwise_constraint=kernel_constraint,
              pointwise_initializer=kernel_initializer,
              bias_regularizer=bias_regularizer,
              bias_constraint=bias_constraint,
              bias_initializer=bias_initializer,
              depthwise_quantizer=depthwise_quantizer,
              pointwise_quantizer=pointwise_quantizer,
              bias_quantizer=bias_quantizer,
              depthwise_activation=depthwise_activation,
              pointwise_range=pointwise_range,
              depthwise_range=depthwise_range,
              bias_range=bias_range,
              name=name + "_c_l_to_h")(x_l)
        else:
          x_l_to_h = QConv2D(
              co_h, local_kernel, strides=local_strides, padding=local_padding,
              kernel_regularizer=kernel_regularizer,
              kernel_constraint=kernel_constraint,
              kernel_initializer=kernel_initializer,
              bias_regularizer=bias_regularizer,
              bias_constraint=bias_constraint,
              bias_initializer=bias_initializer,
              kernel_quantizer=kernel_quantizer,
              bias_quantizer=bias_quantizer,
              kernel_range=kernel_range,
              bias_range=bias_range,
              name=name + "_c_l_to_h")(x_l)

        if acc_quantizer:
          x_l_to_h = QActivation(
              acc_quantizer,
              name=name + "_c_l_to_h_act" + acc_suffix)(x_l_to_h)

        if upsampling:
          x_l_to_h = UpSampling2D(
              size=(2, 2), name=name + "_u_l_to_h")(x_l_to_h)

    if co_l > 0:
      if x_l is not None:
        if use_separable:
          x_l_to_l = QSeparableConv2D(
              co_l, kernel_size, strides=strides, padding=padding,
              depthwise_regularizer=kernel_regularizer,
              depthwise_constraint=kernel_constraint,
              depthwise_initializer=kernel_initializer,
              pointwise_regularizer=kernel_regularizer,
              pointwise_constraint=kernel_constraint,
              pointwise_initializer=kernel_initializer,
              bias_regularizer=bias_regularizer,
              bias_constraint=bias_constraint,
              bias_initializer=bias_initializer,
              depthwise_quantizer=depthwise_quantizer,
              pointwise_quantizer=depthwise_quantizer,
              bias_quantizer=bias_quantizer,
              depthwise_activation=depthwise_activation,
              pointwise_range=pointwise_range,
              depthwise_range=depthwise_range,
              bias_range=bias_range,
              name=name + "_c_l_to_l")(x_l)
        else:
          x_l_to_l = QConv2D(
              co_l, kernel_size, strides=strides, padding=padding,
              kernel_regularizer=kernel_regularizer,
              kernel_constraint=kernel_constraint,
              kernel_initializer=kernel_initializer,
              bias_regularizer=bias_regularizer,
              bias_constraint=bias_constraint,
              bias_initializer=bias_initializer,
              kernel_quantizer=kernel_quantizer,
              bias_quantizer=bias_quantizer,
              kernel_range=kernel_range,
              bias_range=bias_range,
              name=name + "_c_l_to_l")(x_l)

        if acc_quantizer:
          x_l_to_l = QActivation(
              acc_quantizer, name=name + "_c_l_to_l_act" + acc_suffix)(
                  x_l_to_l)

    if x_h_to_h is not None and x_l_to_h is not None:
      x_h = Add(name=name + "_a_h")([x_h_to_h, x_l_to_h])
    elif x_h_to_h is not None:
      x_h = x_h_to_h
    elif x_l_to_h is not None:
      x_h = x_l_to_h
    else:
      x_h = None

    if x_l_to_l is not None and x_h_to_l is not None:
      x_l = Add(name=name + "_a_l")([x_l_to_l, x_h_to_l])
    elif x_l_to_l is not None:
      x_l = x_l_to_l
    elif x_h_to_l is not None:
      x_l = x_h_to_l
    else:
      x_l = None

    if x_h is not None and activation is not None:
      x_h = QActivation(activation,
                        name=name + "_h_act" + act_suffix)(x_h)

    if x_l is not None and activation is not None:
      x_l = QActivation(activation,
                        name=name + "_l_act" + act_suffix)(x_l)

    return [x_h, x_l]

  return _QOctaveConv2DInternal


def OctaveConv2D(
    filters, kernel_size, alpha,
    strides=(1, 1), padding="valid",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    activation=None,
    use_separable=True,
    name="",
    **kwargs):

  """Implements OctaveConv2D."""

  def _OctaveConv2DInternal(x):

    """Computes octave on tensor."""

    acc_quantizer = kwargs.get("acc_quantizer", None)

    x_h, x_l = x

    if alpha == -1.0:
      if use_separable:
        x_h = SeparableConv2D(
            filters, kernel_size, strides=strides, padding=padding,
            depthwise_regularizer=kernel_regularizer,
            depthwise_constraint=kernel_constraint,
            depthwise_initializer=kernel_initializer,
            pointwise_regularizer=kernel_regularizer,
            pointwise_constraint=kernel_constraint,
            pointwise_initializer=kernel_initializer,
            bias_regularizer=bias_regularizer,
            bias_constraint=bias_constraint,
            bias_initializer=bias_initializer,
            name=name + "_c_h_to_h")(x_h)
      else:
        x_h = Conv2D(
            filters, kernel_size, strides=strides, padding=padding,
            kernel_regularizer=kernel_regularizer,
            kernel_constraint=kernel_constraint,
            kernel_initializer=kernel_initializer,
            bias_regularizer=bias_regularizer,
            bias_constraint=bias_constraint,
            bias_initializer=bias_initializer,
            name=name+"_c_h_to_h")(x_h)

      if activation:
        x_h = Activation(activation, name=name + "_c_h_to_h_act")(x_h)

      return [x_h, None]

    co_h = int(filters * (1 - alpha))
    co_l = filters - co_h

    x_h_to_h = None
    x_h_to_l = None
    x_l_to_l = None
    x_l_to_h = None

    if co_h > 0:
      if x_h is not None:
        if use_separable:
          x_h_to_h = SeparableConv2D(
              co_h, kernel_size, strides=strides, padding=padding,
              depthwise_regularizer=kernel_regularizer,
              depthwise_constraint=kernel_constraint,
              depthwise_initializer=kernel_initializer,
              pointwise_regularizer=kernel_regularizer,
              pointwise_constraint=kernel_constraint,
              pointwise_initializer=kernel_initializer,
              bias_regularizer=bias_regularizer,
              bias_constraint=bias_constraint,
              bias_initializer=bias_initializer,
              name=name + "_c_h_to_h")(x_h)
        else:
          x_h_to_h = Conv2D(
              co_h, kernel_size, strides=strides, padding=padding,
              kernel_regularizer=kernel_regularizer,
              kernel_constraint=kernel_constraint,
              kernel_initializer=kernel_initializer,
              bias_regularizer=bias_regularizer,
              bias_constraint=bias_constraint,
              bias_initializer=bias_initializer,
              name=name + "_c_h_to_h")(x_h)

        if activation:
          x_h_to_h = Activation(
              acc_quantizer, name=name + "_c_h_to_h_act")(x_h_to_h)

    if co_l > 0:
      if x_h is not None:
        x_h_to_l = AveragePooling2D(pool_size=2, strides=2,
                                    name=name + "_p_h_to_l")(x_h)

        if use_separable:
          x_h_to_l = SeparableConv2D(
              co_l, kernel_size, strides=strides, padding=padding,
              depthwise_regularizer=kernel_regularizer,
              depthwise_constraint=kernel_constraint,
              depthwise_initializer=kernel_initializer,
              pointwise_regularizer=kernel_regularizer,
              pointwise_constraint=kernel_constraint,
              pointwise_initializer=kernel_initializer,
              bias_regularizer=bias_regularizer,
              bias_constraint=bias_constraint,
              bias_initializer=bias_initializer,
              name=name + "_c_h_to_l")(x_h_to_l)
        else:
          x_h_to_l = Conv2D(
              co_l, kernel_size, strides=strides, padding=padding,
              kernel_regularizer=kernel_regularizer,
              kernel_constraint=kernel_constraint,
              kernel_initializer=kernel_initializer,
              bias_regularizer=bias_regularizer,
              bias_constraint=bias_constraint,
              bias_initializer=bias_initializer,
              name=name + "_c_h_to_l")(x_h_to_l)

        if activation:
          x_h_to_l = Activation(
              acc_quantizer, name=name + "_c_h_to_l_act")(x_h_to_l)

    if co_h > 0:
      if x_l is not None:
        _, height, width, _ = x_l.shape.as_list()
        if height == 1 and width == 1:
          local_kernel = 1
          local_strides = 1
          local_padding = "same"
          upsampling = False
        else:
          local_kernel = kernel_size
          local_strides = strides
          local_padding = padding
          upsampling = True

        if use_separable and upsampling:
          x_l_to_h = SeparableConv2D(
              co_h, kernel_size, strides=strides, padding=padding,
              depthwise_regularizer=kernel_regularizer,
              depthwise_constraint=kernel_constraint,
              depthwise_initializer=kernel_initializer,
              pointwise_regularizer=kernel_regularizer,
              pointwise_constraint=kernel_constraint,
              pointwise_initializer=kernel_initializer,
              bias_regularizer=bias_regularizer,
              bias_constraint=bias_constraint,
              bias_initializer=bias_initializer,
              name=name + "_c_l_to_h")(x_l)
        else:
          x_l_to_h = Conv2D(
              co_h, local_kernel, strides=local_strides, padding=local_padding,
              kernel_regularizer=kernel_regularizer,
              kernel_constraint=kernel_constraint,
              kernel_initializer=kernel_initializer,
              bias_regularizer=bias_regularizer,
              bias_constraint=bias_constraint,
              bias_initializer=bias_initializer,
              name=name + "_c_l_to_h")(x_l)

        if activation:
          x_l_to_h = Activation(
              acc_quantizer, name=name + "_c_l_to_h_act")(x_l_to_h)

        if upsampling:
          x_l_to_h = UpSampling2D(
              size=(2, 2), name=name + "_u_l_to_h")(x_l_to_h)

    if co_l > 0:
      if x_l is not None:
        if use_separable:
          x_l_to_l = SeparableConv2D(
              co_l, kernel_size, strides=strides, padding=padding,
              kernel_regularizer=kernel_regularizer,
              kernel_constraint=kernel_constraint,
              kernel_initializer=kernel_initializer,
              bias_regularizer=bias_regularizer,
              bias_constraint=bias_constraint,
              bias_initializer=bias_initializer,
              name=name + "_c_l_to_l")(x_l)
        else:
          x_l_to_l = Conv2D(
              co_l, kernel_size, strides=strides, padding=padding,
              kernel_regularizer=kernel_regularizer,
              kernel_constraint=kernel_constraint,
              kernel_initializer=kernel_initializer,
              bias_regularizer=bias_regularizer,
              bias_constraint=bias_constraint,
              bias_initializer=bias_initializer,
              name=name + "_c_l_to_l")(x_l)

        if activation:
          x_l_to_l = Activation(
              acc_quantizer, name=name + "_c_l_to_l_act")(x_l_to_l)

    if x_h_to_h is not None and x_l_to_h is not None:
      x_h = Add(name=name + "_a_h")([x_h_to_h, x_l_to_h])
    elif x_h_to_h is not None:
      x_h = x_h_to_h
    elif x_l_to_h is not None:
      x_h = x_l_to_h
    else:
      x_h = None

    if x_l_to_l is not None and x_h_to_l is not None:
      x_l = Add(name=name + "_a_l")([x_l_to_l, x_h_to_l])
    elif x_l_to_l is not None:
      x_l = x_l_to_l
    elif x_h_to_l is not None:
      x_l = x_h_to_l
    else:
      x_l = None

    if x_h is not None:
      x_h = Activation(activation, name=name + "_h_act")(x_h)

    if x_l is not None:
      x_l = Activation(activation, name=name + "_l_act")(x_l)

    return (x_h, x_l)

  return _OctaveConv2DInternal
