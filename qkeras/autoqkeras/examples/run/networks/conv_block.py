# ==============================================================================
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

import os
from tensorflow.initializers import *  # pylint: disable=wildcard-import
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *  # pylint: disable=wildcard-import
from qkeras import *  # pylint: disable=wildcard-import


class ConvBlockNetwork(object):
  """Creates Convolutional block type of network."""

  def __init__(
      self,
      shape,
      nb_classes,
      kernel_size,
      filters,
      dropout_rate=0.0,
      with_maxpooling=True,
      with_batchnorm=True,
      kernel_initializer="he_normal",
      bias_initializer="zeros",
      use_separable=False,
      use_xnornet_trick=False
  ):
    """Creates class.

    Args:
      shape: shape of inputs.
      nb_classes: number of output classes.
      kernel_size: kernel_size of network.
      filters: sizes of filters (if entry is a list, we create a block).
      dropout_rate: dropout rate if > 0.
      with_maxpooling: if true, use maxpooling.
      with_batchnorm: with BatchNormalization.
      kernel_initializer: kernel_initializer.
      bias_initializer: bias and beta initializer.
      use_separable: if "dsp", do conv's 1x3 + 3x1. If "mobilenet",
        use MobileNet separable convolution. If False or "none", perform single
        conv layer.
      use_xnornet_trick: use bn+act after max pool to enable binary
        to avoid saturation to largest value.
    """

    self.shape = shape
    self.nb_classes = nb_classes
    self.kernel_size = kernel_size
    self.filters = filters
    self.dropout_rate = dropout_rate
    self.with_maxpooling = with_maxpooling
    self.with_batchnorm = with_batchnorm
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.use_separable = use_separable
    self.use_xnornet_trick = use_xnornet_trick

  def build(self):
    """Builds model."""
    x = x_in = Input(self.shape, name="input")
    for i in range(len(self.filters)):
      if len(self.filters) > 1:
        name_suffix_list = [str(i)]
      else:
        name_suffix_list = []
      if not isinstance(self.filters[i], list):
        filters = [self.filters[i]]
      else:
        filters = self.filters[i]
      for j in range(len(filters)):
        if len(filters) > 1:
          name_suffix = "_".join(name_suffix_list + [str(j)])
        else:
          name_suffix = "_".join(name_suffix_list)
        if self.use_separable == "dsp":
          kernels = [(1, self.kernel_size), (self.kernel_size, 1)]
        else:
          kernels = [(self.kernel_size, self.kernel_size)]
        for k, kernel in enumerate(kernels):
          strides = 1
          if (
              not self.with_maxpooling and j == len(filters)-1 and
              k == len(kernels)-1
          ):
            strides = 2
          if self.use_separable == "dsp":
            kernel_suffix = (
                "".join([str(k) for k in kernel]) + "_" + name_suffix)
          elif self.use_separable == "mobilenet":
            depth_suffix = (
                "".join([str(k) for k in kernel]) + "_" + name_suffix)
            kernel_suffix = "11_" + name_suffix
          else:
            kernel_suffix = name_suffix
          if self.use_separable == "mobilenet":
            x = DepthwiseConv2D(
                kernel,
                padding="same", strides=strides,
                use_bias=False,
                name="conv2d_dw_" + depth_suffix)(x)
            if self.with_batchnorm:
              x = BatchNormalization(name="conv2d_dw_bn_" + depth_suffix)(x)
            x = Activation("relu", name="conv2d_dw_act_" + depth_suffix)(x)
            kernel = (1, 1)
            strides = 1
          x = Conv2D(
              filters[j], kernel,
              strides=strides, use_bias=not self.with_batchnorm,
              padding="same",
              kernel_initializer=self.kernel_initializer,
              bias_initializer=self.bias_initializer,
              name="conv2d_" + kernel_suffix)(x)
          if not (
              self.with_maxpooling and self.use_xnornet_trick and
              j == len(filters)-1 and k == len(kernels)-1
          ):
            if self.with_batchnorm:
              x = BatchNormalization(
                  beta_initializer=self.bias_initializer,
                  name="bn_" + kernel_suffix)(x)
            x = Activation("relu", name="act_" + kernel_suffix)(x)
      if self.with_maxpooling:
        x = MaxPooling2D(2, 2, name="mp_" + name_suffix)(x)
        # this is a trick from xnornet to enable full binary or ternary
        # networks to be after maxpooling.
        if self.use_xnornet_trick:
          x = BatchNormalization(
              beta_initializer=self.bias_initializer,
              name="mp_bn_" + name_suffix)(x)
          x = Activation("relu", name="mp_act_" + name_suffix)(x)
      if self.dropout_rate > 0:
        x = Dropout(self.dropout_rate, name="drop_" + name_suffix)(x)

    if x.shape.as_list()[1] > 1:
      x = Flatten(name="flatten")(x)
      x = Dense(
          self.nb_classes,
          kernel_initializer=self.kernel_initializer,
          bias_initializer=self.bias_initializer,
          name="dense")(x)
      x = Activation("softmax", name="softmax")(x)
    else:
      x = Conv2D(
          self.nb_classes, 1, strides=1, padding="same",
          kernel_initializer=self.kernel_initializer,
          bias_initializer=self.bias_initializer,
          name="dense")(x)
      x = Activation("softmax", name="softmax")(x)
      x = Flatten(name="flatten")(x)

    model = Model(inputs=[x_in], outputs=[x])

    return model

if __name__ == "__main__":
  import os

  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = ""

  model = ConvBlockNetwork(
      shape=(64, 64, 1),
      nb_classes=10,
      kernel_size=3,
      filters=[16, [32]*3, 48, 64, 128],
      dropout_rate=0.0,
      with_maxpooling=False,
      with_batchnorm=True,
      use_separable="mobilenet",
      use_xnornet_trick=True
  ).build()

  model.summary()
