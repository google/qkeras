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
"""QOctave example."""
import numpy as np
import sys
from tensorflow.keras import activations
from tensorflow.keras import initializers
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from functools import partial
from qkeras import *   # pylint: disable=wildcard-import


def create_model():
  """use qocatve in network."""
  kernel_initializer=initializers.he_normal(seed=42)

  x = x_in = Input(shape=(256, 256, 3))

  # Block 1
  high, low = QOctaveConv2D(
      32, (3, 3),
      alpha=0.5,
      strides=(2, 2),
      padding='valid',
      kernel_initializer=kernel_initializer,
      bias_initializer="zeros",
      bias_quantizer="quantized_bits(4,1)",
      depthwise_quantizer="quantized_bits(4,1)",
      depthwise_activation="quantized_bits(6,2,1)",
      pointwise_quantizer="quantized_bits(4,1)",
      acc_quantizer="quantized_bits(16,7,1)",
      activation="quantized_relu(6,2)",
      use_separable=True,
      name='block1_conv1')([x, None])

  # Block 2
  high, low = QOctaveConv2D(
      64, (3, 3),
      alpha=0.4,
      strides=(2, 2),
      padding='same',
      kernel_initializer=kernel_initializer,
      bias_initializer="zeros",
      bias_quantizer="quantized_bits(4,1)",
      depthwise_quantizer="quantized_bits(4,1)",
      depthwise_activation="quantized_bits(6,2,1)",
      pointwise_quantizer="quantized_bits(4,1)",
      acc_quantizer="quantized_bits(16,7,1)",
      activation="quantized_relu(6,2)",
      use_separable=True,
      name='block2_conv1')([high, low])

  # Block 3
  high, low = QOctaveConv2D(
      64, (3, 3),
      alpha=0.4,
      strides=(2, 2),
      padding='same',
      kernel_initializer=kernel_initializer,
      bias_initializer="zeros",
      bias_quantizer="quantized_bits(4,1)",
      depthwise_quantizer="quantized_bits(4,1)",
      depthwise_activation="quantized_bits(6,2,1)",
      pointwise_quantizer="quantized_bits(4,1)",
      acc_quantizer="quantized_bits(16,7,1)",
      activation="quantized_relu(6,2)",
      use_separable=True,
      name='block3_conv1')([high, low])

  high, low = QOctaveConv2D(
      32, (3, 3),
      alpha=0.4,
      strides=(1, 1),
      padding='same',
      kernel_initializer=kernel_initializer,
      bias_initializer='zeros',
      bias_quantizer="quantized_bits(4,1)",
      depthwise_quantizer="quantized_bits(4,1)",
      depthwise_activation="quantized_bits(6,2,1)",
      pointwise_quantizer="quantized_bits(4,1)",
      acc_quantizer="quantized_bits(16,7,1)",
      activation="quantized_relu(6,2)",
      use_separable=True,
      name='block3_conv2')([high, low])

  high, low = QOctaveConv2D(
      32, (3, 3),
      alpha=0.3,
      strides=(1, 1),
      padding='same',
      kernel_initializer=kernel_initializer,
      bias_initializer='zeros',
      bias_quantizer="quantized_bits(4,1)",
      depthwise_quantizer="quantized_bits(4,1)",
      depthwise_activation="quantized_bits(6,2,1)",
      pointwise_quantizer="quantized_bits(4,1)",
      acc_quantizer="quantized_bits(16,7,1)",
      activation="quantized_relu(6,2)",
      use_separable=True,
      name='block3_conv3')([high, low])

  x, _ = QOctaveConv2D(
      32, (3, 3),
      alpha=0.0,
      strides=(2, 2),
      padding='same',
      kernel_initializer=kernel_initializer,
      bias_initializer='zeros',
      bias_quantizer="quantized_bits(4,1)",
      depthwise_quantizer="quantized_bits(4,1)",
      depthwise_activation="quantized_bits(6,2,1)",
      pointwise_quantizer="quantized_bits(4,1)",
      acc_quantizer="quantized_bits(16,7,1)",
      activation="quantized_relu(6,2)",
      use_separable=True,
      name='block3_conv_down')([high, low])

  # Upsample
  x = UpSampling2D(size=(2, 2), data_format="channels_last")(x)

  x = QConv2D(
      2, (2, 2),
      strides=(1, 1),
      kernel_initializer=kernel_initializer,
      bias_initializer="ones",
      kernel_quantizer=quantized_bits(4, 0, 1),
      bias_quantizer=quantized_bits(4, 0, 1),
      padding="same",
      name="conv_up")(
          x)

  x = Activation("softmax", name="softmax")(x)
  output = x

  model = Model(x_in, output, name='qoctave_network')
  return model


# Create the model
def customLoss(y_true,y_pred):
  log1 = 1.5 * y_true * K.log(y_pred + 1e-9) * K.pow(1-y_pred, 2)
  log0 = 0.5 * (1 - y_true) * K.log((1 - y_pred) + 1e-9) * K.pow(y_pred, 2)
  return (- K.sum(K.mean(log0 + log1, axis = 0)))

if __name__ == '__main__':
  model = create_model()
  model.compile(optimizer="Adam", loss=customLoss, metrics=['acc'])
  model.summary(line_length=100)
  print_qstats(model)
