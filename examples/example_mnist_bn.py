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
"""Tests mnist batchnormalization used as learned scale factor."""

# to run, THRESHOLD=0.05 WITH_BN=1 EPOCHS=5 TRAIN=1 python example_mnist_bn.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import os

import numpy as np
from six.moves import zip
from tensorflow.keras import callbacks
import tensorflow.keras.backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import to_categorical

from qkeras import *

np.random.seed(42)

TRAIN = 1
NB_EPOCH = 2
BATCH_SIZE = 64
VERBOSE = 1
NB_CLASSES = 10
OPTIMIZER = Adam(lr=0.0001)
VALIDATION_SPLIT = 0.1
WITH_BN = 1
THRESHOLD = 0.1


class LearningRateAdjuster(callbacks.Callback):
  def __init__(self):
    self.learning_rate_factor = 1.0
    pass

  def on_epoch_end(self, epochs, logs):
    max_variance = -1

    for layer in self.model.layers:
      if layer.__class__.__name__ in [
          "BatchNormalization",
          "QBatchNormalization"
      ]:
        variance = np.max(layer.get_weights()[-1])
        if variance > max_variance:
          max_variance = variance

    if max_variance > 32 and self.learning_rate_factor < 100:
      learning_rate = K.get_value(self.model.optimizer.learning_rate)
      self.learning_rate_factor /= 2.0
      print("***** max_variance is {} / lr is {} *****".format(
          max_variance, learning_rate))
      K.eval(K.update(
          self.model.optimizer.learning_rate, learning_rate / 2.0
      ))

lra = LearningRateAdjuster()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape + (1,)).astype("float32")
x_test = x_test.reshape(x_test.shape + (1,)).astype("float32")

x_train /= 256.0
x_test /= 256.0

print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

print(y_train[0:10])

y_train = to_categorical(y_train, NB_CLASSES)
y_test = to_categorical(y_test, NB_CLASSES)

x = x_in = Input(x_train.shape[1:], name="input")
#x = QActivation("quantized_relu_po2(4,1)", name="acti")(x)
x = QConv2D(
    128, (3, 3),
    strides=1,
    kernel_quantizer=ternary(threshold=THRESHOLD), #quantized_po2(4, 1),
    bias_quantizer=quantized_bits(4,2,0) if not WITH_BN else None,
    bias_range=4 if not WITH_BN else None,
    use_bias=not WITH_BN,
    name="conv2d_0_m")(x)
if WITH_BN:
  x = QBatchNormalization(
      gamma_quantizer=quantized_relu_po2(4,8),
      variance_quantizer=quantized_relu_po2(6),
      beta_quantizer=quantized_po2(4, 4),
      gamma_range=8,
      beta_range=4,
      name="bn0")(x)
x = QActivation("quantized_relu(3,1)", name="act0_m")(x)
x = MaxPooling2D(2, 2, name="mp_0")(x)
x = QConv2D(
    256, (3, 3),
    strides=1,
    kernel_quantizer=ternary(threshold=THRESHOLD), #quantized_bits(2,0,1),
    bias_quantizer=quantized_bits(4,2,1) if not WITH_BN else None,
    bias_range=4 if not WITH_BN else None,
    use_bias=not WITH_BN,
    name="conv2d_1_m")(x)
if WITH_BN:
  x = QBatchNormalization(
      gamma_quantizer=quantized_relu_po2(4,8),
      variance_quantizer=quantized_relu_po2(6),
      beta_quantizer=quantized_po2(4, 4),
      gamma_range=8,
      beta_range=4,
      name="bn1")(x)
x = QActivation("quantized_relu(3,1)", name="act1_m")(x)
x = MaxPooling2D(2, 2, name="mp_1")(x)
x = QConv2D(
    128, (3, 3),
    strides=1,
    kernel_quantizer=ternary(threshold=THRESHOLD), #quantized_bits(2,0,1),
    bias_quantizer=quantized_bits(4,2,1) if not WITH_BN else None,
    bias_range=4 if not WITH_BN else None,
    use_bias=not WITH_BN,
    name="conv2d_2_m")(x)
if WITH_BN:
  x = QBatchNormalization(
      gamma_quantizer=quantized_relu_po2(4,8),
      variance_quantizer=quantized_relu_po2(6),
      beta_quantizer=quantized_po2(4, 4),
      gamma_range=8,
      beta_range=4,
      name="bn2")(x)
x = QActivation("quantized_relu(3,1)", name="act2_m")(x)
x = MaxPooling2D(2, 2, name="mp_2")(x)
x = Flatten()(x)
x = QDense(
    NB_CLASSES,
    kernel_quantizer=quantized_ulaw(4, 0, 1),
    bias_quantizer=quantized_bits(4, 0, 1),
    name="dense")(
        x)
x = Activation("softmax", name="softmax")(x)

model = Model(inputs=[x_in], outputs=[x])
model.summary()

model.compile(
    loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])


if TRAIN:
  history = model.fit(
      x_train, y_train, batch_size=BATCH_SIZE,
      epochs=NB_EPOCH, initial_epoch=1, verbose=VERBOSE,
      validation_split=VALIDATION_SPLIT,
      callbacks=[]) #lra])

  outputs = []
  output_names = []

  for layer in model.layers:
    if layer.__class__.__name__ in [
        "QActivation", "QBatchNormalization", "Activation", "QDense",
        "QConv2D", "QDepthwiseConv2D"
    ]:
      output_names.append(layer.name)
      outputs.append(layer.output)

  model_debug = Model(inputs=[x_in], outputs=outputs)

  outputs = model_debug.predict(x_train)

  print("{:30} {: 8.4f} {: 8.4f}".format(
      "input", np.min(x_train), np.max(x_train)))

  for n, p in zip(output_names, outputs):
    print("{:30} {: 8.4f} {: 8.4f}".format(n, np.min(p), np.max(p)), end="")
    layer = model.get_layer(n)
    for i, weights in enumerate(layer.get_weights()):
      if layer.get_quantizers()[i]:
        weights = K.eval(layer.get_quantizers()[i](K.constant(weights)))
      print(" ({: 8.4f} {: 8.4f})".format(np.min(weights), np.max(weights)),
            end="")
    print("")

  score = model.evaluate(x_test, y_test, verbose=False)
  print("Test score:", score[0])
  print("Test accuracy:", score[1])

print_qstats(model)
