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
"""Tests qcore model with po2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from collections import defaultdict

import tensorflow.keras.backend as K
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import to_categorical
import numpy as np

from qkeras import *

np.random.seed(42)

NB_EPOCH = 50
BATCH_SIZE = 64
VERBOSE = 1
NB_CLASSES = 10
OPTIMIZER = Adam(lr=0.0001)
VALIDATION_SPLIT = 0.1

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train /= 255.0
x_test /= 255.0

print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

print(y_train[0:10])

y_train = to_categorical(y_train, NB_CLASSES)
y_test = to_categorical(y_test, NB_CLASSES)

x = x_in = Input(x_train.shape[1:], name="input")
x = QActivation("quantized_relu_po2(4,4)", name="acti")(x)
x = QConv2D(
    128, (3, 3),
    strides=1,
    kernel_quantizer=quantized_po2(4, 1),
    bias_quantizer=quantized_po2(4, 4),
    bias_range=4,
    name="conv2d_0_m")(
        x)
x = QActivation("ternary()", name="act0_m")(x)
x = MaxPooling2D(2, 2, name="mp_0")(x)
x = QConv2D(
    256, (3, 3),
    strides=1,
    kernel_quantizer=quantized_po2(4, 1),
    bias_quantizer=quantized_po2(4, 4),
    bias_range=4,
    name="conv2d_1_m")(
        x)
x = QActivation("quantized_relu(6,2)", name="act1_m")(x)
x = MaxPooling2D(2, 2, name="mp_1")(x)
x = QConv2D(
    128, (3, 3),
    strides=1,
    kernel_quantizer=quantized_bits(4, 0, 1),
    bias_quantizer=quantized_bits(4, 0, 1),
    name="conv2d_2_m")(
        x)
x = QActivation("quantized_relu(4,2)", name="act2_m")(x)
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

if int(os.environ.get("TRAIN", 0)):

  history = model.fit(
      x_train, y_train, batch_size=BATCH_SIZE,
      epochs=NB_EPOCH, initial_epoch=1, verbose=VERBOSE,
      validation_split=VALIDATION_SPLIT)

  outputs = []
  output_names = []

  for layer in model.layers:
    if layer.__class__.__name__ in [
        "QActivation", "Activation", "QDense", "QConv2D", "QDepthwiseConv2D"
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
      weights = K.eval(layer.get_quantizers()[i](K.constant(weights)))
      print(" ({: 8.4f} {: 8.4f})".format(np.min(weights), np.max(weights)),
            end="")
      print("")

  score = model.evaluate(x_test, y_test, verbose=VERBOSE)
  print("Test score:", score[0])
  print("Test accuracy:", score[1])

model.summary()

print_qstats(model)
