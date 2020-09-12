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
"""Tests qcore model with BinaryToThermometer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow.keras.backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import numpy as np

from qkeras import *

np.random.seed(42)

NB_EPOCH = 20
BATCH_SIZE = 32
VERBOSE = 1
NB_CLASSES = 10
OPTIMIZER = Adam(lr=0.0001)
N_HIDDEN = 100
VALIDATION_SPLIT = 0.1

T_CLASSES = 256
T_WITH_RESIDUE = 0

(x_train, y_train), (x_test, y_test) = mnist.load_data()

RESHAPED = 784

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

if T_CLASSES == 1:
  x_train /= 256.0
  x_test /= 256.0

print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

print(y_train[0:10])

# x_train = x_train[0:1000]
# y_train = y_train[0:1000]
# x_test = x_test[0:100]
# y_test = y_test[0:100]

y_train = to_categorical(y_train, NB_CLASSES)
y_test = to_categorical(y_test, NB_CLASSES)

# we ran out of memory here, so we split x_train/x_test into smaller groups

x = x_in = Input(
    x_train.shape[1:-1] + (T_CLASSES,), name="input")

# Number is represented as 1.bbb, where number of bits of bbb is
# log2(256/T_CLASSES) if T_WITH_RESIDUE == 1

bits = (
    (T_WITH_RESIDUE == 1) * int(np.ceil(np.log2(256/T_CLASSES))) +
    (T_CLASSES > 1)
)

print("Input quantizer: quantized_relu({},{})".format(bits, int(T_CLASSES > 1)))
x = QActivation("quantized_relu({},{})".format(bits, int(T_CLASSES > 1)))(x)
x = QConv2D(
    64, (3, 3), strides=1, padding="same",
    kernel_quantizer=quantized_po2(4,1),
    bias_quantizer=quantized_bits(4,2,1),
    bias_range=4,
    name="conv2d_0_m")(x)
x = QActivation("quantized_relu(4,0)", name="act0_m")(x)
x = MaxPooling2D(2,2,name="mp_0")(x)
x = QConv2D(
    32, (3, 3), strides=1, padding="same",
    kernel_quantizer=stochastic_ternary(),
    bias_quantizer=quantized_bits(8,5,1),
    bias_range=32,
    name="conv2d_1_m")(x)
x = QActivation("quantized_relu(4,0)", name="act1_m")(x)
x = MaxPooling2D(2,2,name="mp_1")(x)
x = QConv2D(
    16, (3, 3), strides=1, padding="same",
    kernel_quantizer=quantized_bits(4,0,1),
    bias_quantizer=quantized_bits(8,5,1),
    bias_range=32,
    name="conv2d_2_m")(x)
x = QActivation("quantized_relu(6,2)", name="act2_m")(x)
x = MaxPooling2D(2,2,name="mp_2")(x)
x = Flatten()(x)
x = QDense(NB_CLASSES, kernel_quantizer=quantized_bits(4,0,1),
           bias_quantizer=quantized_bits(4,0,1),
           name="dense2")(x)
x = Activation("softmax", name="softmax")(x)

model = Model(inputs=[x_in], outputs=[x])
model.summary()

model.compile(
    loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])

outputs = []
output_names = []

for layer in model.layers:
  if layer.__class__.__name__ in ["QActivation", "Activation",
                                  "QDense", "QConv2D", "QDepthwiseConv2D"]:
    output_names.append(layer.name)
    outputs.append(layer.output)

model_debug = Model(inputs=[x_in], outputs=outputs)

batch_size = 1000 * BATCH_SIZE
n_batches = x_train.shape[0] // batch_size

if T_CLASSES > 1:
  x_test = BinaryToThermometer(x_test, T_CLASSES, 256, T_WITH_RESIDUE)

if int(os.environ.get("TRAIN", 0)):

  for i in range(NB_EPOCH):
    for b in range(n_batches):

      min_b = b * batch_size
      max_b = (b + 1) * batch_size
      if max_b > x_train.shape[0]:
        max_b = x_train.shape[0]

      if T_CLASSES > 1:
        x = BinaryToThermometer(
            x_train[min_b:max_b], T_CLASSES, 256, T_WITH_RESIDUE)
      else:
        x = x_train[min_b:max_b]

      history = model.fit(
          x, y_train[min_b:max_b], batch_size=BATCH_SIZE,
          epochs=i+1, initial_epoch=i, verbose=VERBOSE,
          validation_split=VALIDATION_SPLIT)

  if T_CLASSES > 1:
    x = BinaryToThermometer(x_train[0:100], T_CLASSES, 256, T_WITH_RESIDUE)
  else:
    x = x_train[0:100]

  outputs = model_debug.predict(x)

  print("{:30} {: 8.4f} {: 8.4f}".format("input", np.min(x), np.max(x)))
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

print_qstats(model)

acc = analyze_accumulator_from_sample(model, x_test, mode="sampled")

print(acc)


