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
"""Tests qdense model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import numpy as np

from qkeras import print_qstats
from qkeras import QActivation
from qkeras import QDense
from qkeras import quantized_bits
from qkeras import ternary


np.random.seed(42)
OPTIMIZER = Adam()
NB_EPOCH = 1
BATCH_SIZE = 32
VERBOSE = 1
NB_CLASSES = 10
N_HIDDEN = 100
VALIDATION_SPLIT = 0.1
RESHAPED = 784


def QDenseModel(weights_f, load_weights=False):
  """Construct QDenseModel."""

  x = x_in = Input((RESHAPED,), name="input")
  x = QActivation("quantized_relu(4)", name="act_i")(x)
  x = QDense(N_HIDDEN, kernel_quantizer=ternary(),
             bias_quantizer=quantized_bits(4, 0, 1), name="dense0")(x)
  x = QActivation("quantized_relu(2)", name="act0")(x)
  x = QDense(
      NB_CLASSES,
      kernel_quantizer=quantized_bits(4, 0, 1),
      bias_quantizer=quantized_bits(4, 0, 1),
      name="dense2")(
          x)
  x = Activation("softmax", name="softmax")(x)

  model = Model(inputs=[x_in], outputs=[x])
  model.summary()
  model.compile(loss="categorical_crossentropy",
                optimizer=OPTIMIZER, metrics=["accuracy"])

  if load_weights and weights_f:
    model.load_weights(weights_f)

  print_qstats(model)
  return model


def UseNetwork(weights_f, load_weights=False):
  """Use DenseModel.

  Args:
    weights_f: weight file location.
    load_weights: load weights when it is True.
  """
  model = QDenseModel(weights_f, load_weights)

  batch_size = BATCH_SIZE
  (x_train_, y_train_), (x_test_, y_test_) = mnist.load_data()

  x_train_ = x_train_.reshape(60000, RESHAPED)
  x_test_ = x_test_.reshape(10000, RESHAPED)
  x_train_ = x_train_.astype("float32")
  x_test_ = x_test_.astype("float32")

  x_train_ /= 255
  x_test_ /= 255

  print(x_train_.shape[0], "train samples")
  print(x_test_.shape[0], "test samples")

  y_train_ = to_categorical(y_train_, NB_CLASSES)
  y_test_ = to_categorical(y_test_, NB_CLASSES)

  if not load_weights:
    model.fit(
        x_train_,
        y_train_,
        batch_size=batch_size,
        epochs=NB_EPOCH,
        verbose=VERBOSE,
        validation_split=VALIDATION_SPLIT)

    if weights_f:
      model.save_weights(weights_f)

  score = model.evaluate(x_test_, y_test_, verbose=VERBOSE)
  print_qstats(model)
  print("Test score:", score[0])
  print("Test accuracy:", score[1])


def ParserArgs():
  parser = argparse.ArgumentParser()
  parser.add_argument("-l", "--load_weight", default="0",
                      help="""load weights directly from file.
                            0 is to disable and train the network.""")
  parser.add_argument("-w", "--weight_file", default=None)
  a = parser.parse_args()
  return a


if __name__ == "__main__":
  args = ParserArgs()
  lw = False if args.load_weight == "0" else True
  UseNetwork(args.weight_file, load_weights=lw)
