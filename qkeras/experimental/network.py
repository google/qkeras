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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from args import ParserArgs

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from lo import *
import numpy as np

from qkeras import print_qstats
from qkeras import QActivation
from qkeras import QConv2D
from qkeras import QDense
from qkeras import quantized_bits
from qkeras import ternary


np.random.seed(42)
OPTIMIZER = Adam(lr=0.002)
NB_EPOCH = 10
BATCH_SIZE = 32
VERBOSE = 1
NB_CLASSES = 10
N_HIDDEN = 100
VALIDATION_SPLIT = 0.1
RESHAPED = 784


def QDenseModel(weights_f, load_weights=False):
  """Construct QDenseModel."""

  x = x_in = Input((28*28,), name="input")
  x = QActivation("quantized_relu(2)", name="act_i")(x)

  x = Dense(100, name="d0")(x)
  x = BatchNormalization(name="bn0")(x)

  x = QActivation("quantized_relu(2)", name="act0_m")(x)

  x = Flatten(name="flatten")(x)

  x = QDense(
      NB_CLASSES,
      kernel_quantizer=quantized_bits(4, 0, 1),
      bias_quantizer=quantized_bits(4, 0, 1),
      name="dense2")(x)
  x = Activation("softmax", name="softmax")(x)

  model = Model(inputs=[x_in], outputs=[x])
  model.summary()
  model.compile(loss="categorical_crossentropy",
                optimizer=OPTIMIZER, metrics=["accuracy"])

  if load_weights and weights_f:
    model.load_weights(weights_f)

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

  x_train_ = x_train_.reshape(60000, 28*28)
  x_test_ = x_test_.reshape(10000, 28*28)
  x_train_ = x_train_.astype("float32")
  x_test_ = x_test_.astype("float32")

  x_train_ /= 256.
  x_test_ /= 256.

  # x_train_ = 2*x_train_ - 1.0
  # x_test_ = 2*x_test_ - 1.0

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

  score = model.evaluate(x_test_, y_test_, verbose=False)
  print("Test score:", score[0])
  print("Test accuracy:", score[1])

  return model, x_train_


if __name__ == "__main__":
  args = ParserArgs()
  model, x_train = UseNetwork(args.weight_file, load_weights=args.load_weight)

  if args.logic_optimize:
    i_dict = get_quantized_bits_dict(2,0,0,mode="bin" if args.use_pla else "dec")
    o_dict = get_quantized_bits_dict(2,0,0,mode="bin" if args.use_pla else "dec")

    print("... generating table with {} entries".format(x_train.shape[0]))

    files = optimize_dense_logic(
        model, args.i_name, args.o_name, x_train,
        i_dict, o_dict, output_group=args.output_group,
        samples=int(args.sample) if args.sample else x_train.shape[0],
        generate_pla=args.use_pla)

    if args.use_pla and args.run_abc:
      run_optimizer(args.i_name, files)

