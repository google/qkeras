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

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

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

  x = x_in = Input((28,28,1), name="input")
  x = QActivation("quantized_relu(2)", name="act_i")(x)

  x = Conv2D(32, (3, 3), strides=(2, 2), name="conv2d_0_m")(x)
  x = BatchNormalization(name="bn0")(x)
  #x = QConv2D(32, (3, 3), strides=(2, 2),
  #            kernel_quantizer=quantized_bits(4,0,1),
  #            bias_quantizer=quantized_bits(4,0,1),
  #            name="conv2d_0_m")(x)

  x = QActivation("quantized_relu(2)", name="act0_m")(x)

  x = Conv2D(64, (3, 3), strides=(2, 2), name="conv2d_1_m")(x)
  x = BatchNormalization(name="bn1")(x)
  x = QActivation("quantized_relu(2)", name="act1_m")(x)

  x = Conv2D(64, (3, 3), strides=(2, 2), name="conv2d_2_m")(x)
  x = BatchNormalization(name="bn2")(x)

  x = QActivation("quantized_relu(2)", name="act2_m")(x)

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

  x_train_ = x_train_.reshape(60000, 28, 28, 1)
  x_test_ = x_test_.reshape(10000, 28, 28, 1)
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

  return model, x_train_, x_test_


def ParserArgs():
  parser = argparse.ArgumentParser()
  parser.add_argument("-o", "--logic_optimize", default=False,
                      action="store_true",
                      help="optimize network.")
  parser.add_argument("-l", "--load_weight", default=False,
                      action="store_true",
                      help="load weights directly from file.")
  parser.add_argument("-w", "--weight_file", default=None)
  parser.add_argument("--output_group", type=int, default=1)
  parser.add_argument("--kernel", default=None, type=int,
                      help="kernel if more complex layer")
  parser.add_argument("--strides", default=None, type=int,
                      help="stride if more complex layer")
  parser.add_argument("--padding", default=None,
                      help="padding if more complex layer")
  parser.add_argument("--conv_sample", default=None, type=int,
                      help="number of samples within image for conv layer")
  parser.add_argument("--sample", default=None,
                      help="number of training samples")
  parser.add_argument("--use_pla", default=False,
                      action="store_true", help="use pla table format")
  parser.add_argument("--binary", default=False,
                      action="store_true", help="use binary inputs")
  parser.add_argument("--i_name", default=None,
                      help="input layer name")
  parser.add_argument("--o_name", default=None,
                      help="output layer name")
  parser.add_argument("--run_abc", default=False, action="store_true")
  parser.add_argument("--run_rf", default=False, action="store_true")
  parser.add_argument("--n_trees", default=3, type=int)
  a = parser.parse_args()
  return a


if __name__ == "__main__":
  args = ParserArgs()
  model, x_train, x_test = UseNetwork(args.weight_file, load_weights=args.load_weight)

  if args.logic_optimize:
    # i_dict = get_quantized_po2_dict(4,1,0)
    i_dict = get_quantized_bits_dict(
        2,0,0,mode="bin" if args.use_pla or args.binary else "dec")
    o_dict = get_quantized_bits_dict(
        2,0,0,mode="bin" if args.use_pla else "dec")

    print("... generating table with {} entries".format(x_train.shape[0]))

    strides, kernel, padding = model_to_receptive_field(
        model, args.i_name, args.o_name)

    files = optimize_conv2d_logic(
        model, args.i_name, args.o_name, x_train,
        i_dict, o_dict, output_group=args.output_group,
        kernel=kernel[0], strides=strides[0], padding=padding[0],
        samples=int(args.sample) if args.sample else x_train.shape[0],
        randomize=args.conv_sample, generate_pla=args.use_pla, prefix="results")

    if args.run_abc and args.use_pla:
      run_abc_optimizer(files)
    elif args.run_rf:
      run_rf_optimizer(files, args.n_trees)

    optimize_conv2d_logic(
        model, args.i_name, args.o_name, x_test,
        i_dict, o_dict, output_group=args.output_group,
        kernel=kernel[0], strides=strides[0], padding=padding[0],
        samples=int(args.sample) if args.sample else x_train.shape[0],
        randomize=args.conv_sample, generate_pla=args.use_pla, prefix="test")

    # optimize_conv2d_dt(model, "conv2d_1_m", "act1_m", i_dict, o_dict,
    #                    x_train, single_output=args.single_output,
    #                    samples=5000, pixels=2)
