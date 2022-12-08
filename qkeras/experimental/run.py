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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from args import ParserArgs

from dmnist import UseNetwork as DMUseNetwork
from cmnist import UseNetwork as CMUseNetwork

from lo import *
import numpy as np

def GetQuantizedType(mode):
  values = mode.split(",")
  values = [int(v) for v in values]

  return values

if __name__ == "__main__":

  model_dict = {
      "dmnist": DMUseNetwork,
      "cmnist": CMUseNetwork
  }

  args = ParserArgs()

  if args.model not in ["dmnist", "cmnist"]:
    exit(1)

  model, x_train, x_test = model_dict[args.model](
      args.weight_file, load_weights=args.load_weight)

  if args.logic_optimize:
    # i_dict = get_quantized_po2_dict(4,1,0)
    i_dict = get_quantized_bits_dict(
        *GetQuantizedType(args.qi),
        mode="bin" if args.use_pla or args.binary else "dec")
    o_dict = get_quantized_bits_dict(
        *GetQuantizedType(args.qo),
        mode="bin" if args.use_pla else "dec")

    print("... generating table with {} entries".format(x_train.shape[0]))

    strides, kernel, padding = model_to_receptive_field(
        model, args.i_name, args.o_name)

    if args.model in ["cmnist"]:

      files = optimize_conv2d_logic(
          model, args.i_name, args.o_name, x_train,
          i_dict, o_dict, output_group=args.output_group,
          kernel=kernel[0], strides=strides[0], padding=padding[0],
          samples=int(args.sample) if args.sample else x_train.shape[0],
          randomize=args.conv_sample, generate_pla=args.use_pla, prefix="results")
    else:

      files = optimize_dense_logic(
          model, args.i_name, args.o_name, x_train,
          i_dict, o_dict, output_group=args.output_group,
          samples=int(args.sample) if args.sample else x_train.shape[0],
          generate_pla=args.use_pla)

    if args.run_abc and args.use_pla:
      run_abc_optimizer(files)
    elif args.run_rf:
      flags = {
          "n_trees": args.n_trees,
          "max_bits": args.max_bits,
          "is_regressor": args.is_regressor,
          "n_features": args.n_features,
          "max_depth": args.max_depth,
          "sample_size": args.sample_size
      }

      run_rf_optimizer(files, flags)

    if args.model in ["cmnist"]:
      optimize_conv2d_logic(
          model, args.i_name, args.o_name, x_test,
          i_dict, o_dict, output_group=args.output_group,
          kernel=kernel[0], strides=strides[0], padding=padding[0],
          samples=int(args.sample) if args.sample else x_train.shape[0],
          randomize=args.conv_sample, generate_pla=args.use_pla, prefix="test")
