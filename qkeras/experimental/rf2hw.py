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
"""Generates expressions for random trees."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import warnings

from lo import *

import argparse
import pickle
import os
import sys
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

USE_REGRESSOR=int(os.environ.get("USE_REGRESSOR", 1))
N_ESTIMATORS=int(os.environ.get("N_ESTIMATORS", 3))
N_FEATURES=float(os.environ.get("N_FEATURES", 0.5))
MAX_DEPTH=int(os.environ.get("MAX_DEPTH", 20))

ROUND=int(os.environ.get("ROUND", 1))
BINARY=int(os.environ.get("BINARY",0))
SINGLE=int(os.environ.get("SINGLE",1))


def error(y,p):
  print("acc: ", 100*np.sum(y == p).astype(np.float32)/y.shape[0])
  #print("acc: ", 100*(np.sum(np.abs(y-p)).astype(np.float32)/y.shape[0]))


def ParseArguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("--train", default=False,
                      action="store_true",
                      help="train network")
  parser.add_argument("--synthesize", default=False,
                      action="store_true",
                      help="synthesize random tree")
  parser.add_argument("--with_tb", default=False,
                      action="store_true",
                      help="with testbench")
  parser.add_argument("--use_classifier", default=False,
                      action="store_true",
                      help="use classifier instead of regressor")
  parser.add_argument("--max_depth", default=None,
                      help="maximum depth of tree")
  parser.add_argument("--max_bits", default=1, type=int,
                      help="maximum number of bits")
  parser.add_argument("--n_features", default=None,
                      help="number of features to use per random tree")
  parser.add_argument("--n_trees", default=1, type=int,
                      help="number of random trees")
  parser.add_argument("--sample_size", default=None,
                      help="sample size of table for random tree")
  parser.add_argument("--rf_bin", default="rb.bin",
                      help="name of random forest binary file")
  parser.add_argument("--cc", default=False, action="store_true")
  parser.add_argument("table", help="table file in pla or csv format")
  a = parser.parse_args()
  return a


args = ParseArguments()

is_cc = args.cc

if args.train:
  if args.max_depth:
    os.environ["MAX_DEPTH"] = args.max_depth

  if args.n_features:
    os.environ["N_FEATURES"] = args.n_features

  if args.use_classifier:
    os.environ["USE_REGRESSOR"] = str(int(not args.use_classifier))

  if args.max_bits:
    os.environ["MAX_BITS"] = str(int(args.max_bits))

  flags = {
      "n_trees": args.n_trees,
      "max_bits": args.max_bits,
      "is_regressor": not args.use_classifier,
      "n_features": args.n_features,
      "max_depth": args.max_depth,
      "sample_size": args.sample_size
  }

  suffix = "cc" if is_cc else "v"

  mp_rf_optimizer_func(("./" + args.table, flags, suffix))

  fn = "./" + args.table
  path_split = fn.split("/")
  path = "/".join(path_split[:-1]) + "/"
  fn_split = path_split[-1].split(".")
  rfb_file = path + ".".join(fn_split[0:-2] + ["rb", "bin"])

else:
  rfb_file = args.rf_bin

rf = pickle.load(open(rfb_file,"rb"))
print(rf)

t = load(args.table)
x = t[:,0:-1]
y = t[:,-1]

p = rf.predict(x)

error(y, np.round(p))

print("y:", np.max(y), y[0:30].astype(np.int32))
print("p:", np.max(p), np.round(p)[0:30].astype(np.int32))

if args.synthesize:

  bits = np.ceil(
      np.log2(
          np.abs(
              np.amax(x, axis=0) -
              np.amin(x, axis=0) + 1))).astype(np.int32)
  is_neg = (np.amin(x, axis=0) < 0).astype(np.int8)

  o_bits = np.ceil(
      np.log2(
          np.abs(
              np.amax(y, axis=0) -
              np.amin(y, axis=0) + 1))).astype(np.int32)
  o_is_neg = (np.amin(y, axis=0) < 0).astype(np.int8)

  module = "test"

  # prune results
  #for i in range(len(rf.estimators_)):
  #  tree = rf.estimators_[i].tree_
  #  for n in range(tree.node_count):
  #    if tree.children_left[n] == tree.children_right[n]:
  #      tree.value[n] = np.clip(tree.value[n], 0, (1 << args.max_bits) - 1)

  code = gen_random_forest(
      rf, module, bits, is_neg, o_bits, o_is_neg,
      is_regressor=not args.use_classifier, is_top_level=True,
      is_cc=is_cc
  )

  p = rf.predict(x)

  error(y, np.round(p))

  print("y:", np.max(y), y[0:30].astype(np.int32))
  print("p:", np.max(p), np.round(p)[0:30].astype(np.int32))

  if args.with_tb:
    if is_cc:
      gen_testbench_cc(rf, module, bits, is_neg, o_bits, o_is_neg, x,
                       y, np.round(p), code)
    else:
      gen_testbench_sv(rf, module, bits, is_neg, o_bits, o_is_neg, x,
                       y, np.round(p), code)


  if is_cc:
    filename = "test.cc"
  else:
    filename = "test.v"

  with open(filename,"w") as f:
    f.write("\n".join(code))
    f.write("\n")

