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
"""Implements random forest or logic otimizer function."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing as mp
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
import warnings

import numpy as np
import six

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from .compress import Compressor
from .generate_rf_code import gen_random_forest
from .table import load


def file_compress(fin, fout):
  """Compresses table using hash set."""
  c = Compressor()
  n_lines = 0
  for line in open(fin):
    n_lines += 1
    line = line.strip()
    c.add_entry(line)

  f = open(fout, "w")
  n_compressed = 0
  for line in c():
    n_compressed += 1
    f.write(line + "\n")
  f.close()
  print("... random forrest for {} reduced from {} to {} entries".format(
      os.path.basename(fin), n_lines, n_compressed))


def mp_rf_optimizer_func(fn_tuple):
  """Executes in parallel creation of random forrest creation."""

  fn, flags, file_suffix = fn_tuple

  n_trees = flags["n_trees"]
  is_regressor = flags["is_regressor"]
  sample_size = flags["sample_size"]
  n_features = flags["n_features"]
  max_depth = flags["max_depth"]

  if not file_suffix:
    file_suffix = "none"

  path_split = fn.split("/")
  path = "/".join(path_split[:-1]) + "/"
  fn_split = path_split[-1].split(".")
  # o_file = path + ".".join(fn_split[0:-2] + [fn_split[-1]])
  cv_file = path + ".".join(fn_split[0:-2] + [file_suffix])
  rfb_file = path + ".".join(fn_split[0:-2] + ["rb", "bin"])

  # let's compress the table first to make the job easier for random forest.
  # compression can usually achieve a ratio of 50x or more.

  # compress(fn, o_file)
  train = load(fn)

  n_features = "auto" if not n_features else float(n_features)

  # min_size = 1

  if max_depth:
    max_depth = int(max_depth)

  print("... creating random forrest for " + os.path.basename(fn) + " with " +
        str(sample_size) + " samples")

  if is_regressor:
    rf = RandomForestRegressor(
        n_estimators=n_trees,
        max_depth=max_depth,
        # min_samples_split=2,
        # min_samples_leaf=min_size,
        max_features=n_features,
        # max_leaf_nodes=100,
        # oob_score=True,
        # warm_start=True,
        bootstrap=True,
        random_state=42,
        n_jobs=1)
  else:
    rf = RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=max_depth,
        # min_samples_split=2,
        # min_samples_leaf=min_size,
        max_features=n_features,
        # max_leaf_nodes=100,
        # oob_score=True,
        # warm_start=True,
        bootstrap=True,
        random_state=42,
        n_jobs=1)

  if sample_size and train.shape[0] >= 10000:
    sample_size = int(sample_size)
    np.random.seed(42)
    idx = np.random.choice(train.shape[0], train.shape[0], replace=False)

    x = train[idx[sample_size:], 0:-1]
    y = train[idx[sample_size:], -1]

    x_test = train[idx[0:sample_size], 0:-1]
    y_test = train[idx[0:sample_size], -1]
  else:
    x = train[:, 0:-1]
    y = train[:, -1]

    x_test = x
    y_test = y

  estimators = []
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    rf.fit(x, y)

  func_name = fn_split[0]

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

  rf.bits = bits
  rf.is_neg = is_neg
  rf.o_bits = o_bits
  rf.o_is_neg = o_is_neg

  code = gen_random_forest(
      rf, func_name, bits, is_neg, o_bits, o_is_neg,
      is_regressor=is_regressor, is_top_level=False,
      is_cc=file_suffix == "cc")

  open(cv_file, "w").write("\n".join(code))

  p = 1.0 * np.round(rf.predict(x_test))

  dy = np.max(train[:, -1]) - np.min(train[:, -1])

  error = np.sum(np.abs(y_test - p)) / (1.0 * p.shape[0] * dy)
  score = np.sum(y_test == p) / p.shape[0]

  print("y:", np.max(y_test), y_test[0:30].astype(np.int32))
  print("p:", np.max(p), p[0:30].astype(np.int32))

  print("... model {} with score of {:.2f}% and error of {:.2f}%".format(
      func_name, 100.0*score, 100.0*error))

  print("... saving model in {}".format(rfb_file))
  pickle.dump(rf, open(rfb_file, "wb"))
  return rfb_file


def mp_abc_optimizer_func(fn):
  """Performs espresso and abc optimization on a single espresso input."""

  fn_split = fn.split(".")
  o_file = ".".join(fn_split[0:-2] + [fn_split[-1]])
  v_file = ".".join(fn_split[0:-2] + ["v"])
  b_file = ".".join(fn_split[0:-2] + ["blif"])

  print("...running espresso in " + fn)

  espresso_flags = os.environ.get("ESPRESSO_FLAGS", "-Dexpand")

  cmd = "espresso {} {} > {}".format(fn, espresso_flags, o_file)

  output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

  output = output.strip()
  if output:
    print(output)
    sys.stdout.flush()

  # check if network is empty

  for line in open(o_file):
    line = line.strip()
    if line[0:2] == ".p":
      terms = int(line[2:])
      # empty : espresso optimized away all the logic
      if terms == 0:
        shutil.copyfile(fn, o_file)
      break

  print("...running abc in " + o_file)

  abc_flags = os.environ.get("ABC_FLAGS", "")

  abc_flags_list = abc_flags.split(";") if abc_flags else []

  abc_cmds_list = (
      ["read_pla " + o_file] + abc_flags_list +
      ["strash",
       "dc2",
       "strash",
       "if -K 3",
       "write_verilog " + v_file,
       "write_blif " + b_file
       ])

  abc_cmds = ";".join(abc_cmds_list)

  cmd = "abc -c '" + abc_cmds + "'"

  output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

  output = output.strip()
  if output:
    print(output)
    sys.stdout.flush()

  print("...generated " + v_file)


def run_abc_optimizer(files):
  """Implements logic optimizer using espresso/abc."""

  # intel processors sometimes return number of threads, not processors

  cpus = mp.cpu_count() // 2

  start_time = time.time()
  pool = mp.Pool(cpus)
  pool.map(mp_abc_optimizer_func, files)
  pool.close()
  print("Optimizer ran in {} seconds.".format(time.time() - start_time))


def run_rf_optimizer(files, flags, file_suffix="cc"):
  """Implements random forest main optimizer."""

  # intel processors sometimes return number of threads, not processors

  cpus = mp.cpu_count() // 2

  start_time = time.time()
  pool = mp.Pool(cpus)
  pool.map(mp_rf_optimizer_func, zip(
      files, [flags]*len(files), [file_suffix]*len(files)))
  pool.close()
  print("Optimizer ran in {} seconds.".format(time.time() - start_time))

  # generates header file

  # .../.../.../conv2d_0_m.csv/conv2d_0_m_0.csv
  #
  # returns conv2d_0_m for module_name

  module_name = files[0].split("/")[-2].split(".")[0]

  path_split = files[0].split("/")
  path = "/".join(path_split[:-1]) + "/"
  fn_split = path_split[-1].split(".")
  rfb_file = path + ".".join(fn_split[0:-2] + ["rb", "bin"])

  rf = pickle.load(open(rfb_file, "rb"))

  f = open(path + module_name + "." + file_suffix, "w")

  if file_suffix == "cc":
    f.write("#include <ac_int.h>\n\n")

  modules = []

  for fn in files:
    path_split = fn.split("/")
    path = "/".join(path_split[:-1]) + "/"
    fn_split = path_split[-1].split(".")
    v_file = ".".join(fn_split[0:-2] + [file_suffix])

    func_name = fn_split[0]

    if file_suffix == "v":
      f.write("'include \"" + v_file + "\"\n")
    else:
      f.write("#include \"" + v_file + "\"\n")

    modules.append(func_name)

  f.write("\n\n")

  if file_suffix == "v":
    f.write("module " + module_name + "(")
    f.write("input [" + str(np.sum(rf.bits)-1) + ":0] in, ")
    o_sign = " signed " if rf.o_is_neg else ""
    f.write("output " + o_sign + "[" + str(len(modules)*rf.o_bits-1) +
            ":0] out);\n")
  else:
    f.write("void " + module_name + "(")
    f.write("ac_int<" + str(np.sum(rf.bits)) + ",false> in, ")
    f.write("ac_int<" + str(len(modules)*rf.o_bits) + "," +
            ("true" if rf.o_is_neg else "false") +
            "> &out)\n")
    f.write("{\n")

  for o in range(len(modules)):
    if file_suffix == "v":
      f.write("  wire " + ("signed " if rf.o_is_neg else "") +
              "[" + str(rf.bits[-1]-1) + ":0] "
              "o_" + str(o) + ";\n")
      f.write("  " + modules[o] + "(in, o_" + str(o) + ");\n")
      f.write("  assign out[" + str(rf.o_bits*(o+1)-1) + ":" +
              str(rf.bits[-1]*o) + "] = o_" + str(o) + ";\n")
    else:
      f.write("  ac_int<" + str(rf.o_bits) + "," +
              ("true" if rf.o_is_neg else "false") +
              "> o_" + str(o) + "; " + modules[o] +
              "(in, o_" + str(o) + "); out.set_slc<" +
              str(rf.o_bits) + ">(" +
              str(rf.o_bits*o) + "," +
              "o_" + str(o) + ");\n")

  if file_suffix == "cc":
    f.write("}")

  f.close()
