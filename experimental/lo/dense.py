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

"""Implements dense (?, features) fancing input layer optimization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing as mp
import os
import shutil

from .compress import Compressor
import numpy as np
import six
from tensorflow.keras.models import Model

DEBUG = int(os.getenv("DEBUG", 0))

OG_IS_SYMBOLIC = 0


def parallel_index_table(
    p, ni, size, i_dict, o_dict, generate_pla):
  """Processes the table in parallel and use espresso to optimize it."""

  print("... indexing table from {} to {} ({} => {})".format(
      ni, ni+size, p[0].shape, p[1].shape))

  table_ins = []
  table_ous = []

  table_set = Compressor(hash_only_input=True)

  if DEBUG:
    table_set_line = {}

  for n in range(size):

    i_values = p[0][n].flatten()
    o_values = p[1][n].flatten()

    # if we generate a pla entry, we care about a list of
    # bits. Otherwise, we care about a list of floating point
    # values.

    table_i = "".join([i_dict[v] for v in i_values])
    table_o = "".join([o_dict[v] for v in o_values])

    if generate_pla:
      table_s = "".join([str(v) for v in table_i])
      bit_str = table_s
    else:
      table_s = ",".join([str(v) for v in table_i])
      table_i = table_s
      bit_str = "".join(str(i_dict[v]) for v in i_values)
    is_table_zero = bit_str != "0"*len(bit_str)

    if table_set.has_entry(table_s) and not is_table_zero:

      # if table is already stored, we do not store it again.
      # from time to time, we may want to check if we have found
      # diverging output values.

      if DEBUG:

        (table_o_old, old_n) = table_set_line[table_s]

        if table_o != table_o_old:
          print("contradicting outputs n={} old_n={} out_p={} out={}".format(
              n, old_n, table_o_old, table_o))
          print(" I:", table_s)
          print(" I:", i_values)
          print("<<<", table_o_old)
          print(">>>", table_o)
          return (None, None)

      continue

    # these are unique table entries

    table_ins.append(table_i)
    table_ous.append(table_o)

    # we store this information in order to be able to debug
    # and discard information.

    table_set.add_entry(table_s)

    if DEBUG:
      table_set_line[table_s] = (table_o, n)

  print("... indexing table from {} to {} completed".format(ni, ni+size))

  return (table_ins, table_ous)


def parallel_compress_output_table(
    filename, header, table_ins, table_ous, output_group, generate_pla,
    n_bits_og, o, o_bits):
  """Processes in parallel compression of table and writes it to a disk."""

  f = open(filename, "w")

  f.write("".join(header))

  c = Compressor()

  for n in range(len(table_ins)):
    for og in range(output_group):

      if output_group > 1:
        if generate_pla:
          if OG_IS_SYMBOLIC:
            og_l = ["0"] * n_bits_og
            og_l[n_bits_og - 1 - og] = "1"
            og_b = "".join(og_l)
            table_i_suffix = " " + og_b
          else:
            og_b = bin(og)[2:]
            table_i_suffix = " " + "0"*(n_bits_og - len(og_b)) + og_b
        else:
          table_i_suffix = "," + str(og)
      else:
        table_i_suffix = ""
      table_i = table_ins[n] + table_i_suffix
      table_o = table_ous[n][(o+og)*o_bits:(o+og+1)*o_bits]

      if generate_pla:
        c.add_entry(table_i + " " + table_o)
      else:
        c.add_entry(table_i + "," + str(table_o[0]))

  for line in c():
    f.write("{}\n".format(line[0]))

  if generate_pla:
    f.write(".e\n")
  f.close()


def optimize_dense_logic(
    model, i_name, o_name, x_train, i_dict, o_dict,
    output_group=1, samples=2000,
    generate_pla=True, prefix=""):

  """Generates table for logic synthesis for dense or flattened layer.

  Generates table in either espresso format or csv format to be optimized
  for logic synthesis.

  Arguments:
    model: Keras model
    i_name: name of convolutional layer (input to this layer must be
      quantized).
    o_name: name of quantized output layer.
    x_train: training set to be used to dump table.
    i_dict: dictionary of floating point values to encoding for inputs.
    o_dict: dictionary of floating point values to encoding for outputs.
    output_group: by default, we compute one PE per channel output. The user
      can override that by specifying how many output channels should be
      bundled into the same PE.
    samples: how many images from x_train should be sampled when generating the
      tables.
    generate_pla: if true, we generate table in pla format. Otherwise, we
      generate a csv file.
    prefix: prefix name to create a directory.
  Returns:
    list of files generated.
  """

  i_layer = model.get_layer(i_name)
  o_layer = model.get_layer(o_name)

  # resample inputs

  skip = min(2000, samples)

  indexes = np.array(range(x_train.shape[0]))
  np.random.shuffle(indexes)

  x_train = x_train[indexes[:samples]]

  outputs = []

  x = i_layer.input
  y = o_layer.output

  if not isinstance(x, list):
    x = [x]

  outputs = x + [y]

  mo = Model(inputs=model.inputs, outputs=outputs)
  p = mo.predict(x_train)

  # in csv mode, each entry has "1" value, for PLA,
  # we encode the floating point into multiple bits.

  if not generate_pla:
    i_bits = 1
    # i_dict = {v:v for v in i_dict.keys()}
  else:
    i_bits = len(six.next(six.itervalues(i_dict)))

  if not generate_pla:
    o_bits = 1
    # o_dict = {v:v for v in o_dict.keys()}
  else:
    o_bits = len(six.next(six.itervalues(o_dict)))

  print("inputs:")
  for i in range(len(x)):
    print(i, np.min(p[i]), np.max(p[i]))
  print("outputs:")
  print(0, np.min(p[-1]), np.max(p[-1]))

  o_size = y.shape[-1]
  i_size = p[0].shape[-1]

  if generate_pla:
    suffix = "pla"
  else:
    suffix = "csv"

  prefix = prefix + "/" if prefix else ""

  # lets try to remove the directory and create a new one

  try:
    shutil.rmtree(prefix + i_layer.name + "." + suffix)
  except OSError:
    pass

  try:
    os.makedirs(prefix + i_layer.name + "." + suffix)
  except OSError:
    pass

  print("...indexing inputs")

  # for each image in sampled x_train

  # on Intel processors, mp.cpu_count() returns number of threads

  number_of_processes = mp.cpu_count() // 2
  pool = mp.Pool(number_of_processes)

  results = []

  for n in range(0, x_train.shape[0], skip):

    res = pool.apply_async(
        parallel_index_table,
        args=((p[0][n:n+skip], p[1][n:n+skip]), n, skip, i_dict, o_dict,
              generate_pla))
    results.append(res)

  pool.close()
  pool.join()

  all_pools = [res.get(timeout=1) for res in results]

  table_ins = sum([ap[0] for ap in all_pools], [])
  table_ous = sum([ap[1] for ap in all_pools], [])

  # input and output size

  ni = len(table_ins[0])
  no = len(table_ous[0])

  print("... generating tables {} outputs, {} entries".format(
      o_size, len(table_ins)))

  # this step should be very fast

  files = []

  if OG_IS_SYMBOLIC:
    if output_group > 1:
      n_bits_og = output_group
    else:
      n_bits_og = 1
  else:
    if output_group == 2:
      n_bits_og = 1
    else:
      n_bits_og = int(np.ceil(np.log2(output_group)))

  # sometimes linux get very grumpy with too many files opened.
  # let's limit to 20.

  number_of_processes = min(20, mp.cpu_count() // 2)
  pool = mp.Pool(number_of_processes)

  for o in range(0, o_size, output_group):

    filename = "{}{}.{}/{}_{}.raw.{}".format(
        prefix, i_name, suffix, i_name, o, suffix)

    files.append(filename)

    header = []

    if generate_pla:
      header.append(".i {}\n".format(ni + n_bits_og))
      header.append(".o {}\n".format(no // o_size))
      header.append(".type fr\n")

      if OG_IS_SYMBOLIC and output_group > 1:
        header.append(".mv {} {} {} {}\n".format(
            3, ni, n_bits_og, no // o_size))

      # let's generate some labels

      header.append(".ob " + " ".join([
          "o_" + str(o) + "_" + str(o_bits - 1 - v)
          for v in range(o_bits)]) + "\n")

      i_names = []

      # name is i_<features>_bit

      assert ni == (i_size * i_bits)

      for feature in range(i_size):
        for bit in range(i_bits):
          i_names.append("i_{}_{}".format(
              feature, (i_bits - 1 - bit)))

      # if we are grouping multiple channels, these will be the inputs

      for c in range(n_bits_og):
        i_names.append("og_{}".format(n_bits_og - 1 - c))

      header.append(".ilb " + " ".join(i_names) + "\n")

    pool.apply_async(
        parallel_compress_output_table,
        args=((filename, header, table_ins, table_ous, output_group,
               generate_pla, n_bits_og, o, o_bits)))

  pool.close()
  pool.join()

  return files


