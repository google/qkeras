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
"""Implements convolutional (?, h, w, c) facing input layer optimization."""

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
from .utils import get_padding_value

DEBUG = int(os.getenv("DEBUG", 0))

OG_IS_SYMBOLIC = 0


def parallel_index_table(
    p, ni, size, idx_height, idx_width, i_dict, o_dict,
    kernel, strides, padding, generate_pla):
  """Processes the table in parallel and use espresso to optimize it."""

  print("... indexing table from {} to {} ({} => {})".format(
      ni, ni+size, p[0].shape, p[1].shape))

  table_ins = []
  table_ous = []

  table_set = Compressor(hash_only_input=True)

  if DEBUG:
    table_set_line = {}

  for n in range(size):

    # we need to traverse the outputs to compute the input coordinates

    for ho in idx_height:
      min_hi = strides[0]*ho - 2*padding[0]
      max_hi = strides[0]*ho - 2*padding[0] + kernel[0]

      if min_hi < 0 or max_hi > p[0].shape[0]:
        continue

      for wo in idx_width:
        min_wi = strides[1]*wo - 2*padding[1]
        max_wi = strides[1]*wo - 2*padding[1] + kernel[1]

        if min_wi < 0 or max_wi > p[0].shape[1]:
          continue

        i_values = p[0][n, min_hi:max_hi, min_wi:max_wi].flatten()

        # o_values has dimension (1, 1, C_O)

        o_values = p[1][n, ho, wo]

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
          bit_str = "".join(i_dict[v] for v in i_values)
        is_table_zero = bit_str != "0"*len(bit_str)

        if table_set.has_entry(table_s) and not is_table_zero:

          # if table is already stored, we do not store it again.
          # from time to time, we may want to check if we have found
          # diverging output values.

          if DEBUG:

            (table_o_old, (old_n, old_ho, old_wo)) = table_set_line[table_s]

            if table_o != table_o_old:
              print(
                  "contradicting outputs n={} old_n={} out_p={} out={}".format(
                      (n, ho, wo), (old_n, old_ho, old_wo), table_o_old,
                      table_o))
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
          table_set_line[table_s] = (table_o, (n, ho, wo))

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
            table_i_suffix = " " + "0" * (n_bits_og - len(og_b)) + og_b
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

  print("... file {} generated".format(filename))


def optimize_conv2d_logic(
    model, i_name, o_name, x_train,
    i_dict=None, o_dict=None,
    kernel=None, strides=None, padding=None,
    output_group=1, samples=2000,
    randomize=None, generate_pla=True, prefix=""):
  """Generates table for logic synthesis for conv2d or conv2d-like shape.

  Generates table in either espresso format or csv format to be optimized
  for logic synthesis. The parameters kernel, strides and padding usually
  do not require any values, unless we want to embed maxpooling layer or
  multiple convolutional layers between i_name and o_name. In that case,
  we require the user to compute the proper kernel, strides, and padding
  that will correspond to the combined layer, as Keras and tensorflow do not
  provide a way to compute the receptive field between two layers.

  Arguments:
    model: Keras model
    i_name: name of convolutional layer (input to this layer must be
      quantized).
    o_name: name of quantized output layer.
    x_train: training set to be used to dump table.
    i_dict: dictionary of floating point values to encoding for inputs.
    o_dict: dictionary of floating point values to encoding for outputs.
    kernel: kernel size, to be specified if we want to override convolution
      kernel.
    strides: strides, to be specified if we want to override first convolution
      strides.
    padding: padding, to be specified if we want to override first convolution
      padding.
    output_group: by default, we compute one PE per channel output. The user
      can override that by specifying how many output channels should be
      bundled into the same PE.
    samples: how many images from x_train should be sampled when generating the
      tables.
    randomize: if specified, it should be the number of coordinates within the
      same image we will use to derive the convolution table.
    generate_pla: if true, we generate table in pla format. Otherwise, we
      generate a csv file.
    prefix: prefix name to create directory.

  Returns:
    list of files generated.
  """

  # if no i_dict or no o_dict, we do not know how to encode, so we generate
  # csv file.

  if not i_dict or not o_dict:
    generate_pla = False

  # extract layer from i_name and o_name

  i_layer = model.get_layer(i_name)
  o_layer = model.get_layer(o_name)

  # if kernel is not specified, use the kernel size from i_layer

  if not kernel:
    kernel = i_layer.kernel_size

  # if strides is not specified, use the strides from i_layer

  if not strides:
    strides = i_layer.strides

  # if padding is not specified, use the padding from i_layer

  if not padding:
    padding = i_layer.padding

  # for conv2d, we want a list for kernel, strides and padding

  if not isinstance(kernel, list) and not isinstance(kernel, tuple):
    kernel = [kernel, kernel]

  if not isinstance(strides, list) and not isinstance(strides, tuple):
    strides = [strides, strides]

  if not isinstance(padding, list) and not isinstance(padding, tuple):
    padding = [padding, padding]

  # compute the padding value

  padding[0] = get_padding_value(padding[0], kernel[0])
  padding[1] = get_padding_value(padding[1], kernel[1])

  # resample inputs

  skip = min(2000, samples)

  indexes = np.array(range(x_train.shape[0]))
  np.random.shuffle(indexes)
  x_train = x_train[indexes[:samples]]

  # we want to create a smaller model that from inputs generate
  # i_layer.output + o_layer.output tensors, so that we can predict
  # its values.

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

  # if randomize is specified, we will sample sqrt(randomize)
  # from each image, as the conv2d performs the filter everywhere
  # in the image. Because the same image may contain a lot of
  # reduntant information, we may want to restrict the number of
  # samples.

  if randomize:
    idx_height = np.random.choice(
        p[-1].shape[1],
        int(np.round(np.sqrt(randomize))))

    idx_width = np.random.choice(
        p[-1].shape[2],
        int(np.round(np.sqrt(randomize))))
  else:
    idx_height = range(p[-1].shape[1])
    idx_width = range(p[-1].shape[2])

  # this is just to inspect that the inputs and outputs are really quantized.

  print("inputs:")
  for i in range(len(x)):
    print(i, np.min(p[i]), np.max(p[i]))
  print("outputs:")
  print(np.min(p[-1]), np.max(p[-1]))

  # i_size and o_size are the channel sizes of the inputs and outputs

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

  table_ins = list()
  table_ous = list()

  print("...indexing inputs")

  # for each image in sampled x_train

  # on Intel processors, mp.cpu_count() returns number of threads

  number_of_processes = mp.cpu_count() // 2
  pool = mp.Pool(number_of_processes)

  results = []

  for n in range(0, x_train.shape[0], skip):

    res = pool.apply_async(
        parallel_index_table,
        args=((p[0][n:n+skip], p[1][n:n+skip]), n, skip, idx_height,
              idx_width, i_dict, o_dict, kernel, strides, padding,
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

      # name is i_<channel>_<kernel_row>_<kernel_col>_bit

      assert ni == (i_size * kernel[0] * kernel[1] * i_bits)

      for channel in range(i_size):
        for row in range(kernel[0]):
          for col in range(kernel[1]):
            for bit in range(i_bits):
              i_names.append("i_{}_{}_{}_{}".format(
                  channel, row, col, (i_bits - 1 - bit)))

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
