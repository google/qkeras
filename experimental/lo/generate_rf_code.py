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

import os

import numpy as np

DEBUG = int(os.environ.get("DEBUG", 0))
PRINT_DEBUG = int(os.environ.get("PRINT_DEBUG", 0))


def gen_random_tree_regressor(
    tree, code, bits, o_bits, o_decimal_digits, o_is_neg, bdd, offset, is_cc=True):
  """Generates HLS friendly C++ code for random tree regressor.

  Generates HLS friendly C++ code for Catapult.

  Arguments:
    tree: decision tree regressor from SkLearn.
    code: list of code lines to be append to.
    bits: list containing number of bits for each of the inputs.
    o_bits: number of bits for output.
    o_decimal_digits: number of decimal digits (right of the decimal point
        of o_bits for approximation of regressor in RandomTreeRegressor.
    o_is_neg: True or 1 if output can be negative.
    bdd: we actually try to cache entries (i,v,n1,n0) entries so that if
        they appear again, we reuse previously computed nodes.
    offset: each variable created in this function call is incremented by
        offset.
    is_cc: if True, generates C++, else Verilog.

  Returns:
    Tuple containing last variable name and current number of variables.

  """

  # extract information from tree

  n_nodes = tree.node_count
  children_left = tree.children_left
  children_right = tree.children_right
  feature = tree.feature
  threshold = tree.threshold
  values = np.copy(tree.value)

  o_suffix = ""
  if DEBUG:
    o_type = "float"
  elif is_cc:
    o_type = "ac_fixed<{},{},{}>".format(
        o_bits + o_decimal_digits,
        o_bits + o_is_neg,
        o_is_neg)
  else:
    o_sign = " signed" if o_is_neg else ""
    if o_bits + o_decimal_digits > 1:
      o_suffix = "[{}:0]".format(o_bits + o_decimal_digits - 1)
    o_type = "wire" + o_sign + " " + o_suffix


  def round_digits(x, decimal_digits):
    """Rounds to decimal_digits to the right of the decimal point."""

    if DEBUG:
      return x
    factor = (1 << decimal_digits) * 1.0
    x = x * factor
    return np.round(x) / factor

  is_leaves = np.zeros(shape=n_nodes, dtype=bool)

  stack = [(0, -1)]

  while stack:
    node_id, parent_depth = stack.pop()

    if children_left[node_id] != children_right[node_id]:
      stack.append((children_left[node_id], parent_depth+1))
      stack.append((children_right[node_id], parent_depth+1))
    else:
      is_leaves[node_id] = True
      values[node_id] = round_digits(tree.value[node_id], o_decimal_digits)
      if (
          values[node_id].flatten()[0] != tree.value[node_id].flatten()[0] and
          DEBUG
      ):
        print(node_id, values[node_id].flatten()[0],
              tree.value[node_id].flatten()[0])

  v_name = {}
  n_vars = offset

  bdd = {}

  def round_value_to_int(x):
    v = hex(int(np.round(x * (1 << (o_decimal_digits)))))
    if is_cc:
      if DEBUG:
        return str(x)
      else:
        return x
      #v + " /* {} */".format(x)
    else:
      return (
          str(o_bits + o_decimal_digits) + "'h" + v[2:] + " /* {} */".format(x)
      )

  if is_leaves[0]:
    v_name[0] = round_value_to_int(values[0].flatten()[0])
    code.append("  {} n_{} = {};".format(o_type, n_vars, v_name[0]))
    last_var = "n_{}".format(n_vars)
    n_vars += 1
  else:
    for i in range(n_nodes-1, -1, -1):
      if is_leaves[i]:
        continue

      if v_name.get(children_left[i], None) is not None:
        n1 = v_name[children_left[i]]
      elif is_leaves[children_left[i]]:
        n1 = round_value_to_int(values[children_left[i]].flatten()[0])
        v_name[children_left[i]] = n1
      else:
        n1 = "n_" + str(n_vars)
        n_vars += 1
        v_name[children_left[i]] = n1
        raise ValueError((children_left[i], n1, is_leaves[children_left[i]]))

      if v_name.get(children_right[i], None) is not None:
        n0 = v_name[children_right[i]]
      elif is_leaves[children_right[i]]:
        n0 = round_value_to_int(values[children_right[i]].flatten()[0])
        v_name[children_right[i]] = n0
      else:
        n0 = "n_" + str(n_vars)
        n_vars += 1
        v_name[children_right[i]] = n0
        raise ValueError((children_right[i], n0, is_leaves[children_right[i]]))

      if v_name.get(i, None) is not None:
        n = v_name[i]
        last_var = v_name[i]
      elif bdd.get((feature[i], threshold[i], n1, n0), None) is not None:
        n = bdd[(feature[i], threshold[i], n1, n0)]
        v_name[i] = n
        last_var = n
      elif n1 == n0:
        # store intermediate results so that we can build a dag, not a tree
        bdd[(feature[i], threshold[i], n1, n0)] = n1
        v_name[i] = n1
        last_var = n1
      else:
        n = "n_" + str(n_vars)
        n_vars += 1
        v_name[i] = n
        # store intermediate results so that we can build a dag, not a tree
        bdd[(feature[i], threshold[i], n1, n0)] = n
        t = int(threshold[i])
        if bits[feature[i]] == 1:
          if t == 0:
            n1, n0 = n0, n1
          code.append(
              "  {} {} = (i_{}) ? {} : {}; // x_{} {}".format(
                  o_type, v_name[i], feature[i], n1, n0, i,
                  threshold[i]))
        else:
          code.append(
              "  {} {} = (i_{} <= {}) ? {} : {}; // x_{} {}".format(
                  o_type, v_name[i], feature[i], t, n1, n0, i,
                  threshold[i]))
        last_var = v_name[i]

  return (last_var, n_vars)


def entry_to_hex(entry, max_value, size, is_cc):
  """Converts class instance to hexa number."""

  e_vector = [np.power(max_value+1, i) for i in range(len(entry)-1, -1, -1)]
  entry = np.array(entry)
  v = hex(np.sum(entry * e_vector))

  if is_cc:
    return v
  else:
    return str(size) + "'h" + v[2:] + " /* {} */".format(entry)


def gen_random_tree_classifier(
    tree, code, bits, bdd, max_value, values_rom, offset, is_cc=True):
  """Generates C++ or Verilog friendly code for random tree classifier.

  Generates HLS Catapult friendly code or RTL in Verilog for random tree
  classifier from SkLearn.

  Arguments:
    tree: RandomTreeClassifier from sklearn.
    code: list of strings containing code generated.
    bits: list containing number of bits for each of the inputs.
    bdd: we actually try to cache entries (i,v,n1,n0) entries so that if
        they appear again, we reuse previously computed nodes.
    max_value: random tree classifiers returns vector of classes with the
        number of instances found in the terminal leaf node. This variable
        specifies a clipping factor for each class type so that we have
        a bounded problem to synthesize.
    values_rom: to save space in classifier, we store class values in
        values_rom.
    offset: each variable created in this function call is incremented by
        offset.
    is_cc: if True, generates C++ code; otherwise, Verilog.

  Returns:
    Tuple containing last variable name and current number of variables.
  """

  # extract information from tree

  n_nodes = tree.node_count
  children_left = tree.children_left
  children_right = tree.children_right
  feature = tree.feature
  threshold = tree.threshold

  values = {}

  is_leaves = np.zeros(shape=n_nodes, dtype=bool)

  stack = [(0, -1)]

  rom_l = []

  use_rom = max_value >= 7

  n_classes = len(tree.value[0].flatten())

  max_bits = int(np.ceil(np.log2(max_value + 1)))

  while stack:
    node_id, parent_depth = stack.pop()

    if children_left[node_id] != children_right[node_id]:
      stack.append((children_left[node_id], parent_depth+1))
      stack.append((children_right[node_id], parent_depth+1))
    else:
      # is leaf node
      is_leaves[node_id] = True
      # get tree node output
      p_input_tuple = tree.value[node_id].flatten().astype(np.int32)
      max_input_value = np.max(p_input_tuple)
      min_input_value = np.min(p_input_tuple)
      # if max_value == 1, only keep top ones
      if max_value == 1:
        input_tuple = (p_input_tuple == max_input_value).astype(np.int32)
        tree.value[node_id] = (tree.value[node_id] == max_input_value).astype(
            tree.value[node_id].dtype)
      else: # if max_value <= 3:
        # SKLearn classifier computes probability for each entry instead of
        # suming them all. We should do the same.
        max_input_value = np.sum(p_input_tuple)
        min_input_value = 0
        # Just update tree.value to number so that we can compare accuracy of
        # quantization later.
        tree.value[node_id] = np.round(
            max_value *
            (tree.value[node_id] - min_input_value) /
            (max_input_value - min_input_value))
        input_tuple = tree.value[node_id].flatten()
      input_tuple = tuple(list(input_tuple.astype(np.int32)))

      # stores values in rom - we will use rom to store values if use_rom is
      # true.
      if values_rom.get(input_tuple, None) is None:
        values_rom[input_tuple] = len(values_rom)
        rom_l.append(input_tuple)
        if DEBUG:
          print(values_rom[input_tuple], input_tuple)

      if use_rom:
        values[node_id] = values_rom[input_tuple]
      else:
        values[node_id] = entry_to_hex(
            input_tuple, max_value, max_bits * n_classes, is_cc)

  # t_bits: entry type
  # l_bits: table line type
  if use_rom:
    t_bits = int(np.ceil(np.log2(len(values_rom))))
    l_bits = max_bits * n_classes
  else:
    t_bits = max_bits * n_classes

  # we only store the index here, as we read from a rom
  if is_cc:
    if DEBUG:
      t_type = "int"
    else:
      t_type = "ac_int<{},false>".format(t_bits)
  else:
    t_type = "wire [{}:0]".format(t_bits-1)

  v_name = {}
  n_vars = offset

  bdd = {}

  if is_leaves[0]:
    v_name[0] = t_type + "(" + str(values[0]) + ")"
    code.append("  {} n_{} = {};".format(
        t_type, n_vars, values[0]))
    last_var = "n_{}".format(n_vars)
    n_vars += 1
  else:
    for i in range(n_nodes-1, -1, -1):
      if is_leaves[i]:
        continue

      if v_name.get(children_left[i], None) is not None:
        n1 = v_name[children_left[i]]
      elif is_leaves[children_left[i]]:
        if is_cc:
          n1 = t_type + "(" + str(values[children_left[i]]) + ")"
        else:
          n1 = str(values[children_left[i]])
        v_name[children_left[i]] = n1
      else:
        n1 = "n_" + str(n_vars)
        n_vars += 1
        v_name[children_left[i]] = n1
        raise ValueError((children_left[i], n1, is_leaves[children_left[i]]))

      if v_name.get(children_right[i], None) is not None:
        n0 = v_name[children_right[i]]
      elif is_leaves[children_right[i]]:
        if is_cc:
          n0 = t_type + "(" + str(values[children_right[i]]) + ")"
        else:
          n0 = str(values[children_right[i]])
        v_name[children_right[i]] = n0
      else:
        n0 = "n_" + str(n_vars)
        n_vars += 1
        v_name[children_right[i]] = n0
        raise ValueError((children_right[i], n0, is_leaves[children_right[i]]))

      if v_name.get(i, None) is not None:
        n = v_name[i]
        last_var = v_name[i]
      elif bdd.get((feature[i], threshold[i], n1, n0), None) is not None:
        n = bdd[(feature[i], threshold[i], n1, n0)]
        v_name[i] = n
        last_var = n
      elif n1 == n0:
        # store intermediate results so that we can build a dag, not a tree
        bdd[(feature[i], threshold[i], n1, n0)] = n1
        v_name[i] = n1
        last_var = n1
      else:
        n = "n_" + str(n_vars)
        n_vars += 1
        v_name[i] = n
        # store intermediate results so that we can build a dag, not a tree
        bdd[(feature[i], threshold[i], n1, n0)] = n
        t = int(threshold[i])
        if bits[feature[i]] == 1:
          if t == 0:
            n1, n0 = n0, n1
          code.append(
              "  {} {} = (i_{}) ? {} : {}; // x_{} {}".format(
                  t_type, v_name[i], feature[i], n1, n0, i,
                  threshold[i]))
        else:
          code.append(
              "  {} {} = (i_{} <= {}) ? {} : {}; // x_{} {}".format(
                  t_type, v_name[i], feature[i], t, n1, n0, i,
                  threshold[i]))
        last_var = v_name[i]

  if use_rom:
    if is_cc:
      if DEBUG:
        l_type = "int"
      else:
        l_type = "ac_int<{},false>".format(l_bits)

      code.append("  {} {}_rom[{}]".format(l_type, last_var, len(values_rom)) +
                  " {")
      for i in range(len(values_rom)):
        code_s = "    " + entry_to_hex(rom_l[i], max_value, l_bits, is_cc)
        if i < len(values_rom)-1:
          code_s = code_s + ","
        code.append(code_s)
      code.append("  };")

    else:
      l_type = "wire [{}:0]".format(l_bits - 1)
      code.append("  function [{}:0] {}_rom;".format(l_bits-1, last_var))
      code.append("  input [{}:0] address;".format(t_bits-1))
      code.append("  begin")
      code.append("    case (address)")
      for i in range(len(values_rom)):
        code.append("    {}'d{}: {}_rom = {};".format(
            l_bits, i, last_var, entry_to_hex(rom_l[i], max_value, l_bits, is_cc)))
      code.append("    default: {}_rom = 0;".format(last_var))
      code.append("    endcase")
      code.append("  end")
      code.append("  endfunction")

    code.append("  {} v_{} = {}_rom[{}];".format(
        l_type, last_var, last_var, last_var))

    last_var = "v_" + last_var

  return last_var, n_vars


def gen_random_forest(
    rf, name, bits, is_neg, o_bits, o_is_neg, is_regressor=True,
    is_top_level=False, is_cc=True):
  """Generates HLS based C++ or SystemVerilog code for random forest."""

  # TODO(nunescoelho): need to take care of multiple outputs for classifier.
  # we can get better result if we do not look at the winning classifier,
  # but sum how many of them appear in each classifier for leaf nodes.

  bdd = {}
  values_rom = {}
  offset = 0
  code = []

  max_value = (1 << int(os.environ.get("MAX_BITS",1))) - 1
  decimal_digits = int(os.environ.get("MAX_BITS", 5))

  assert max_value > 0

  o_list = []
  for i in range(len(rf.estimators_)):
    tree = rf.estimators_[i].tree_
    code.append("  //----- TREE {}".format(i))
    if is_regressor:
      last_var, offset = gen_random_tree_regressor(
          tree, code, bits, o_bits, decimal_digits, o_is_neg, bdd, offset, is_cc)
    else:
      values_rom = {}
      last_var, offset = gen_random_tree_classifier(
          tree, code, bits, bdd, max_value, values_rom, offset, is_cc)

    o_list.append(last_var)

  if is_cc:
    header = [
        "#include <ac_int.h>",
        "#include <ac_fixed.h>",
        "#include <iostream>",
        "using namespace std;",
        "//#define _PRINT_DEBUG_",
        "#define PB(n) cout << #n << \":\" << n << endl;",
        "#define PS(n) \\",
        "  cout << #n << \":\" << n.to_double() << \" \"; \\",
        "  for(int i=n.width-1; i>=0; i--) cout << n[i]; cout << endl;"
    ]

    if DEBUG:
      header = header + [
          "static inline float round_even(float x) {",
          "  int x_int = truncf(x);",
          "  float x_dec = x - x_int;",
          "  if ((x_dec == 0.5) && (x_int % 2 == 0)) {",
          "    return truncf(x);",
          "  } else {",
          "    return truncf(x + 0.5);"
          "  }",
          "}"
      ]
      if is_top_level:
        header.append("#pragma hls_design top")
      header.append("void {}(int in[{}], int &out)".format(
          name, np.sum(bits), o_bits) + " {")
    else:
      n_bits = int(np.ceil(np.log2(len(o_list))))
      header = header + [
          "static inline ac_int<{},{}> round_even(ac_fixed<{},{},{}> x)".format(
              o_bits, o_is_neg,
              n_bits + o_bits + decimal_digits, n_bits + o_bits + o_is_neg,
              o_is_neg
          ) + " {",
          "  bool x_int_is_even = x[{}] == 0;".format(decimal_digits + n_bits),
          "  bool x_frac_is_0_5 = x[{}] && (x.slc<{}>(0) == 0);".format(
              n_bits + decimal_digits-1, n_bits + decimal_digits-1),
          "  if (x_frac_is_0_5 && x_int_is_even) {",
          "    return x.slc<{}>({});".format(o_bits, n_bits + decimal_digits),
          "  } else {",
          "    ac_int<{},{}> r = x.slc<{}>({}) + 1;".format(
              o_bits + 1, o_is_neg,
              o_bits + 1, n_bits + decimal_digits - 1),
          "    return r.slc<{}>(1);".format(o_bits + 1),
          #"    return (x + ac_fixed<{},{},{}>({})).slc<{}>({});".format(
          #    n_bits + o_bits + decimal_digits, n_bits + o_bits + o_is_neg,
          #    o_is_neg, 1<<(n_bits+decimal_digits-1),
          #    o_bits, n_bits + decimal_digits),
          #    #o_is_neg, len(o_list)/2, o_bits, n_bits + decimal_digits),
          "  }",
          "}"
      ]
      if is_top_level:
        header.append("#pragma hls_design top")
      header.append("void {}(ac_int<{},0> in, ac_int<{},{}> &out)".format(
          name, np.sum(bits), o_bits, o_is_neg) + " {")
  else:
    n_bits = int(np.ceil(np.log2(len(o_list))))
    i_decl = "  input [{}:0] in;".format(np.sum(bits)-1)
    o_sign = "signed " if o_is_neg else ""
    o_decl = "  output " + o_sign + "[{}:0] out;".format(o_bits-1)
    header = [
        "module " + name + "(in, out);",
        i_decl,
        o_decl,
        "",
        "  function {}[{}:0] round_even;".format(o_sign, o_bits),
        "  input {}[{}:0] x;".format(o_sign, n_bits + o_bits + decimal_digits - 1),
        "  reg x_int_is_even;",
        "  reg x_frac_is_0_5;",
        "  reg {}[{}:0] round_sum;".format(o_sign, o_bits + 1),
        "  begin",
        "    x_int_is_even = x[{}] == 0;".format(decimal_digits + n_bits),
        "    x_frac_is_0_5 = x[{}] && (x[{}:0] == 0);".format(
            n_bits + decimal_digits-1, n_bits + decimal_digits - 2),
        "    if (x_frac_is_0_5 && x_int_is_even)",
        "      round_even = x[{}:{}];".format(
            n_bits + decimal_digits + o_bits - 1, n_bits + decimal_digits),
        "    else",
        "    begin",
        "      round_sum = x[{}:{}] + 1;".format(
            n_bits + decimal_digits + o_bits - 1, n_bits + decimal_digits - 1),
        "      round_even = round_sum[{}:1];".format(o_bits + 1),
        "    end",
        #"      round_even = (x + {})[{}:{}];".format(
        #    #(1 << (n_bits + decimal_digits - 1)),
        #    n_bits + decimal_digits + o_bits - 1, n_bits + decimal_digits),
        "  end",
        "  endfunction"
    ]


  all_bits = np.sum(bits)
  sum_i = 0
  for i in range(bits.shape[0]):
    if is_cc:
      if bits[i] > 1:
        if DEBUG:
          header.append("  int i_{} = in[{}];".format(i, i))
        else:
          header.append("  ac_int<{},{}> i_{} = in.slc<{}>({});".format(
              bits[i], is_neg[i], i, bits[i], sum_i))
      else:
        header.append("  bool i_{} = in[{}];".format(i, sum_i))
    else:
      if bits[i] == 1:
        header.append("  wire i_{} = in[{}];".format(i, all_bits - sum_i - 1))
      else:
        header.append("  wire i_{}[{}:0] = in[{}:{}];".format(
            i, bits[i], sum_i + bits[i] - 1, all_bits - sum_i - 1))
    sum_i += bits[i]

  footer = []

  if is_regressor:
    n_bits = int(np.ceil(np.log2(len(o_list))))
    assert 1 << n_bits == len(o_list)

    if is_cc:

      if DEBUG:
        tmp_type = "float"
      else:
        tmp_type = "ac_fixed<{},{},{}>".format(
            n_bits + o_bits + decimal_digits, n_bits + o_bits + o_is_neg,
            o_is_neg)
      avg_o = "  {} o_tmp = {};".format(tmp_type, " + ".join(o_list))

      # rnd_o = "  o_tmp += {}({});".format(tmp_type, len(o_list)/2)

      if DEBUG:
        out = "  out = round_even(o_tmp / {});".format(len(o_list))
      else:
        out = "  out = round_even(o_tmp);"

      footer.append("  #ifdef _PRINT_DEBUG_")
      for o_name in o_list:
        footer.append("  PS({});".format(o_name))
      footer.append("  #endif")
      closing = "}"

    else:
      tmp_sign = "signed " if o_is_neg else ""
      avg_o = "  wire " + tmp_sign + "[{}:0] o_tmp = {};".format(
          n_bits + o_bits + decimal_digits - 1, " + ".join(o_list))

      for n in o_list:
        footer.append("  // always @({}) $display(\"{} = %f (%b)\", {} / 32.0, {});".format(n,n,n,n))
      footer.append("  // always @(o_tmp) $display(\"o_tmp = %b\", o_tmp);")

      out = "  assign out = round_even(o_tmp);"

      closing = "endmodule"

    footer = footer + [avg_o, out, closing]

  else:

    assert not o_is_neg

    footer = []

    o_suffix = ""
    if DEBUG:
      o_type = "int"
    elif is_cc:
      o_type = "ac_int<{},{}>".format(o_bits, o_is_neg)
    else:
      o_sign = " signed" if o_is_neg else ""
      o_suffix = "[{}:0]".format(o_bits)
      o_type = "wire" + o_sign + " " + o_suffix

    if is_cc:
      n_classes = 1 << o_bits
      max_bits = int(np.ceil(np.log2(max_value + 1)))
      log2_o_list = int(np.ceil(np.log2(len(o_list))))
      if DEBUG:
        log2_o_type = "int"
      else:
        log2_o_type = "ac_int<{},false>".format(log2_o_list + max_bits)
      sum_v = (
          "  {} sum[{}] = ".format(
              log2_o_type, 1 << o_bits) + "{" +
          ",".join("0" * (1 << o_bits)) + "};"
      )
      footer = [sum_v]
      for o_name in o_list:
        for i in range(n_classes):
          if DEBUG:
            footer.append("  sum[{}] += ({} >> {}) & {};".format(
                i, o_name, (n_classes - i) * max_bits - max_bits,
                hex((1 << max_bits) - 1)))
          else:
            footer.append("  sum[{}] += {}.slc<{}>({});".format(
                i, o_name, max_bits, (n_classes - i) * max_bits - max_bits))
        debug_print = []
        for i in range(n_classes):
          debug_print.append("{}.slc<{}>({}).to_string(AC_DEC)".format(
              o_name, max_bits, (n_classes - i) * max_bits - max_bits))
        footer_s = (
            "  cout << \"{} \" <<".format(o_name) +
            " << \" \" << ".join(debug_print) + " << endl;"
        )
        footer.append("  #ifdef _PRINT_DEBUG_")
        footer.append(footer_s)
        footer.append("  #endif")
      footer.append("  {} max_tmp = sum[0];".format(log2_o_type))
      footer.append("  {} max_id = 0;".format(o_type))
      footer.append("  for(int i=1; i<{}; i++)".format(1 << o_bits))
      footer.append(
        "    if (sum[i] >= max_tmp) { max_tmp = sum[i]; max_id = i; }")
      out = "  out = max_id;"

      footer.append(out)
      footer += ["}"]
    else:
      n_classes = 1 << o_bits
      max_bits = int(np.ceil(np.log2(max_value + 1)))
      log2_o_list = int(np.ceil(np.log2(len(o_list))))
      log2_o_type = "wire [{}:0]".format(log2_o_list + max_bits)
      footer = []
      for i in range(n_classes):
        code_s = "  {} sum_{} = ".format(log2_o_type, i)
        code_term = []
        for o_name in o_list:
          code_term.append("{}[{}:{}]".format(
              o_name, (n_classes - i) * max_bits, (n_classes - i) * max_bits - max_bits))
        code_s += " + ".join(code_term) + ";"
        footer.append(code_s)
        footer.append("  // always @(sum_{}) $display(\"sum_{} = %d\", sum_{});".format(
            i, i, i))
      footer.append("  reg [{}:0] max_tmp;".format(
          log2_o_list + max_bits - 1))
      footer.append("  reg [{}:0] max_id;".format(o_bits-1))
      footer.append("  integer i;")
      footer.append("  always @(" +
                    " or ".join(
                        ["sum_" + str(i) for i in range(n_classes)]) + ")")
      footer.append("  begin")
      footer.append("    max_tmp = sum_0; max_id = 0;")
      for i in range(1, n_classes):
        footer.append(
            "    if (sum_{} >= max_tmp) begin max_tmp = sum_{}; max_id = {}; end".format(
                i, i, i))
      footer.append("  end")
      footer.append("  assign out = max_id;")
      footer.append("endmodule")

  return header + code + footer


def gen_testbench_sv(rf, name, bits, is_neg, o_bits, o_is_neg, x, y, p, code):
  code.append("module tb;")
  x_0, x_1 = x.shape
  x_0_log2 = int(np.ceil(np.log2(x_0)))
  code.append("reg [{}:0] x_rom[{}:0];".format(x_1-1, x_0-1))
  code.append("initial $readmemb(\"x.rom\", x_rom, 0, {});".format(x_0-1))
  with open("x.rom", "w") as f:
    for i in range(len(x)):
      f.write("".join([str(int(v)) for v in x[i]]) + "\n")

  o_sign = "signed " if o_is_neg else ""
  o_type = o_sign + "[{}:0]".format(o_bits - 1)
  code.append("reg {} y_rom[{}:0];".format(o_type,x_0-1))
  code.append("reg {} p_rom[{}:0];".format(o_type,x_0-1))
  with open("y.rom","w") as f:
    for i in range(len(y)):
      f.write(hex(int(y[i]))+ "\n")
  with open("p.rom","w") as f:
    for i in range(len(y)):
      f.write(hex(int(p[i]))+ "\n")
  code.append("initial $readmemh(\"y.rom\", y_rom, 0, {});".format(x_0-1))
  code.append("initial $readmemh(\"p.rom\", p_rom, 0, {});".format(x_0-1))
  code.append("integer i;")
  code.append("integer cnt;")
  code.append("reg [{}:0] in;".format(x_1-1))
  code.append("wire {} out;".format(o_type))
  code.append("{} {}(in, out);".format(name, name))
  code.append("initial")
  code.append("begin")
  code.append("  cnt = 0;")
  code.append("  in = x_rom[i];")
  code.append("  for (i=0; i<{}; i=i+1)".format(x_0))
  code.append("  begin")
  code.append("    in = x_rom[i];")
  code.append("    #1000;")
  code.append("    if (p_rom[i] != out && y_rom[i] != out)")
  code.append("    begin")
  code.append("      $display(\"%d: %b y=%d p=%d -> %d\", i, x_rom[i], y_rom[i], p_rom[i], out);")
  code.append("    end")
  code.append("    else")
  code.append("    begin")
  code.append("      cnt = cnt + 1;")
  code.append("    end")
  code.append("  end")
  code.append("  $display(\"acc = %f\", 100.0 * cnt / {});".format(x_0))
  code.append("end")
  code.append("endmodule")


def gen_testbench_cc(rf, name, bits, is_neg, o_bits, o_is_neg, x, y, p, code):
  code.append("int x[{}][{}] = ".format(*x.shape) + "{")
  for i in range(len(x)):
    code_s = "  {" + ",".join([str(int(v)) for v in x[i]]) + "}"
    if i < len(x) - 1:
      code_s = code_s + ","
    code.append(code_s)
  code.append("};")
  code_s = (
      "int y[{}] = ".format(y.shape[0]) + "{" +
      ",".join([str(int(v)) for v in y]) + "};"
  )
  code.append(code_s)
  code_s = (
      "int p[{}] = ".format(p.shape[0]) + "{" +
      ",".join([str(int(v)) for v in p]) + "};"
  )
  code.append(code_s)

  code.append("int main()")
  code.append("{")
  code.append("  double acc = 0.0;")
  if DEBUG:
    code.append("  int in[{}];".format(x.shape[1]))
    code.append("  int out;")
  else:
    code.append("  ac_int<{},0> in;".format(x.shape[1]))
    code.append("  ac_int<{},{}> out;".format(o_bits, o_is_neg))

  code.append("  for (int i=0; i<{}; i++)".format(x.shape[0]) + "{")
  code.append("    for (int j=0; j<{}; j++) in[j] = x[i][j];".format(
      x.shape[1]))
  code.append("    {}(in, out);".format(name))
  code.append("    if (p[i] != out && y[i] != out) {")
  code.append("      cout << i << \": \";")
  code.append("      for (int j=0; j<{}; j++) cout << in[j];".format(
      x.shape[1]))
  if DEBUG:
    code.append("      cout << \" y=\" << y[i] << \" p=\" << p[i] << \" \" << out << endl;")
    code.append("    }")
    code.append("    acc += (y[i] == out);")
  else:
    code.append("      cout << \" y=\" << y[i] << \" p=\" << p[i] << \" \" << out.to_int() << endl;")
    code.append("      #ifdef _PRINT_DEBUG_")
    code.append("        exit(1);")
    code.append("      #endif")
    code.append("    }")
    code.append("    acc += (y[i] == out.to_int());")
  code.append("  }")
  code.append("  cout << \"acc = \" << 100.0 * acc  / {} << endl;".format(
      x.shape[0]))
  code.append("}")

