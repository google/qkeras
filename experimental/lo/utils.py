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
"""Computes padding and quantization dictionary values."""

import numpy as np


def get_padding_value(padding, kernel):
  """Returns padding value for kernel."""

  if padding == "valid":
    return 0
  elif padding == "same":
    return kernel // 2
  elif padding == "full":
    return kernel - 1

  raise ValueError("accepted paddings are 'valid', 'same' or 'full', found " +
                   padding)


def get_quantized_bits_dict(bits, ibits, sign=False, mode="bin"):
  """Returns map from floating values to bit encoding."""

  o_dict = {}

  n_bits = bits

  for b in range(1 << (bits - sign)):
    v = (1.0 * b) * (1 << ibits) / (1 << bits)
    if mode == "bin":
      b_str = bin(b)[2:]
      b_str = "0" * (n_bits - len(b_str)) + b_str
    else:  # mode == "dec":
      b_str = str(b)

    o_dict[v] = b_str

    if b > 0 and sign:
      if mode == "bin":
        b_str = bin(-b & ((1 << n_bits) - 1))[2:]
      else:  # mode == "dec"
        b_str = str(-b)

      o_dict[-v] = b_str

  if sign:
    v = (1.0 * (1 << (bits - sign))) * (1 << ibits) / (1 << bits)
    if mode == "bin":
      b_str = bin(-(1 << (bits - sign)) & ((1 << bits) - 1))[2:]
    else:
      b_str = str(-(1 << (bits - sign)))
    o_dict[-v] = b_str
  return o_dict


def get_quantized_po2_dict(
    bits, max_exp, sign=False, make_smaller_zero=True, mode="bin"):
  """Returns map from floating values to bit encoding."""

  # if make_smaller_zero we will make sure smaller number is 000...0

  # mode = "bin" |-> make_smaller_zero

  assert mode != "bin" or  make_smaller_zero

  o_dict = {}

  if max_exp > 0:
    v = 1.0
    if mode == "bin":
      b_str = "0" * bits
    else:
      b_str = "1"

    o_dict[v] = b_str

    if sign:
      v = -1.0
      if mode == "bin":
        b_str = "1" + "0"*(bits-sign)
      else:
        b_str = "-1"

      o_dict[v] = b_str

  for b in range(1, 1<<(bits - sign - 1)):
    v = np.power(2.0, -b)
    if mode == "bin":
      b_sign = "0" if sign else ""
      b_str = b_sign + bin((-b) & ((1 << (bits - sign + 1)) - 1))[3:]
    else:
      b_str = str(v)
    o_dict[v] = b_str

    if b <= max_exp:
      v = np.power(2.0, b)
      if mode == "bin":
        b_str = bin(b)[2:]
        b_str = b_sign + "0"*(bits - sign - len(b_str)) + b_str
      else:
        b_str = str(v)
      o_dict[v] = b_str

    if sign:
      v = -np.power(2.0, -b)
      if mode == "bin":
        b_sign = "1" if sign else ""
        b_str = b_sign + bin((-b) & ((1 << (bits - sign + 1)) - 1))[3:]
      else:
        b_str = str(v)
      o_dict[v] = b_str

      if b <= max_exp:
        v = -np.power(2.0, b)
        if mode == "bin":
          b_str = bin(b)[2:]
          b_str = b_sign + "0"*(bits - sign - len(b_str)) + b_str
        else:
          b_str = str(v)
        o_dict[v] = b_str

  b = 1 << (bits - sign - 1)
  v = np.power(2.0, -b)
  if mode == "bin":
    b_sign = "0" if sign else ""
    b_str = b_sign + bin((-b) & ((1 << (bits - sign + 1)) - 1))[3:]
  else:
    b_str = str(v)
  o_dict[v] = b_str

  smaller_mask = b_str

  if sign:
    v = -np.power(2.0, -b)
    if mode == "bin":
      b_sign = "1" if sign else ""
      b_str = b_sign + bin((-b) & ((1 << (bits - sign + 1)) - 1))[3:]
    else:
      b_str = str(v)
    o_dict[v] = b_str

  def invert_bit(bit, mask):
    """Inverts bits if mask is 1."""

    if mask == "0":
      return bit
    else:
      return "0" if bit == "1" else "1"

  if mode == "bin":
    if make_smaller_zero:
      for v in o_dict:
        o_dict[v] = "".join(
            invert_bit(bit, mask_bit)
            for bit, mask_bit in zip(o_dict[v], smaller_mask))
  else:
    keys_sorted = list(sorted(o_dict.keys()))
    if make_smaller_zero:
      min_positive_key = min([abs(v) for v in keys_sorted])
      min_positive_index = keys_sorted.index(min_positive_key)
    else:
      min_positive_index = 0
    for i, k in enumerate(keys_sorted):
      o_dict[k] = str(i - min_positive_index)

  return o_dict


def get_ternary_dict(mode="bin"):
  """Returns map from floating values to bit encoding."""

  if mode == "bin":
    return {-1.0: "11", 0.0: "00", 1.0: "01"}
  else:
    return {-1.0: "-1", 0.0: "0", 1.0: "1"}


def get_binary_dict(symmetric=False, mode="bin"):
  """Returns map from floating values to bit encoding."""

  if mode == "bin":
    if symmetric:
      return {-1.0: "10", 1.0: "01"}
    else:
      return {0.0: "0", 1.0: "1"}
  else:
    if symmetric:
      return {-1.0: "-1", 1.0: "1"}
    else:
      return {0.0: "0", 1.0: "1"}
