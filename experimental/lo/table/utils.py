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
"""Reads and processes tables of PLAs and CSVs."""

from csv import reader
from csv import QUOTE_NONNUMERIC
from math import sqrt
import os
import pprint
from random import seed
from random import randrange
import sys

import numpy as np
from .parser import parse, _X, _0, _1


def str_column_to_float(dataset, column):
  """Converts string column to float."""
  for row in dataset:
    row[column] = float(row[column].strip())

def str_column_to_int(dataset, column, d_values):
  """Converts string column to int."""
  for row in dataset:
    v = int(row[column].strip())
    row[column] = v if not d_values else d_values[v]

def str_column_to_number(dataset, column):
  """Converts output to integer if possible or float."""

  class_values = [row[column] for row in dataset]
  unique = set(class_values)
  lookup = dict()
  is_symbolic = False
  for value in unique:
    try:
      # try int first
      lookup[value] = int(value)
    except ValueError:
      try:
        # if it fails, try float
        lookup[value] = float(value)
      except ValueError:
        # if it fails, it is symbolic
        is_symbolic = True
        break

  # best we an do is to assign unique numbers to the classes
  if is_symbolic:
    for i, value in enumerate(unique):
      lookup[value] = i

  # convert output to unique number
  for row in dataset:
    row[column] = lookup[row[column]]

  return lookup


def int2bin(v, bits):
  str_v = format((v & ((1<<bits)-1)), "#0" + str(bits+2) + "b")[2:]
  return [int(b) for b in str_v]


def load_csv(filename):
  """Loads CSV file."""
  dataset = list()

  with open(filename, 'r') as file:
    csv_reader = reader(file, quoting=QUOTE_NONNUMERIC)
    for row in csv_reader:
      if not row:
        continue
      dataset.append(row)
      #dataset.append([int(v) for v in row])

  return np.array(dataset)


def load_pla(filename):
  """Loads PLA file."""
  dataset = list()
  pla = parse(filename)
  for i,o in zip(pla.pla_i, pla.pla_o):
    i_s = [1 if v == _1 else 0 if v == _0 else 0 for v in i]
    o_s = [sum([(1 << (len(o)-1-oo)) if o[oo] == _1 else 0
                for oo in range(len(o))])]
    dataset.append(i_s + o_s)
  dataset = np.array(dataset)
  return dataset


def load(filename):
  """Loads and decides if we will load PLA or CSV file based on suffix."""

  suffix_split = filename.split(".")

  if suffix_split[-1] == "pla":
    print("... loading pla")
    dataset = load_pla(filename)
  else:
    dataset = load_csv(filename)
  return dataset

