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

def str_column_to_int(dataset, column):
  """Converts string column to int."""
  for row in dataset:
    row[column] = int(row[column].strip())

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


def load_csv(filename):
  """Loads CSV file."""
  dataset = list()
  with open(filename, 'r') as file:
    csv_reader = reader(file)
    for row in csv_reader:
      if not row:
        continue
      dataset.append(row)

  # converts data to int's
  for i in range(0, len(dataset[0])-1):
    str_column_to_int(dataset, i)

  # converts output to int or float
  str_column_to_number(dataset, len(dataset[0])-1)
  dataset = np.array(dataset)

  return dataset


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

