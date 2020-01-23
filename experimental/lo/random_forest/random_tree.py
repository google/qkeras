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
"""Implements Random Forest for quantized netlist."""

from csv import reader
from math import sqrt
import os
import pprint
from random import seed
from random import randrange
import sys

import numpy as np
from .parser import parse, _X, _0, _1

class RandomTree:
  def __init__(self, max_depth, min_size):
    self.min_size = min_size
    self.max_depth = max_depth
    self.n_features = None

  def split_into_groups(self, index, value, dataset):
    mask_l = dataset[:, index] < value
    mask_r = np.logical_not(mask_l)
    left = dataset[mask_l,...]
    right = dataset[mask_r,...]
    return left, right

  def gini_index(self, groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
      size = float(len(group))
      # avoid divide by zero
      if size == 0:
        continue
      score = 0.0
      # score the group based on the score for each class
      for class_val in classes:
        p = np.array([np.sum(group[:, -1] == class_val) / size
                      for class_val in classes])
        score += np.sum(np.power(p, 2))

      # weight the group score by its relative size
      gini += (1.0 - score) * (size / n_instances)
    return gini

  def select_best_split(self, dataset):
    class_values = list(set(list(dataset[:,-1].flatten())))

    b_index, b_value, b_score, b_groups = 9999, 9999, 9999, None

    # because several of the entries may be don't cares, we will select the
    # whole set and restrict to only the ones that are not don't cares

    features = list(
        np.random.choice(len(dataset[0])-1, self.n_features, p=self.probs,
                         replace=False))

    for index in features:
      assert self.mask[index] == True
      b_values = list(set(list(dataset[:, index])))
      for b in b_values:
        groups = self.split_into_groups(index, b, dataset)
        gini = self.gini_index(groups, class_values)
        if gini < b_score:
          b_index, b_value, b_score, b_groups = index, b, gini, groups

    return {'index': b_index, 'value': b_value, 'groups': b_groups}

  def select_terminal(self, group):
    outcomes = list(group[:,-1].flatten())
    return max(set(outcomes), key=outcomes.count)

  def split_node(self, node, depth):
    left, right = node['groups']
    del(node['groups'])

    # check for a no split
    if left.shape[0] == 0:
      node['left'] = node['right'] = self.select_terminal(right)
      return
    elif right.shape[0] == 0:
      node['left'] = node['right'] = self.select_terminal(left)
      return

    # check for max depth
    if depth >= self.max_depth:
      node['left'], node['right'] = (self.select_terminal(left),
                                     self.select_terminal(right))
      return

    # process left child
    if len(set(list(
        left[:, -1].flatten()))) == 1 or left.shape[0] <= self.min_size:
      node['left'] = self.select_terminal(left)
    else:
      node['left'] = self.select_best_split(left)
      self.split_node(node['left'], depth + 1)

    # process right child
    if len(set(list(
        right[:, -1].flatten()))) == 1 or right.shape[0] <= self.min_size:
      node['right'] = self.select_terminal(right)
    else:
      node['right'] = self.select_best_split(right)
      self.split_node(node['right'], depth+1)

  def create_mask(self, dataset):
    self.mask = np.amin(dataset, axis=0) != np.amax(dataset, axis=0)

  def fit(self, dataset, mask_stuck_at_values=False):
    if mask_stuck_at_values:
      self.create_mask(dataset)
    else:
      self.mask = np.ones(dataset.shape[1])

    self.probs = self.mask[:-1].astype(np.float32) / np.sum(self.mask[:-1])

    if not self.n_features:
      self.n_features = int(np.sqrt(dataset.shape[1] - 1))

    self.root = self.select_best_split(dataset)
    self.split_node(self.root, 1)

  def predict_internal(self, node, data):
    if data[node['index']] < node['value']:
      if isinstance(node['left'], dict):
        return self.predict_internal(node['left'], data)
      else:
        return node['left']
    else:
      if isinstance(node['right'], dict):
        return self.predict_internal(node['right'], data)
      else:
        return node['right']


  def predict_row(self, row):
    return self.predict_internal(self.root, row)


  def predict(self, data):
    return np.array(self.predict_row(data[i]) for i in range(data.shape[0]))

  def gen_code_internal(self, node, var_dict, n_offset):
    # traverse left
    cond = '( i_' + str(node['index']) + ' < ' + str(node['value']) + ' )'
    if isinstance(node['left'], dict):
      n0 = self.gen_code_internal(node['left'], var_dict, n_offset)
    else:
      n0 = str(node['left'])

    if isinstance(node['right'], dict):
      n1 = self.gen_code_internal(node['right'], var_dict, n_offset)
    else:
      n1 = str(node['right'])

    index = len(var_dict) + n_offset
    r = 'n_' + str(index)
    stmt = cond + '? ' + n0 + ' : ' + n1
    var_dict[r] = stmt

    return r

  def gen_code(self, n_offset=0):
    var_dict = {}

    self.gen_code_internal(self.root, var_dict, n_offset)

    return var_dict
