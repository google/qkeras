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

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

def gen_random_tree_cc(tree):
  n_nodes = tree.node_count
  children_left = tree.children_left
  children_right = tree.children_right
  feature = tree.feature
  threshold = tree.threshold

  node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
  is_leaves = np.zeros(shape=n_nodes, dtype=bool)

  stack = [(0, -1)]

  while (len(stack) > 0):
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1

    if children_left[node_id] != children_right[node_id]:
      stack.append((chidren_left[node_id], parent_depth+1))
      stack.append((children_right[node_id], parent_depth+1))
    else:
      is_leaves[node_id] = True

  for i in range(n_nodes):
    if is_leaves[i]:
      print("{}n_{} leaf node.".format("  "*node_depth[i], i))
    else:
      print("{}n_{} (i_{} <= {}) ? n_{} : n_{}".format(
          "  "*node_depth[i], i, feature[i], threshold[i],
          children_left[i], children_right[i]))
