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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse


def ParserArgs():
  """Args Parser."""
  parser = argparse.ArgumentParser()
  parser.add_argument("--debug", default=False, action="store_true",
                      help="set debug mode")
  parser.add_argument("--print_debug", default=False, action="store_true",
                      help="print debug information")

  parser.add_argument("--model", default="",
                      help="which model to run (dmnist, cmnist)")

  parser.add_argument("-o", "--logic_optimize", default=False,
                      action="store_true",
                      help="optimize network.")

  parser.add_argument("-l", "--load_weight", default=False,
                      action="store_true",
                      help="load weights directly from file.")
  parser.add_argument("-w", "--weight_file", default=None,
                      help="name of weights file")

  parser.add_argument("--output_group", type=int, default=1,
                      help="number of outputs to group together")
  parser.add_argument("--kernel", default=None, type=int,
                      help="kernel if more complex layer")
  parser.add_argument("--strides", default=None, type=int,
                      help="stride if more complex layer")
  parser.add_argument("--padding", default=None,
                      help="padding if more complex layer")

  parser.add_argument("--conv_sample", default=None, type=int,
                      help="number of samples within image for conv layer")
  parser.add_argument("--sample", default=None,
                      help="number of training samples")

  parser.add_argument("--use_pla", default=False,
                      action="store_true",
                      help="use pla table format")
  parser.add_argument("--binary", default=False,
                      action="store_true",
                      help="use binary inputs")

  parser.add_argument("--i_name", default=None,
                      help="input layer name")
  parser.add_argument("--o_name", default=None,
                      help="output layer name")
  parser.add_argument("--qi", default="2,0,0",
                      help="quantized input type")
  parser.add_argument("--qo", default="2,0,0",
                      help="quantized output type")

  parser.add_argument("--run_abc", default=False, action="store_true",
                      help="use abc to optimize logic")
  parser.add_argument("--espresso_flags", default="-Dexpand",
                      help="flags to be passed to espresso")
  parser.add_argument("--abc_flags", default="",
                      help="flags to be passed to abc")

  parser.add_argument("--run_rf", default=False, action="store_true",
                      help="use ranform forest to optimize logic")
  parser.add_argument("--n_trees", default=3, type=int,
                      help="number of trees to optimize")
  parser.add_argument("--max_bits", default=1, type=int,
                      help="maximum number of bits for random forest")
  parser.add_argument("--is_regressor", default=False, action="store_true",
                      help="use regressor instead of classifier")
  parser.add_argument("--n_features", default=None,
                      help="number of features for random forest")
  parser.add_argument("--max_depth", default=None,
                      help="maximum depth of random tree")
  parser.add_argument("--sample_size", default=None,
                      help="sample size of table for random tree")

  return parser.parse_args()
