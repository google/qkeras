# ==============================================================================
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
"""Plots history of runs when running in scheduler mode."""

import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

filenames = glob.glob("log_*.csv")
filenames.sort()

block_sizes = int(np.ceil(np.sqrt(len(filenames))))

for i in range(len(filenames)):
  history = pd.read_csv(filenames[i])
  title = "block_" + str(i)
  fig = plt.subplot(block_sizes, block_sizes, i + 1, title=title)
  ax1 = fig
  ax1.set_xlabel("trial")
  ax1.set_ylabel("score / accuracy")
  plt1 = ax1.plot(history["score"], "ro-", label="score")
  plt2 = ax1.plot(history["accuracy"], "go-", label="accuracy")
  plt3 = ax1.plot(history["val_accuracy"], "bo-", label="val_accuracy")

  ax2 = ax1.twinx()
  ax2.set_ylabel("energy", color="m")
  plt4 = ax2.plot(history["trial_size"], "mo-", label="trial_size")

  plts = plt1+plt2+plt3+plt4
  labs = [l.get_label() for l in plts]

  ax1.legend(plts, labs, loc=0)
plt.show()
