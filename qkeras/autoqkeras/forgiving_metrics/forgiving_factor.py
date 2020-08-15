# Lint as: python3
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
"""Implements forgiving factor metrics."""

import numpy as np


class ForgivingFactor:
  """Base class. Should never be invoked."""

  def __init__(self, delta_p, delta_n, rate):
    self.delta_p = np.float32(delta_p) / 100.0
    self.delta_n = np.float32(delta_n) / 100.0
    self.rate = np.float32(rate)

  def get_reference(self, model):
    """Computes reference size of model."""

    raise Exception("class not implemented.")

  def get_trial(self, model, schema):
    """Computes size of quantization trial."""

    raise Exception("class not implemented.")

  def delta(self):
    return np.where(
        self.trial_size < self.reference_size,
        self.delta_p * (np.log(self.reference_size/self.trial_size) /
                        np.log(self.rate)),
        self.delta_n * (np.log(self.reference_size/self.trial_size) /
                        np.log(self.rate)))

