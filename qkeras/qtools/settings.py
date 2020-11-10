# Lint as: python3
# Copyright 2019 Google LLC
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
"""configurations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class ConfigClass:
  """configuration class."""

  def __init__(self):

    self.default_source_quantizer = "quantized_bits(8, 0, 1)"
    self.default_interm_quantizer = "fp32"

    # Horowitz estimates from ISSCC 2014

    self.fpm_add = np.poly1d([0.003125, 0])
    self.fpm_mul = np.poly1d([0.002994791667, 0.001041666667, 0])
    self.fp16_add = np.poly1d([0.4])
    self.fp16_mul = np.poly1d([1.1])
    self.fp32_add = np.poly1d([0.9])
    self.fp32_mul = np.poly1d([3.7])

    self.sram_rd = np.poly1d([0.02455, -0.2656, 0.8661])
    self.dram_rd = np.poly1d([20.3125, 0])
    self.sram_mul_factor = 1/64.
    self.dram_mul_factor = 1.0

    self.include_energy = {}
    self.include_energy["default"] = ["inputs", "parameters", "op_cost"]
    self.include_energy["QActivation"] = ["outputs"]
    self.include_energy["QAdaptiveActivation"] = ["outputs"]
    self.include_energy["Activation"] = ["outputs"]
    self.include_energy["QBatchNormalization"] = ["parameters"]
    self.include_energy["BatchNormalization"] = ["parameters"]
    self.include_energy["Add"] = ["op_cost"]
    self.include_energy["Subtract"] = ["op_cost"]
    self.include_energy["MaxPooling2D"] = ["op_cost"]
    self.include_energy["default"] = ["inputs", "parameters", "op_cost"]

  def update(self, process, cfg_setting):
    """update config."""

    # pylint: disable=bare-except
    try:
      self.default_source_quantizer = cfg_setting[
          "default_source_quantizer"]
    except:
      pass

    try:
      self.default_interm_quantizer = cfg_setting[
          "default_interm_quantizer"]
    except:
      pass

    try:
      self.fpm_add = np.poly1d(cfg_setting[process]["fpm_add"])
    except:
      pass

    try:
      self.fpm_mul = np.poly1d(cfg_setting[process]["fpm_mul"])
    except:
      pass

    try:
      self.fp16_add = np.poly1d(cfg_setting[process]["fp16_add"])
    except:
      pass

    try:
      self.fp16_mul = np.poly1d(cfg_setting[process]["fp16_mul"])
    except:
      pass

    try:
      self.fp32_add = np.poly1d(cfg_setting[process]["fp32_add"])
    except:
      pass

    try:
      self.fp32_mul = np.poly1d(cfg_setting[process]["fp32_mul"])
    except:
      pass

    try:
      self.sram_rd = np.poly1d(cfg_setting[process]["sram_rd"])
    except:
      pass

    try:
      self.dram_rd = np.poly1d(cfg_setting[process]["dram_rd"])
    except:  # pylint: disable=broad-except
      pass

    try:
      for key in cfg_setting["include_energy"]:
        self.include_energy[key] = cfg_setting["include_energy"][key]
        if "Q" == key[0]:
	        # use the same rule for keras layer and qkeras layer
          self.include_energy[key[1:]] = cfg_setting["include_energy"][key]
    except:
      pass


cfg = ConfigClass()

