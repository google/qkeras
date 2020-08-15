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
"""Implements forgiving factor metrics for energy consumption."""

import json
import numpy as np
from qkeras.autoqkeras.forgiving_metrics.forgiving_factor import ForgivingFactor   # pylint: disable=line-too-long
from qkeras.qtools import run_qtools
from qkeras.qtools import settings as qtools_settings


class ForgivingFactorPower(ForgivingFactor):
  """Get Power cost of a given model."""

  def __init__(self, delta_p, delta_n, rate, stress=1.0, **kwargs):

    # input parameters:
    # delta_p, delta_n, rate: same as parent class
    # stress: stress level to shift reference curve
    # process: technology process to use in configuration (horowitz, ...)
    # parameters_on_memory: whether to store parameters in dram, sram, or fixed
    # activations_on_memory: store activations in dram, sram
    # min_sram_size: minimum sram size in number of bits
    # rd_wr_on_io: whether load data from dram to sram (consider sram as a cache
    #   for dram. If false, we will assume data will be already in SRAM
    # config_json: if None, use qtools/config_json by default
    #   define default source quantizers;
    #   default quantizers for intermediate variables if no quantizer provided
    #   parameters for energy calculation
    # source_quantizers: quantizer for model input
    # trained_model: whether model has been trained already, which is
    #   needed to compute tighter bounds for qBatchNorm Power estimation.
    # reference_internal: size to use for weight/bias/activation in
    #   get_reference energy calculation (int8, fp16, fp32)
    # reference_accumulator: accumulator and multiplier type in get_reference
    #   energy calculation
    # keras_layer_quantizer: quantizer for keras layers in hybrid models

    super(ForgivingFactorPower, self).__init__(delta_p, delta_n, rate)

    self.stress = stress
    # process: horowitz... - must be present in config_json
    self.process = kwargs.get("process", "horowitz")
    # parameters_on_memory: fixed, sram, dram
    self.parameters_on_memory = kwargs.get(
        "parameters_on_memory", ["fixed"] * 2)
    # activations_on_memory: sram, dram
    self.activations_on_memory = kwargs.get(
        "activations_on_memory", ["dram"] * 2
    )
    self.min_sram_size = kwargs.get("min_sram_size", [0] * 2)
    # rd_wr_on_io: true/false
    self.rd_wr_on_io = kwargs.get("rd_wr_on_io", [True] * 2)
    self.config_json = kwargs.get("config_json", None)
    self.source_quantizers = kwargs.get("source_quantizers", None)
    # trained_model: true/false
    self.trained_model = kwargs.get("trained_model", False)
    # reference_internal: int8, fp16, fp32
    self.reference_internal = kwargs.get("reference_internal", "fp32")
    # reference_internal: int8, int16, int32, fp16, fp32
    self.reference_accumulator = kwargs.get("reference_accumulator", "fp32")

    self.reference_size = None

    # energy_dict is a dictionary that lists energy consumption for each layer
    # format:
    #  {
    #     "layer0_name":
    #     {
    #        "mem_cost": 148171,
    #        "op_cost": 0
    #     },
    #     "layer1_name":
    #     {
    #         "mem_cost": 179923,
    #         "op_cost": 34
    #     },
    #     ...
    #
    #     "total_cost": 328129
    #  }

    self.ref_energy_dict = None
    self.trial_energy_dict = None

    assert self.parameters_on_memory[0] in ["dram", "sram", "fixed"]
    assert self.parameters_on_memory[1] in ["dram", "sram", "fixed"]
    assert self.activations_on_memory[0] in ["dram", "sram", "fixed"]
    assert self.activations_on_memory[1] in ["dram", "sram", "fixed"]
    assert self.reference_internal in ["fp16", "fp32", "int8"]
    assert self.reference_accumulator in ["int16", "int32", "fp16", "fp32"]

  def get_reference(self, model):
    # we only want to compute reference once
    if self.reference_size is not None:
      return self.reference_size * self.stress

    q = run_qtools.QTools(
        model, process=self.process,
        source_quantizers=self.reference_internal,
        is_inference=self.trained_model,
        weights_path=None,
        keras_quantizer=self.reference_internal,
        keras_accumulator=self.reference_accumulator,
        for_reference=True)

    energy_dict = q.pe(
        weights_on_memory=self.parameters_on_memory[0],
        activations_on_memory=self.activations_on_memory[0],
        min_sram_size=self.min_sram_size[0],
        rd_wr_on_io=self.rd_wr_on_io[0])

    self.ref_energy_dict = energy_dict
    self.reference_size = q.extract_energy_sum(
        qtools_settings.cfg.include_energy, energy_dict)

    self.reference_energy_profile = q.extract_energy_profile(
        qtools_settings.cfg.include_energy, energy_dict)

    return self.reference_size * self.stress

  def get_trial(self, model):
    """Computes size of quantization trial."""

    q = run_qtools.QTools(
        model, process=self.process,
        source_quantizers=self.source_quantizers,
        is_inference=self.trained_model,
        weights_path=None,
        keras_quantizer=self.reference_internal,
        keras_accumulator=self.reference_accumulator,
        for_reference=False)

    energy_dict = q.pe(
        weights_on_memory=self.parameters_on_memory[1],
        activations_on_memory=self.activations_on_memory[1],
        min_sram_size=self.min_sram_size[1],
        rd_wr_on_io=self.rd_wr_on_io[1])

    self.trial_energy_dict = energy_dict
    # self.trial_size = energy_dict["total_cost"]
    self.trial_size = q.extract_energy_sum(
        qtools_settings.cfg.include_energy, energy_dict)

    self.trial_energy_profile = q.extract_energy_profile(
        qtools_settings.cfg.include_energy, energy_dict)

    return self.trial_size

  def get_total_factor(self):
    """we adjust the learning rate by size reduction."""
    return (self.trial_size - self.reference_size) / self.reference_size

  def get_reference_stats(self):
    return self.reference_energy_profile

  def get_trial_stats(self):
    return self.trial_energy_profile

  def print_stats(self, verbosity=0):
    """Prints statistics of current model."""

    delta = self.delta()

    if (self.ref_energy_dict and self.trial_energy_dict):
      str_format = (
          "stats: delta_p={} delta_n={} rate={} trial_size={} "
          "reference_size={}\n"
          "       delta={:.2f}%"
      )

      print(
          str_format.format(
              self.delta_p, self.delta_n, self.rate, self.trial_size,
              int(self.reference_size), 100 * delta)
      )

    if verbosity > 0 and self.ref_energy_dict:
      print("Reference Cost Distribution:")
      dict_to_json = json.dumps(self.ref_energy_dict, indent=4)
      print(dict_to_json)

    if verbosity > 0 and self.trial_energy_dict:
      print("Trial Cost Distribution:")
      dict_to_json = json.dumps(self.trial_energy_dict, indent=4)
      print(dict_to_json)

    if (self.ref_energy_dict and self.trial_energy_dict):
      print("Total Cost Reduction:")
      reduction_percentage = np.round(
          100.0 * (self.trial_size - self.reference_size) /
          self.reference_size, 2)

      print(
          ("       {} vs {} ({:.2f}%)").format(
              int(self.trial_size), int(self.reference_size),
              reduction_percentage
          ))
