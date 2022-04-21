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
"""Example code to generate weight and MAC sizes in a json file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.keras as keras

from qkeras import QActivation
from qkeras import QDense
from qkeras import quantizers
from qkeras.qtools import run_qtools
from qkeras.qtools import settings as qtools_settings


def hybrid_model():
  """hybrid model that mixes qkeras and keras layers."""

  x = x_in = keras.layers.Input((784,), name="input")
  x = keras.layers.Dense(300, name="d0")(x)
  x = keras.layers.Activation("relu", name="d0_act")(x)
  x = QDense(100, kernel_quantizer=quantizers.quantized_bits(4, 0, 1),
             bias_quantizer=quantizers.quantized_bits(4, 0, 1),
             name="d1")(x)
  x = QActivation("quantized_relu(4,0)", name="d1_qr4")(x)
  x = QDense(
      10, kernel_quantizer=quantizers.quantized_bits(4, 0, 1),
      bias_quantizer=quantizers.quantized_bits(4, 0, 1),
      name="d2")(x)
  x = keras.layers.Activation("softmax", name="softmax")(x)

  return keras.Model(inputs=[x_in], outputs=[x])


if __name__ == "__main__":
  # input parameters:
  # process: technology process to use in configuration (horowitz, ...)
  # weights_on_memory: whether to store parameters in dram, sram, or fixed
  # activations_on_memory: store activations in dram or sram
  # rd_wr_on_io: whether load data from dram to sram (consider sram as a cache
  #   for dram. If false, we will assume data will be already in SRAM
  # source_quantizers: quantizers for model input
  # is_inference: whether model has been trained already, which is
  #   needed to compute tighter bounds for QBatchNormalization Power estimation.
  # reference_internal: size to use for weight/bias/activation in
  #   get_reference energy calculation (int8, fp16, fp32)
  # reference_accumulator: accumulator and multiplier type in get_reference
  #   energy calculation
  model = hybrid_model()
  model.summary()

  reference_internal = "int8"
  reference_accumulator = "int32"

  # By setting for_reference=True, we create QTools object which uses
  # keras_quantizer to quantize weights/bias and
  # keras_accumulator to quantize MAC variables for all layers. Obviously, this
  # overwrites any quantizers that user specified in the qkeras layers. The
  # purpose of doing so is to enable user to calculate a baseline energy number
  # for a given model architecture and compare it against quantized models.
  q = run_qtools.QTools(
      model,
      # energy calculation using a given process
      process="horowitz",
      # quantizers for model input
      source_quantizers=[quantizers.quantized_bits(8, 0, 1)],
      is_inference=False,
      # absolute path (including filename) of the model weights
      weights_path=None,
      # keras_quantizer to quantize weight/bias in un-quantized keras layers
      keras_quantizer=reference_internal,
      # keras_quantizer to quantize MAC in un-quantized keras layers
      keras_accumulator=reference_accumulator,
      # whether calculate baseline energy
      for_reference=True)

  # caculate energy of the derived data type map.
  ref_energy_dict = q.pe(
      # whether to store parameters in dram, sram, or fixed
      weights_on_memory="sram",
      # store activations in dram or sram
      activations_on_memory="sram",
      # minimum sram size in number of bits
      min_sram_size=8*16*1024*1024,
      # whether load data from dram to sram (consider sram as a cache
      # for dram. If false, we will assume data will be already in SRAM
      rd_wr_on_io=False)

  # get stats of energy distribution in each layer
  reference_energy_profile = q.extract_energy_profile(
      qtools_settings.cfg.include_energy, ref_energy_dict)
  # extract sum of energy of each layer according to the rule specified in
  # qtools_settings.cfg.include_energy
  total_reference_energy = q.extract_energy_sum(
      qtools_settings.cfg.include_energy, ref_energy_dict)
  print("Baseline energy profile:", reference_energy_profile)
  print("Total baseline energy:", total_reference_energy)

  # By setting for_reference=False, we quantize the model using quantizers
  # specified by users in qkeras layers. For hybrid models where there are
  # mixture of unquantized keras layers and quantized qkeras layers, we use
  # keras_quantizer to quantize weights/bias and keras_accumulator to quantize
  # MAC variables for all keras layers.
  q = run_qtools.QTools(
      model, process="horowitz",
      source_quantizers=[quantizers.quantized_bits(8, 0, 1)],
      is_inference=False, weights_path=None,
      keras_quantizer=reference_internal,
      keras_accumulator=reference_accumulator,
      for_reference=False)
  trial_energy_dict = q.pe(
      weights_on_memory="sram",
      activations_on_memory="sram",
      min_sram_size=8*16*1024*1024,
      rd_wr_on_io=False)
  trial_energy_profile = q.extract_energy_profile(
      qtools_settings.cfg.include_energy, trial_energy_dict)
  total_trial_energy = q.extract_energy_sum(
      qtools_settings.cfg.include_energy, trial_energy_dict)
  print("energy profile:", trial_energy_profile)
  print("Total energy:", total_trial_energy)
