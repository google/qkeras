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
"""Calculate energy consumption of a given quantized model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from qkeras.qtools.generate_layer_data_type_map import KERAS_LAYERS
from qkeras.qtools.generate_layer_data_type_map import QKERAS_LAYERS
from qkeras.qtools.quantized_operators.quantizer_impl import IQuantizer
from qkeras.qtools.settings import cfg

# Model based on：
#   Mark Horowitz, Computing’s Energy Problem (and what we can
#   do about it). IEEE ISSCC, pp. 10–14, 2014
#   www.youtube.com/watch?v=eZdOkDtYMoo&feature=youtu.be&t=497

# all metrics converted to pJ/bit

OP = {
    "fp32": {
        "add": lambda x: max(cfg.fp32_add(x), 0),
        "mul": lambda x: max(cfg.fp32_mul(x), 0)
    },
    "fp16": {
        "add": lambda x: max(cfg.fp16_add(x), 0),
        "mul": lambda x: max(cfg.fp16_mul(x), 0)
    },
    "fpm": {
        "add": lambda x: max(cfg.fpm_add(x), 0),
        "mux": lambda x: max(cfg.fpm_add(x), 0),
        "xor": lambda x: max(cfg.fpm_add(x), 0),
        "and": lambda x: max(cfg.fpm_add(x), 0),
        "or": lambda x: max(cfg.fpm_add(x), 0),
        "shifter": lambda x: max(cfg.fpm_add(x), 0),
        "mul": lambda x: max(cfg.fpm_mul(x), 0)
    },
    "sram": {"rd": lambda x: max(cfg.sram_rd(x), 0),
             "wr": lambda x: max(cfg.sram_rd(x), 0),
             "mul_factor": cfg.sram_mul_factor},
    "dram": {"rd": lambda x: max(cfg.dram_rd(x), 0),
             "wr": lambda x: max(cfg.dram_rd(x), 0),
             "mul_factor": cfg.dram_mul_factor}
}


def get_op_type(quantizer):
  assert isinstance(quantizer, IQuantizer)

  if quantizer.is_floating_point:
    return "fp" + str(quantizer.bits)
  else:
    return "fpm"


def memory_read_energy(is_input_layer, tensor_shape, mode, min_sram_size,
                       rd_wr_on_io, quantizer_bits, is_tensor=True):
  """compute energy to bring tensors from DRAM to SRAM."""

  if is_input_layer:
    if rd_wr_on_io:
      mode = "dram"
    else:
      mode = "sram"

  energy_mem = 0

  if is_tensor:
    tensor_shape = tensor_shape[1:]

  total_bits = np.prod(tensor_shape) * quantizer_bits
  total_bits_log2 = np.log2(max(total_bits, min_sram_size))

  if mode == "dram":
    # load input from dram; wx_sizes[1]-> input x quantizer bits
    # total_bits * 20
    energy_mem += OP["dram"]["rd"](total_bits)
    if rd_wr_on_io:
      # write input to sram
      # total_bits * sqrt(data_size/2^18)*0.3125
      # bits1 = total_bits * OP["sram"]["mul_factor"](np.prod(tensor_shape))
      # energy_mem += OP["sram"]["wr"](bits1)
      energy_mem += (
          np.ceil(total_bits * OP["sram"]["mul_factor"]) *
          OP["sram"]["wr"](total_bits_log2)
      )
  elif mode == "sram":
    # read input from sram
    # total_bits * sqrt(data_size/2^18)*0.3125
    # bits1 = total_bits * OP["sram"]["mul_factor"](np.prod(tensor_shape))
    # energy_mem += OP["sram"]["rd"](bits1)
    energy_mem += (
        np.ceil(total_bits * OP["sram"]["mul_factor"]) *
        OP["sram"]["rd"](total_bits_log2)
    )

  return energy_mem


def parameter_read_energy(
    layer, layer_item, weights_on_memory, min_sram_size, rd_wr_on_io):
  """read weights/bias from memory."""

  node_type = layer.__class__.__name__
  rd_energy = 0
  if node_type in ["QBatchNormalization", "BatchNormalization"]:
    gamma_quantizer = layer_item["gamma_quantizer"]
    beta_quantizer = layer_item["beta_quantizer"]
    mean_quantizer = layer_item["mean_quantizer"]
    variance_quantizer = layer_item["variance_quantizer"]

    # gamma, beta, mean, stddev
    weights = layer.get_weights()
    s = len(weights[0])
    for q in [gamma_quantizer, beta_quantizer, mean_quantizer,
              variance_quantizer]:
      if q:
        rd_energy += memory_read_energy(
            False, (s), weights_on_memory, min_sram_size, rd_wr_on_io,
            q.bits, is_tensor=False)

  elif node_type in QKERAS_LAYERS or node_type in KERAS_LAYERS:
    weight_quantizer = layer_item.weight_quantizer
    w_shapes = layer_item.w_shapes
    bias_quantizer = layer_item.bias_quantizer
    b_shapes = layer_item.b_shapes

    rd_energy += memory_read_energy(
        False, w_shapes, weights_on_memory, min_sram_size, rd_wr_on_io,
        weight_quantizer.bits, is_tensor=False
    )

    if bias_quantizer:
      # if use_bias=0, no bias
      bias_shapes = (b_shapes)
      rd_energy += memory_read_energy(
          False, bias_shapes, weights_on_memory, min_sram_size, rd_wr_on_io,
          bias_quantizer.bits, is_tensor=False
      )

  return rd_energy


def memory_write_energy(is_output_layer, tensor_shape, mode, min_sram_size,
                        rd_wr_on_io, quantizer_bits):
  """compute energy to bring tensors from SRAM to DRAM."""
  if is_output_layer:
    if rd_wr_on_io:
      mode = "dram"
    else:
      mode = "sram"

  energy_mem = 0

  tensor_shape = tensor_shape[1:]

  total_bits = np.prod(tensor_shape) * quantizer_bits
  total_bits_log2 = np.log2(max(total_bits, min_sram_size))

  if mode == "dram":
    # load input from dram; wx_sizes[1]-> input x quantizer bits
    if rd_wr_on_io:
      # read input from sram
      # total_bits * sqrt(data_size/2^18)*0.3125
      # bits1 = total_bits * OP["sram"]["mul_factor"](np.prod(tensor_shape))
      # energy_mem += OP["sram"]["rd"](bits1)
      energy_mem += (
          np.ceil(total_bits * OP["sram"]["mul_factor"]) *
          OP["sram"]["rd"](total_bits_log2)
      )
    # write output to dram
    energy_mem += OP["dram"]["wr"](total_bits)

  elif mode == "sram":
    # write to sram
    # total_bits * sqrt(data_size/2^18)*0.3125
    # bits1 = total_bits * OP["sram"]["mul_factor"](np.prod(tensor_shape))
    # energy_mem +=  OP["sram"]["wr"](bits1)
    energy_mem += (
        np.ceil(total_bits * OP["sram"]["mul_factor"]) *
        OP["sram"]["wr"](total_bits_log2)
    )

  return energy_mem


def energy_estimate(model, layer_map, weights_on_memory,
                    activations_on_memory, min_sram_size,
                    rd_wr_on_io):
  """estimate energy."""

  output_layers = layer_map["output_layers"]
  input_layers = layer_map["input_layers"]
  layer_data_type_map = layer_map["layer_data_type_map"]

  result = {}
  total_energy = 0

  # compute MAC and memory access energy for intermediate layers
  for layer in model.layers:
    if layer not in layer_data_type_map.keys():
      continue

    layer_item = layer_data_type_map[layer]

    if hasattr(layer_item, "input_quantizer_list"):
      input_quantizer_list = layer_item.input_quantizer_list
      operation_count = layer_item.operation_count
      output_shapes = layer_item.output_shapes
      output_quantizer = layer_item.output_quantizer
    else:
      input_quantizer_list = layer_item["input_quantizer_list"]
      operation_count = layer_item["operation_count"]
      output_shapes = layer_item["output_shapes"]
      output_quantizer = layer_item["output_quantizer"]

    is_input_layer = layer in input_layers
    is_output_layer = layer in output_layers

    input_rd_energy = 0
    energy_op = 0
    input_shape = layer.input_shape
    if not isinstance(input_shape, list):
      input_shape = [input_shape]

    for (input_shape, input_quantizer) in zip(
        input_shape, input_quantizer_list):
      input_rd_energy += memory_read_energy(
          is_input_layer, input_shape,
          activations_on_memory, min_sram_size, rd_wr_on_io,
          input_quantizer.bits)

    parameter_rd_energy = parameter_read_energy(
        layer, layer_item, weights_on_memory, min_sram_size,
        rd_wr_on_io)

    output_wr_energy = memory_write_energy(
        is_output_layer, output_shapes,
        activations_on_memory, min_sram_size, rd_wr_on_io,
        output_quantizer.bits)
    # QActivation Layer
    if layer.__class__.__name__ in ["QActivation", "QAdaptiveActivation", "Activation"]:
      pass

    # QBN Layer
    elif layer.__class__.__name__ in [
        "QBatchNormalization", "BatchNormalization"]:
      # assume QBN is embedded with conv/dense layers
      # -> no memory read/write cost

      divider = layer_item["internal_divide_quantizer"]
      if divider:
        gate_factor = divider.gate_factor
        mode = divider.implemented_as()
        energy_op += gate_factor * OP[
            get_op_type(divider.output)][mode](divider.gate_bits)

      multiplier = layer_item["internal_multiplier"]
      if multiplier:
        gate_factor = multiplier.gate_factor
        mode = multiplier.implemented_as()
        energy_op += gate_factor * OP[
            get_op_type(multiplier.output)][mode](multiplier.gate_bits)

      energy_op *= operation_count

    # Merge layer
    elif layer.__class__.__name__ in ["Add", "Multiply", "Subtract"]:

      # multiply or add operation energy
      # TODO(lishanok): check energy for concatenate
      merge_quantizer = layer_item.multiplier
      mode = merge_quantizer.implemented_as()
      number_of_inputs = len(layer_item.input_quantizer_list)
      gate_factor = merge_quantizer.gate_factor

      q = get_op_type(merge_quantizer.output)
      b = merge_quantizer.gate_bits
      energy_op = (number_of_inputs - 1) * operation_count * gate_factor * OP[
          q][mode](b)

    # AveragePooling and GlobalAveragePooling
    elif layer.__class__.__name__ in [
        "AveragePooling2D", "AvgPool2D", "GlobalAvgPool2D",
        "GlobalAveragePooling2D"]:
      # accumulation operation energy
      accumulator = layer_item.accumulator
      add_energy = OP[get_op_type(accumulator.output)]["add"](
          accumulator.output.bits)
      energy_op = operation_count * add_energy

    # MAC energy calculation
    elif layer.__class__.__name__ in ["QConv2D", "QConv1D", "QDepthwiseConv2D",
                                      "QDense", "Conv2D", "Conv1D",
                                      "DepthwiseConv2D", "Dense"]:
      multiplier = layer_item.multiplier
      accumulator = layer_item.accumulator

      # implementation mode: xor/andgate/shift etc.
      mode = multiplier.implemented_as()
      gate_factor = multiplier.gate_factor

      op = get_op_type(multiplier.output)
      bits = multiplier.gate_bits
      c1 = gate_factor * OP[op][mode](bits)
      c2 = OP[get_op_type(accumulator.output)]["add"](accumulator.output.bits)
      energy_op = operation_count * (c1 + c2)

    else:
      pass

    result[layer.name] = {
        "class_name": layer.__class__.__name__,
        "energy": {
            "inputs": float("{0:.2f}".format(input_rd_energy)),
            "outputs": float("{0:.2f}".format(output_wr_energy)),
            "parameters": float("{0:.2f}".format(parameter_rd_energy)),
            "op_cost": float("{0:.2f}".format(energy_op))
        }
    }
    total_energy += input_rd_energy + output_wr_energy + parameter_rd_energy + energy_op

  result["total_cost"] = int(total_energy)

  return result
