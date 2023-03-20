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
"""Interface for running qtools and qenergy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

from qkeras.qtools import generate_layer_data_type_map
from qkeras.qtools import interface
from qkeras.qtools import qgraph
from qkeras.qtools import qtools_util
from qkeras.qtools.config_public import config_settings
from qkeras.qtools.qenergy import qenergy
from qkeras.qtools.settings import cfg


class QTools:
  """integration of different qtools functions."""

  def __init__(self, model, process, source_quantizers=None,
               is_inference=False, weights_path=None,
               keras_quantizer=None, keras_accumulator=None,
               for_reference=False,
               model_weights_already_quantized=True,
               hw_weight_dict=None):

    if model is not None:
      self._model = model

    if weights_path is not None:
      self._model.load_weights(weights_path)

    cfg.update(process, config_settings)

    # if source_quantizers is None, CreateGraph will use
    # default_source_quantizers defined in cfg
    (graph, source_quantizer_list) = qgraph.CreateGraph(
        model, source_quantizers, cfg.default_source_quantizer)

    # qgraph.PrintGraph(graph)
    qgraph.GraphPropagateActivationsToEdges(graph)
    self._layer_map = generate_layer_data_type_map.generate_layer_data_type_map(
        graph, source_quantizer_list, is_inference,
        keras_quantizer, keras_accumulator, for_reference,
        model_weights_already_quantized=model_weights_already_quantized,
        hw_weight_dict=hw_weight_dict)

    self._output_dict = interface.map_to_json(self._layer_map)
    self.source_quantizer_list = source_quantizer_list

  def qtools_stats_to_json(self, json_name):
    """dump the layer stats to a json file."""

    with open(json_name, "w") as outfile:
      json.dump(self._output_dict, outfile, indent=4)

  def qtools_stats_print(self):
    """print out the layer stats."""

    dict_to_json = json.dumps(self._output_dict, indent=4)
    print(dict_to_json)

  def pe(self, weights_on_memory="dram",
         activations_on_memory="dram",
         min_sram_size=0,
         rd_wr_on_io=True,
         verbose=False):
    """energy consumption calculation."""

    assert weights_on_memory in ["dram", "sram", "fixed"]
    energy_dict = qenergy.energy_estimate(
        self._model, self._layer_map, weights_on_memory,
        activations_on_memory, min_sram_size,
        rd_wr_on_io)

    if verbose:
      print("COST:")
      dict_to_json = json.dumps(energy_dict, indent=4)
      print(dict_to_json)

    return energy_dict

  def extract_energy_sum(self, cfg_setting, energy_dict):
    """extracted energy needed in caculating sum."""

    value = 0
    for layer in energy_dict.keys():
      if layer == "total_cost":
        continue

      class_name = energy_dict[layer]["class_name"]
      keys = cfg_setting.get(class_name, cfg_setting.get("default", []))
      value += sum([energy_dict[layer]["energy"][key] for key in keys])

    return int(value)

  def extract_energy_profile(self, cfg_setting, energy_dict):
    """extract energy consumption in each layer."""

    energy = {}
    for layer in energy_dict.keys():
      if layer == "total_cost":
        continue

      class_name = energy_dict[layer]["class_name"]
      keys = cfg_setting.get(class_name, cfg_setting.get("default", []))
      energy[layer] = {}
      energy[layer]["energy"] = energy_dict[layer]["energy"]
      energy[layer]["total"] = sum(
          [energy_dict[layer]["energy"][key] for key in keys])

    return energy

  def calculate_ace(self, default_float_bits):
    """Computes ACE numbers from conv/dense layers."""

    def _get_ace(layer):
      ace = 0
      ace_float = 0
      if layer.name in self._output_dict:
        layer_item = self._output_dict[layer.name]
        # Here we only consider the number of multiplication as the
        # operation count. To include the number of
        # accumulators, we should multiply the value by 2, assuming
        # accumulation count ~= multiplication count.
        operation_count = layer_item["operation_count"]

        # Input bitwidth.
        input_quantizer_list = layer_item["input_quantizer_list"]
        input_bits = input_quantizer_list[0]["bits"]

        # Weight bitwidth
        weight_quantizer = qtools_util.get_val(layer_item, "weight_quantizer")
        if weight_quantizer:
          # Only layers such as Conv/Dense have weight_quantizers.
          w_bits = weight_quantizer["bits"]
          ace = operation_count * input_bits * w_bits
          ace_float = operation_count * default_float_bits * default_float_bits
      return (ace, ace_float)

    print("WARNING: ACE are computed from conv/dense layers only!")
    return (sum([_get_ace(l)[0] for l in self._model.layers]),
            sum([_get_ace(l)[1] for l in self._model.layers]))

  def calculate_output_bytes(self, include_model_input_size,
                             default_float_bits):
    """Computes activation layers' output size in bytes."""

    def _get_activation_size(layer):
      # Since in hardare previous conv/dense layers will be fused with
      # the following activation layers, we only consider the output of
      # Activation layers when calculating output sizes.
      if layer.__class__.__name__ in ["QActivation"]:
        layer_item = self._output_dict[layer.name]

        output_quantizer = layer_item["output_quantizer"]
        output_shape = output_quantizer["shape"]
        o_bits = output_quantizer["bits"]
        return (int(np.prod(output_shape[1:]) * o_bits / 8.0),
                int(np.prod(output_shape[1:]) * default_float_bits / 8.0))
      else:
        return (0, 0)

    output_bytes = sum([_get_activation_size(l)[0] for l in self._model.layers])
    output_bytes_float = sum([_get_activation_size(l)[1] for l in
                              self._model.layers])
    if include_model_input_size:
      # Include model input size.
      output_bytes += (np.prod(self._model.input_shape[1:])
                       * self.source_quantizer_list[0].bits / 8.0)
      output_bytes_float += (np.prod(self._model.input_shape[1:]) *
                             default_float_bits/ 8.0)

    return (output_bytes, output_bytes_float)

  def calculate_weight_bytes(self, default_float_bits):
    """Computes weight size in bytes from conv/dense layers."""

    def _get_weight_size(layer):
      weight_bytes = 0
      weight_bytes_float = 0

      if layer.name in self._output_dict:
        layer_item = self._output_dict[layer.name]
        weight_quantizer = qtools_util.get_val(layer_item, "weight_quantizer")

        if weight_quantizer:
          # Calculates kernel bytes.
          w_bits = weight_quantizer["bits"]
          weight_bytes += int(np.prod(layer.weights[0].shape) * w_bits / 8.0)
          weight_bytes_float += int(np.prod(layer.weights[0].shape) *
                                    default_float_bits / 8.0)
          # Calculates bias bytes.
          if hasattr(layer, "use_bias") and layer.use_bias:
            bias_quantizer = qtools_util.get_val(layer_item, "bias_quantizer")

            assert bias_quantizer is not None, (
                f"{layer.name} has no bias_quantizer!")
            b_bits = bias_quantizer["bits"]
            weight_bytes += int(np.prod(layer.weights[1].shape) * b_bits / 8.0)
            weight_bytes_float += int(np.prod(layer.weights[1].shape) *
                                      default_float_bits / 8.0)
      return (weight_bytes, weight_bytes_float)

    return (sum([_get_weight_size(l)[0] for l in self._model.layers]),
            sum([_get_weight_size(l)[1] for l in self._model.layers]))

  def get_roofline_numbers(self, include_model_input_size=True,
                           default_float_bits=32):
    """Extracts model numbers for roofline model analysis."""

    return {"ACE": self.calculate_ace(default_float_bits)[0],
            "weight_in_bytes": self.calculate_weight_bytes(
                default_float_bits)[0],
            "activation_in_bytes": self.calculate_output_bytes(
                include_model_input_size, default_float_bits)[0],
            "ACE_float": self.calculate_ace(
                default_float_bits)[1],
            "weight_in_bytes_float": self.calculate_weight_bytes(
                default_float_bits)[1],
            "activation_in_bytes_float": self.calculate_output_bytes(
                include_model_input_size, default_float_bits)[1]}


