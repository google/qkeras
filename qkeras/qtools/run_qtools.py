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
"""Interface for running qtools and qenergy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from qkeras.qtools import generate_layer_data_type_map
from qkeras.qtools import interface
from qkeras.qtools import qgraph
from qkeras.qtools.config_public import config_settings
from qkeras.qtools.qenergy import qenergy
from qkeras.qtools.settings import cfg


class QTools:
  """integration of different qtools functions."""

  def __init__(self, model, process, source_quantizers=None,
               is_inference=False, weights_path=None,
               keras_quantizer=None, keras_accumulator=None,
               for_reference=False):

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
        keras_quantizer, keras_accumulator, for_reference)

    self._output_dict = interface.map_to_json(self._layer_map)

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
