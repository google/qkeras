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
"""Implements utility functions for support of auto-quantization."""

import json


def print_qmodel_summary(q_model):
  """Prints quantized model summary."""

  for layer in q_model.layers:
    if layer.__class__.__name__ == "QActivation":
      print("{:20} {}".format(layer.name, str(layer.activation)))
    elif (
        hasattr(layer, "get_quantizers") and
        layer.__class__.__name__ != "QBatchNormalization"
    ):
      print("{:20} ".format(layer.name), end="")
      if "Dense" in layer.__class__.__name__:
        print("u={} ".format(layer.units), end="")
      elif layer.__class__.__name__ in [
          "Conv2D", "QConv2D", "Conv1D", "QConv1D"]:
        print("f={} ".format(layer.filters), end="")
      quantizers = layer.get_quantizers()
      for q in range(len(quantizers)):
        if quantizers[q] is not None:
          print("{} ".format(str(quantizers[q])), end="")
      if (
          layer.activation is not None and
          not (
              hasattr(layer.activation, "__name__") and
              layer.activation.__name__ == "linear"
          )
      ):
        print("act={}".format(layer.activation), end="")
      print()
  print()


def get_quantization_dictionary(q_model):
  """Returns quantization dictionary."""

  q_dict = {"QBatchNormalization": {}}
  for layer in q_model.layers:
    if layer.__class__.__name__ == "QActivation":
      q_dict[layer.name] = str(layer.activation)
    elif (
        hasattr(layer, "get_quantizers") and
        layer.__class__.__name__ != "QBatchNormalization"
    ):
      q_dict[layer.name] = {}
      if "Dense" in layer.__class__.__name__:
        q_dict[layer.name]["units"] = layer.units
      elif layer.__class__.__name__ in [
          "Conv2D", "QConv2D", "Conv1D", "QConv1D"]:
        q_dict[layer.name]["filters"] = layer.filters
      quantizers = layer.get_quantizers()
      field_names = ["kernel_quantizer", "bias_quantizer"]
      for q in range(len(quantizers)):
        if quantizers[q] is not None:
          q_dict[layer.name][field_names[q]] = str(quantizers[q])
      if (
          layer.activation is not None and
          hasattr(layer.activation, "bits")
      ):
        q_dict[layer.name]["activation"] = str(layer.activation)
  return q_dict


def save_quantization_dict(fn, q_model):
  """Saves quantization dictionary as json object in disk."""
  q_dict = get_quantization_dictionary(q_model)
  json_dict = json.dumps(q_dict)

  f = open(fn, "w")
  f.write(json_dict + "\n")
  f.close()

