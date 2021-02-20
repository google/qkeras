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
import tensorflow as tf


Q_SEQUENCE_LAYERS = ["QSimpleRNN", "QLSTM", "QGRU", "QBidirectional"]

def print_qmodel_summary(q_model):
  """Prints quantized model summary."""

  for layer in q_model.layers:
    if (layer.__class__.__name__ == "QActivation" or
        layer.__class__.__name__ == "QAdaptiveActivation"):
      print("{:20} {}".format(layer.name, str(layer.activation)))
    elif (
        hasattr(layer, "get_quantizers") and
        layer.__class__.__name__ != "QBatchNormalization"
    ):
      print("{:20} ".format(layer.name), end="")
      if "Dense" in layer.__class__.__name__:
        print("u={} ".format(layer.units), end="")
      elif layer.__class__.__name__ in [
          "Conv2D", "QConv2D", "Conv1D", "QConv1D",
          "QConv2DBatchnorm", "QDepthwiseConv2DBatchnorm"]:
        print("f={} ".format(layer.filters), end="")
      quantizers = layer.get_quantizers()
      for q in range(len(quantizers)):
        if quantizers[q] is not None:
          print("{} ".format(str(quantizers[q])), end="")
      if hasattr(layer, "recurrent_activation"):
        print("recurrent act={}".format(layer.recurrent_activation), end="")
      if (
          layer.activation is not None and
          not (
              hasattr(layer.activation, "__name__") and
              layer.activation.__name__ == "linear"
          )
      ):
        print("act={}".format(layer.activation), end="")
      print()
    elif layer.__class__.__name__ == "QBatchNormalization":
      print("{:20} QBN, mean={}".format(layer.name,
          str(tf.keras.backend.eval(layer.moving_mean))), end="")
      print()
    elif layer.__class__.__name__ == "BatchNormalization":
      print("{:20} is normal keras bn layer".format(layer.name), end="")
      print()

  print()


def get_quantization_dictionary(q_model):
  """Returns quantization dictionary."""

  q_dict = {}
  for layer in q_model.layers:
    if hasattr(layer, "get_quantization_config"):
      q_dict[layer.name] = layer.get_quantization_config()

  return q_dict


def save_quantization_dict(fn, q_model):
  """Saves quantization dictionary as json object in disk."""
  q_dict = get_quantization_dictionary(q_model)
  json_dict = json.dumps(q_dict)

  f = open(fn, "w")
  f.write(json_dict + "\n")
  f.close()

