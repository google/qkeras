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
"""utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import tensorflow.keras.backend as K


def is_shape_alternation_layers(layer):
  lname = layer.__class__.__name__
  if lname:
    return "MaxPool" in lname or "Reshape" in lname or "Flatten" in lname
  return False


def is_merge_layers(layer):

  if layer.__class__.__name__ in [
      "Add", "Multiply", "Subtract", "Average", "Maximum", "Minimum",
      "Concatenate", "Dot"]:
    return True
  else:
    return False


def get_input_quantizers(graph, node_id, quantizer_factory, debug=False):
  """get the current layer's input quantizer."""

  # in merge layers, therea are more than 1 input

  output = []
  for parent_node_id in graph.predecessors(node_id):

    edge = graph.edges[(parent_node_id, node_id)]

    if debug:
      print("parent_node_id:", parent_node_id)
      print(edge)

    quantizer_on_edge = edge["quantizer"]
    input_quantizer = quantizer_factory.make_quantizer(quantizer_on_edge)

    output.append((input_quantizer, edge))

  return output


def get_input_quantizers_advanced(graph, node_id, is_input_layer, quantizer_factory,
                                  cfg, debug=False):
  """get input quantizer, deal with keras layer or lack of input quantizer in qkeras layer."""

  # in merge layers, therea are more than 1 input
  default_source_quantizer = cfg.default_source_quantizer
  default_interm_quantizer = cfg.default_interm_quantizer

  output = []
  for parent_node_id in graph.predecessors(node_id):

    edge = graph.edges[(parent_node_id, node_id)]

    if debug:
      print("parent_node_id:", parent_node_id)
      print(edge)

    quantizer_on_edge = edge["quantizer"]
    input_quantizer = quantizer_factory.make_quantizer(quantizer_on_edge)

    if is_input_layer and not input_quantizer:
      # input layer without input_quantizer specified->use default_source_quantizer
      input_quantizer = quantizer_factory.make_default_quantizer(
          mode=default_source_quantizer)
    elif not input_quantizer:
      # if no input quantizer is available-> use default quantizer from config.json
      input_quantizer = quantizer_factory.make_default_quantizer(
          mode=default_interm_quantizer)

    output.append((input_quantizer, edge))

  return output


def get_operation_count(layer, input_shape):
  """Determines number of multiplier operations in a qkeras layer."""

  # Check if the inputs are a list of Dimensions
  if isinstance(input_shape, list):
    input_shape = input_shape[0]

  operation_count = 0

  if is_merge_layers(layer) or is_shape_alternation_layers(layer):
    operation_count = np.prod(input_shape[1:])

  elif layer.__class__.__name__ in [
      "AveragePooling2D", "AvgPool2D", "GlobalAvgPool2D",
      "GlobalAveragePooling2D"
  ]:

    if hasattr(layer, "pool_size"):
      pool_size = layer.pool_size
    else:
      pool_size = input_shape[1:-1]
    add_ops = np.prod(pool_size)

    output_shape = layer.compute_output_shape(input_shape)
    channels_o = output_shape[-1]

    # total number of add ops
    operation_count = channels_o * add_ops

  elif "UpSampling" in layer.__class__.__name__:
    # UpSampling1D/2D/3D
    output_shape = layer.compute_output_shape(input_shape)
    operation_count = np.prod(output_shape[1:])

  elif ("Activation" in layer.__class__.__name__ or
        "BatchNormalization" in layer.__class__.__name__):
    operation_count = np.prod(input_shape[1:])

  elif layer.__class__.__name__ in ["QConv2D", "Conv2D", "QConv2DBatchnorm"]:

    output_shape = layer.compute_output_shape(input_shape)
    _, _, _, channels_i = input_shape

    _, height_o, width_o, channels_o = output_shape

    weight = layer.get_weights()[0]

    kernel_h, kernel_w, _, _ = weight.shape

    operation_count = (
        height_o * width_o * channels_o * kernel_h * kernel_w * channels_i)

  elif layer.__class__.__name__ in ["QConv1D", "Conv1D"]:
    output_shape = layer.compute_output_shape(input_shape)
    _, _, channels_i = input_shape

    _, time_o, channels_o = output_shape

    weight = layer.get_weights()[0]

    kernel_length, _, _ = weight.shape

    operation_count = (
        time_o * channels_o * kernel_length * channels_i)

  elif layer.__class__.__name__ in ["QDepthwiseConv2D", "DepthwiseConv2D"]:
    output_shape = layer.compute_output_shape(input_shape)
    _, _, _, channels_i = input_shape

    _, height_o, width_o, channels_o = output_shape

    weight_1 = layer.get_weights()[0]

    kernel_h, kernel_w, _, _ = weight_1.shape

    operation_count = (
        kernel_h * kernel_w * height_o * width_o * channels_i)

  elif layer.__class__.__name__ in ["QDense", "Dense"]:
    output_shape = layer.compute_output_shape(input_shape)
    _, size_i = input_shape
    _, size_o = output_shape

    operation_count = (size_i * size_o)

  else:
    print("operation count for {} is defaulted to 0".format(
        layer))

  return int(operation_count)


def get_weights(layer):
  weights = layer.get_weights()
  out = copy.deepcopy(weights)
  for j, weight in enumerate(weights):
    if hasattr(layer, "get_quantizers") and layer.get_quantizers()[j]:
      out[j] = K.eval(
          layer.get_quantizers()[j](K.constant(weight)))

  return out
