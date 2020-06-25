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
"""I/O implementation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from qkeras.qtools import generate_layer_data_type_map
from qkeras.qtools import qtools_util


def print_qstats(graph):
  """Prints quantization statistics for the model."""

  layer_data_type_map = generate_layer_data_type_map(graph)

  multipliers_counter = collections.Counter()

  print("")
  print("Number of operations in model:")
  for layer, data_type in layer_data_type_map.items():
    multiplier = data_type.multiplier
    multiplier_detail_str = "{}_{}, total_bits:{}, int_bits:{}".format(
        "signed" if multiplier.output.is_signed == 1 else "unsigned",
        multiplier.implemented_as(),
        multiplier.output.bits,
        multiplier.output.int_bits,
    )

    print("{}: {} x {}".format(
        layer.name,
        data_type.operation_count,
        multiplier_detail_str,
    ))

    multipliers_counter[
        multiplier_detail_str] += data_type.operation_count

  print("")
  print("Number of operation types in model:")
  for (multiplier_detail_str,
       total_multiplier_operation_count) in multipliers_counter.items():
    print("{}, x {}".format(multiplier_detail_str,
                            total_multiplier_operation_count))


def populate_quantizer(quantizer, shape=None, implemented_as=None):
  """write all the needed fields in the quantizer to dictionary."""

  mydict = collections.OrderedDict()

  if quantizer is not None:
    mydict["quantizer_type"] = quantizer.name

    # floats
    if quantizer.is_floating_point:
      mydict["bits"] = quantizer.bits

    # po2
    elif quantizer.is_po2:
      mydict["bits"] = quantizer.bits
      mydict["is_signed"] = quantizer.is_signed
      mydict["max_value"] = quantizer.max_val_po2

    # binary
    elif quantizer.mode in [3, 4]:

      mydict["bits"] = quantizer.bits
      mydict["int_bits"] = quantizer.int_bits
      mydict["is_signed"] = quantizer.is_signed
      if quantizer.mode == 4:
        mydict["values"] = [0, 1]
      else:
        mydict["values"] = [-1, 1]

    # ternary(-1, 0, 1)
    elif quantizer.mode == 2:
      mydict["bits"] = 2
      mydict["int_bits"] = 2
      mydict["is_signed"] = 1
      mydict["values"] = [0, -1, 1]

    # quantized_bits
    elif quantizer.mode == 0:
      mydict["bits"] = quantizer.bits
      mydict["int_bits"] = quantizer.int_bits + quantizer.is_signed
      mydict["is_signed"] = quantizer.is_signed

    if shape is not None:
      if isinstance(shape, tuple) and shape[0] is None:
        shape = list(shape)
        shape[0] = -1
        mydict["shape"] = tuple(shape)
      else:
        mydict["shape"] = shape

    if implemented_as is not None:
      mydict["op_type"] = implemented_as
  return mydict


def map_to_json(mydict):
  """write the dictionary to json format."""

  source_quantizer_list = mydict["source_quantizer_list"]
  layer_data_type_map = mydict["layer_data_type_map"]

  output_dict = collections.OrderedDict()

  q_list = []
  for source_quantizer in source_quantizer_list:
    tmp = populate_quantizer(source_quantizer)
    q_list.append(tmp)

  if bool(q_list):
    output_dict["source_quantizers"] = q_list

  for layer, feature in layer_data_type_map.items():
    layer_item = collections.OrderedDict()
    layer_item["layer_type"] = layer.__class__.__name__

    if layer_item["layer_type"] in [
        "QBatchNormalization", "BatchNormalization"]:
      layer_item["input_quantizer_list"] = [
          populate_quantizer(q) for q in feature["input_quantizer_list"]]

      if feature["gamma_quantizer"]:
        layer_item["gamma_quantizer"] = populate_quantizer(
            feature["gamma_quantizer"])

      if feature["beta_quantizer"]:
        layer_item["beta_quantizer"] = populate_quantizer(
            feature["beta_quantizer"])

      if feature["mean_quantizer"]:
        layer_item["mean_quantizer"] = populate_quantizer(
            feature["mean_quantizer"])

      if feature["variance_quantizer"]:
        layer_item["variance_quantizer"] = populate_quantizer(
            feature["variance_quantizer"])

      if feature["internal_divide_quantizer"]:
        layer_item["internal_divide_quantizer"] = populate_quantizer(
            feature["internal_divide_quantizer"].output,
            implemented_as=feature[
                "internal_divide_quantizer"].implemented_as())

      if feature["internal_multiplier"]:
        layer_item["internal_multiplier"] = populate_quantizer(
            feature["internal_multiplier"].output,
            implemented_as=feature[
                "internal_multiplier"].implemented_as())

      if feature["internal_accumulator"]:
        layer_item["internal_accumulator"] = populate_quantizer(
            feature["internal_accumulator"].output,
            implemented_as=feature["internal_accumulator"].implemented_as())

      if feature["output_quantizer"]:
        layer_item["output_quantizer"] = populate_quantizer(
            feature["output_quantizer"], shape=feature["output_shapes"])

    else:
      # populate the feature to dictionary
      layer_item["input_quantizer_list"] = [
          populate_quantizer(q) for q in feature.input_quantizer_list]

      tmp = populate_quantizer(feature.weight_quantizer, feature.w_shapes)
      if bool(tmp):
        layer_item["weight_quantizer"] = tmp

      tmp = populate_quantizer(feature.bias_quantizer, feature.b_shapes)
      if bool(tmp):
        layer_item["bias_quantizer"] = tmp

      if feature.multiplier:
        method = feature.multiplier.implemented_as()
        tmp = populate_quantizer(
            feature.multiplier.output,
            implemented_as=method)
        if bool(tmp):
          if qtools_util.is_merge_layers(layer):
            qname = layer.__class__.__name__ + "_quantizer"
            layer_item[qname] = tmp
          else:
            layer_item["multiplier"] = tmp

      if feature.accumulator:
        tmp = populate_quantizer(
            feature.accumulator.output,
            implemented_as=feature.accumulator.implemented_as())
        if bool(tmp):
          layer_item["accumulator"] = tmp

      tmp = populate_quantizer(feature.output_quantizer,
                               feature.output_shapes)
      if bool(tmp):
        layer_item["output_quantizer"] = tmp

      layer_item["operation_count"] = feature.operation_count

    output_dict[layer.name] = layer_item

  return output_dict
