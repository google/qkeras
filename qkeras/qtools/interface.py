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

  def get_val(feature, key):
    if isinstance(feature, dict):
      val = feature.get(key, None)
    else:
      val = getattr(feature, key, None)
    return val

  def set_layer_item(layer_item, key, feature, shape=None,
                     is_compound_datatype=False, output_key_name=None):

    val = get_val(feature, key)
    if val is not None:
      quantizer = val
      implemented_as = None
      if is_compound_datatype:
        quantizer = val.output
        implemented_as = val.implemented_as()
      if output_key_name is None:
        key_name = key
      else:
        key_name = output_key_name
      tmp = populate_quantizer(
          quantizer, shape=shape, implemented_as=implemented_as)
      if bool(tmp):
        layer_item[key_name] = tmp

  for layer, feature in layer_data_type_map.items():
    layer_item = collections.OrderedDict()
    layer_item["layer_type"] = layer.__class__.__name__
    layer_item["input_quantizer_list"] = [
        populate_quantizer(q) for q in get_val(feature, "input_quantizer_list")]

    set_layer_item(layer_item, key="output_quantizer", feature=feature,
                   shape=get_val(feature, "output_shapes"))

    if layer_item["layer_type"] in [
        "QBatchNormalization", "BatchNormalization"]:

      for key in ["gamma_quantizer", "beta_quantizer", "mean_quantizer",
                  "variance_quantizer", "variance_quantizer"]:
        set_layer_item(layer_item, key=key, feature=feature)

      for key in ["internal_divide_quantizer", "internal_divide_quantizer",
                  "internal_multiplier", "internal_accumulator"]:
        set_layer_item(layer_item, key=key, feature=feature,
                       is_compound_datatype=True)
    else:
      # populate the feature to dictionary
      set_layer_item(layer_item, key="weight_quantizer", feature=feature,
                     shape=get_val(feature, "w_shapes"))
      set_layer_item(layer_item, key="bias_quantizer", feature=feature,
                     shape=get_val(feature, "b_shapes"))

      output_key_name = None
      if qtools_util.is_merge_layers(layer):
        output_key_name = layer.__class__.__name__ + "_quantizer"
      set_layer_item(layer_item, key="multiplier", feature=feature,
                     is_compound_datatype=True,
                     output_key_name=output_key_name)
      set_layer_item(layer_item, key="accumulator", feature=feature,
                     is_compound_datatype=True)

      if get_val(feature, "fused_accumulator"):
        # Add fused weights to the dictionary
        for key in ["bn_beta_quantizer", "bn_mean_quantizer",
                    "bn_inverse_quantizer"]:
          set_layer_item(layer_item, key=key, feature=feature)

        set_layer_item(layer_item, key="fused_accumulator", feature=feature,
                       is_compound_datatype=True)

      layer_item["operation_count"] = get_val(feature, "operation_count")

    output_dict[layer.name] = layer_item

  return output_dict
