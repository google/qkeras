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

"""divide_and_conquer hardware cost profiling.

Given a target throughput and a ML model, this implementation determines
the key HW design parameters (bitwidth, unroll factors) for ML area
optimization in a pipelined architecture.

It generates recommended design parameters to assist downstream HW synthesis
design. With this, it provides accurate HW cost modeling for ML training
and ML complexity evaluation such as AV2/ROOF_ML.
"""

import enum
import logging
from typing import Any, List, Union

import numpy as np
import tensorflow as tf

from qkeras import base_quantizer
from qkeras import quantizers
from qkeras.qtools import generate_layer_data_type_map
from qkeras.qtools import qgraph
from qkeras.qtools import qtools_util
from qkeras.qtools.DnC import dnc_layer_cost_ace


class CostMode(enum.Enum):
  ACE = 1  # cost is computed from theoretical equations.
  PE_AREA = 2  # cost is computed from compute area only.
  PE_BW_AREA = 3  # cost is computed from both compute and memory bandwidth.


# pylint: disable=invalid-name
class DivideConquerGraph:
  """This class creates model graph structure and methods to access layers."""

  def __init__(
      self,
      model: tf.keras.Model,
      source_quantizers: base_quantizer.BaseQuantizer = None,
  ):
    self._model = model
    self._source_quantizer_list = source_quantizers or [
        quantizers.quantized_bits(8, 0, 1)]

    (self._graph, self._source_quantizer_list) = qgraph.CreateGraph(
        model, source_quantizers, "quantized_bits(8, 0, 1)")

    # Propagate output quantizer info into the graph edges.
    qgraph.GraphPropagateActivationsToEdges(self._graph)

    self._layer_map = generate_layer_data_type_map.generate_layer_data_type_map(
        self._graph, self._source_quantizer_list, is_inference=False,
        keras_accumulator=None, for_reference=False)["layer_data_type_map"]

    # Create layer-to-index mapping dict.
    self._layer_to_idx_dict = {}
    for idx in self._graph._node.keys():
      self._layer_to_idx_dict[self.idx_to_layer(idx)] = idx

  def idx_to_layer(self, idx: int):
    # Map layer index to the layer object.
    return self._graph._node[idx]["layer"][0]

  def layer_to_idx(self, layer: tf.keras.layers.Layer):
    # Map a layer object to index.
    return self._layer_to_idx_dict.get(layer, None)

  def get_first_node(self):
    # Get the source node of the graph.
    return qgraph.SOURCE

  def is_first_node(self, node: Union[int, tf.keras.layers.Layer]):
    # Find whether a given node is the first node of the graph.
    # Node could be either index value or layer object.
    idx = node if isinstance(node, int) else self.layer_to_idx(node)
    return idx == qgraph.SOURCE

  def get_last_node(self):
    # Find the last node of the graph.
    return qgraph.SINK

  def is_last_node(self, node: Union[int, tf.keras.layers.Layer]):
    # Find whether a given node is the last node of the graph.
    # Node could be either index value or layer object.
    idx = node if isinstance(node, int) else self.layer_to_idx(node)
    return idx == qgraph.SINK

  def get_prev_nodes(self, node: Union[int, tf.keras.layers.Layer]):
    # Find the predecessor nodes in the graph of the given node.
    # Node could be either index value or layer object.
    idx = node if isinstance(node, int) else self.layer_to_idx(node)
    return list(self._graph.predecessors(idx))

  def get_next_nodes(self, node: Union[int, tf.keras.layers.Layer]):
    # Find the successor nodes in the graph of the given node.
    # node could be either index value or layer object.
    idx = node if isinstance(node, int) else self.layer_to_idx(node)
    return list(self._graph.successors(idx))

  def get_layer_quantizer_bitwidth(
      self, node: Union[int, tf.keras.layers.Layer]):
    """Find various quantizer bitwidth of the current layer."""
    layer = self.idx_to_layer(node) if isinstance(node, int) else node

    if layer:
      layer_item = self._layer_map[layer]
      weight_quantizer = qtools_util.get_val(layer_item, "weight_quantizer")
      mac_quantizer = qtools_util.get_val(layer_item, "multiplier")
      acc_quantizer = qtools_util.get_val(layer_item, "accumulator")
      input_quantizer_list = qtools_util.get_val(
          layer_item, "input_quantizer_list")
      output_quantizer = qtools_util.get_val(layer_item, "output_quantizer")

      return  {
          # TODO(lishanok@): Handle multiple input quantizers
          # in non-sequential models.
          "input_bits": input_quantizer_list[0].bits,
          # When the current layer has no concept of weight, there won't
          # be any weight quantizer.
          "weight_bits": weight_quantizer.bits if weight_quantizer else 0,
          # If mac bits don't exist, that means we don't have x * w type of
          # operations. In this case, pass input_bits through.
          "mac_bits": (
              mac_quantizer.output.bits if mac_quantizer else
              input_quantizer_list[0].bits),
          "acc_bits": (
              acc_quantizer.output.bits if acc_quantizer else
              input_quantizer_list[0].bits),
          "output_bits": output_quantizer.bits}
    else:
      # For the "dummy" head and tail nodes in the graph that we inserted at
      # the begining and ending of the model graph, we run this branch.
      return {
          "input_bits": 0,
          "weight_bits": 0,
          "mac_bits": 0,
          "acc_bits": 0,
          "output_bits": 0
      }

  def get_layer_mac_count(self, node: Union[int, tf.keras.layers.Layer]):
    """Find the number of multiplier ops in the current layer."""
    layer = self.idx_to_layer(node) if isinstance(node, int) else node

    return (
        qtools_util.get_val(self._layer_map[layer], "operation_count", 0)
        if layer else 0)

  def get_layer_shapes(self, node: Union[int, tf.keras.layers.Layer]):
    layer = self.idx_to_layer(node) if isinstance(node, int) else node

    # Multiple inputs with merge layers.
    input_shape_list = layer.input_shape if layer else 0
    if not isinstance(input_shape_list, list):
      input_shape_list = [input_shape_list]

    return {
        "weight_shape": (
            qtools_util.get_val(self._layer_map[layer], "w_shapes", 0)
            if layer else 0),
        "output_shape": (
            qtools_util.get_val(self._layer_map[layer], "output_shapes", 0)
            if layer else 0),
        "input_shape_list": (input_shape_list)}


class Choice:
  """This class stores a combination of HW design param values."""

  def __init__(self, l: float = 0, k: float = 0, cin_unroll: int = 0,
               cout_unroll: int = 0, kh_unroll: int = 0, kw_unroll: int = 0):
    """Intializer for a combination of hardware design parameters.

    Args:
      l: Ratio between OutElementPerClk and ComputeOutElementPerClk
      k: Ratio between InElementPerClk and ComputeInElementPerClk
      cin_unroll: Unroll factors for input channel
      cout_unroll: Unroll factors for output channel
      kh_unroll: Unroll factors for kernel height
      kw_unroll: Unroll factors for kernel width
    """

    self.k = k
    self.l = l
    self.cin_unroll = cin_unroll
    self.cout_unroll = cout_unroll
    self.kh_unroll = kh_unroll
    self.kw_unroll = kw_unroll

  def __str__(self):
    return (f"Choice(k={self.k}, l={self.l}, cin_unroll={self.cin_unroll}, "
            f"cout_unroll={self.cout_unroll} kh_unroll={self.kh_unroll}, "
            f"kw_unroll={self.kw_unroll})")


def get_valid_unrolls(layer: tf.keras.layers.Layer, cout_unroll: int,
                      target_pe_throughput: float):
  """Get valid unroll values where resulting throughput>=Target throughput."""

  input_channel = qtools_util.get_layer_info(layer, "input_channel")
  output_channel = qtools_util.get_layer_info(layer, "output_channel")
  kernel_height = qtools_util.get_layer_info(layer, "kernel_height")
  kernel_width = qtools_util.get_layer_info(layer, "kernel_width")
  layer_type = qtools_util.get_layer_info(layer, "layer_type")

  if layer_type in ["QDepthwiseConv2D", "QAveragePooling2D", "MaxPooling2D",
                    "QGlobalAveragePooling2D", "GlobalMaxPooling2D"]:
    # Since ops are done in each channel without cross-channel ops,
    # cin_unroll == cout_unroll in hardware.
    cin_unroll_list = [cout_unroll]
  else:
    # Cin_unroll needs to be a divisor of layer.input_channel
    cin_unroll_list = qtools_util.find_divisors(input_channel)

  # kw_unroll needs to be a divisor of layer.kernel_width
  kw_unroll_list = qtools_util.find_divisors(kernel_width)
  # kh_unroll needs to be a divisor of layer.kernel_height
  kh_unroll_list = qtools_util.find_divisors(kernel_height)

  valid_unrolls = []
  for cin_unroll in cin_unroll_list:
    for kw_unroll in kw_unroll_list:
      for kh_unroll in kh_unroll_list:
        logging.debug("............cin_unroll: %d kh_unroll: %d kw_unroll: %d",
                      cin_unroll, kh_unroll, kw_unroll)
        # Caculate computation throughput.
        pe_throughput = get_pe_throughput(
            layer_type, cin_unroll, cout_unroll, kh_unroll, kw_unroll,
            input_channel, output_channel, kernel_height, kernel_width)
        logging.debug("............pe_throughput: %.2f", pe_throughput)
        if pe_throughput >= target_pe_throughput:
          # Save the valid combination of unroll factors to valid_unrolls.
          valid_unrolls.append((cin_unroll, kh_unroll, kw_unroll))

  return valid_unrolls


def get_per_layer_cost(layer_quantizer_bitwidth, layer_mac_count, layer_shapes,
                       cin_unroll, cout_unroll, kh_unroll, kw_unroll,
                       InElementPerClk, OutElementPerClk, mode):
  """Area per layer, including both PE and memory Bandwidth."""

  # TODO(lishanok@): needs to add modes that support data-driven cost modeling.
  assert mode == CostMode.ACE, "Only CostMode.ACE is supported for now."

  # Compute memory is calculated according to ACE metric, translated to gates.
  mac_gates = dnc_layer_cost_ace.get_ace_mac_gates(
      xbit=layer_quantizer_bitwidth["input_bits"],
      wbit=layer_quantizer_bitwidth["weight_bits"],
      abit=layer_quantizer_bitwidth["acc_bits"],
      regen_params=False)

  # pe_area is not dependent on total num of MACs in the layer.
  pe_area = (mac_gates * cin_unroll * cout_unroll * kh_unroll * kw_unroll)

  # Memory includes input, output and weight memory, translated to gates.
  # TODO(lishanok@): weights could be stored in either SRAM or ROM, dependent
  # on user specification.
  memory_area = (
      InElementPerClk * layer_quantizer_bitwidth["input_bits"] *
      dnc_layer_cost_ace.MemoryGatesPerBit["Register"] +
      OutElementPerClk * layer_quantizer_bitwidth["output_bits"] *
      dnc_layer_cost_ace.MemoryGatesPerBit["Register"] +
      np.prod(layer_shapes["weight_shape"]) *
      layer_quantizer_bitwidth["weight_bits"] *
      dnc_layer_cost_ace.MemoryGatesPerBit["ROM"])

  return (pe_area + memory_area)


def get_valid_candidates(input_value, output_to_input_ratio_max):
  candidate_list = qtools_util.find_divisors(input_value)
  # Add the other scenario where ComputeElementPerClk is multiple
  # of ElementPerClk.
  if output_to_input_ratio_max >= 2:
    candidate_list += [input_value * x for x in list(
        range(2, output_to_input_ratio_max+1))]

  return candidate_list


def get_InBufferThru(InElementPerClk, input_channel):
  return InElementPerClk / input_channel


def get_OutBufferThru(OutElementPerClk, output_channel, kernel_height,
                      kernel_width, layer_type):
  if layer_type in ["UpSampling2D"]:
    return OutElementPerClk / (
        output_channel * kernel_height * kernel_width)
  else:
    return OutElementPerClk / output_channel


def is_bufferThru_greater_than_targetThru(
    layer_type: str, InElementPerClk: int, OutElementPerClk: int,
    input_channel: int, output_channel: int, kernel_height: int,
    kernel_width: int, target_out_throughput: float,
    target_in_throughput: float):
  """Verify whether the resulting buffer throughput > target throughput."""

  # Calculate throughput of input buffer.
  InBuf_throughput = get_InBufferThru(InElementPerClk, input_channel)
  # Calculate throughput of output buffer.
  OutBuf_throughput = get_OutBufferThru(
      layer_type=layer_type,
      OutElementPerClk=OutElementPerClk, output_channel=output_channel,
      kernel_height=kernel_height, kernel_width=kernel_width)

  logging.debug(
      "...............InBuf_throughput: %.2f OutBuf_throughput: %.2f",
      InBuf_throughput, OutBuf_throughput)

  # Valid unroll values must meet buffer throughput requirements.
  return (InBuf_throughput >= target_out_throughput and
          OutBuf_throughput >= target_in_throughput)


def set_best_global_cost_in_paths(
    OutElementPerClk_list, paths, layer_idx, cur_layer_idx,
    layer_quantizer_bitwidth, layer_mac_count, layer_shapes, mode):
  """Find the best global cost of the entire model and update the paths dict.

  Args:
    OutElementPerClk_list: list of OutElementPerClk for the current layer.
    paths: Dict that contains the choices that each layer has.
    layer_idx: Int. The index value of the current layer's predecessor.
    cur_layer_idx: current layer's index value.
    layer_quantizer_bitwidth: Dict that contains layer-related quantizer
      bitwidth, including acc_bits, mac_bits, input_bits and output_bits.
    layer_mac_count: Int. Use the number of multiplication as the operation
      count. To include the number of accumulations, we should multiply the
      value by 2, assuming accumulation count ~= multiplication count.
    layer_shapes: Dict with keys: weight_shape, input_shape_list and
      output_shape.
    mode: CostMode. The mode to calculate per layer cost.

  Returns:
    None.
  """

  def calculate_cost(OutElementPerClk):
    cur_layer_cost = get_per_layer_cost(
        layer_quantizer_bitwidth, layer_mac_count, layer_shapes, 0, 0, 0, 0, 0,
        OutElementPerClk, mode)
    accumulative_cost = cur_layer_cost + paths[layer_idx][
        OutElementPerClk]["acc_cost"]
    return (cur_layer_cost, accumulative_cost, OutElementPerClk)

  cost_and_values = list(map(calculate_cost, OutElementPerClk_list))

  layer_cost, min_accumulative_cost, best_OutElementPerClk = (
      min(cost_and_values, key=lambda x: x[1]))

  # For the initial node, we find the best path which contains a sentinel
  # choice, cost with that path, and the chosen OutElementPerClk
  # that will point to the corresponding choice of the following layer.
  paths[cur_layer_idx] = {
      best_OutElementPerClk: {
          "choice": Choice().__str__(),
          "cur_cost": layer_cost,
          "acc_cost": min_accumulative_cost,
          "OutElementPerClk": best_OutElementPerClk
      }}


def backtrack(graph, paths):
  """Backtracking of the best path from the first layer to the last."""
  best_path = {}
  # Get the second node from the graph as the first node is a sentinel node.
  layer_idx = graph.get_first_node()

  logging.debug("=======================")
  logging.debug("Trimmed Paths:")
  logging.debug("paths: %s", paths)
  logging.debug("=======================")

  # Find the best choice of the first layer.
  # TODO(lishanok@): extend code to non-sequential model where there are
  # multiple input layers
  best_OutElementPerClk = list(paths[layer_idx].keys())[0]
  best_entry = paths[layer_idx][best_OutElementPerClk]
  # Add layer name to improve readability.
  layer = graph.idx_to_layer(layer_idx)
  best_entry["layer_name"] = layer.name if layer else "None"
  best_path[layer_idx] = best_entry
  best_OutElementPerClk = best_entry["OutElementPerClk"]
  best_accumlative_cost = best_entry["acc_cost"]

  layer_idx = graph.get_next_nodes(layer_idx)[0]
  # Given the best choice of 1st layer, find the best choice for all following
  # layers by backtracking.
  while not graph.is_last_node(layer_idx):
    # Find current layer's best choice from the ptr (ie. best_OutElementPerClk)
    # stored in the best choice of the previous layer.
    best_entry = paths[layer_idx][best_OutElementPerClk]
    layer = graph.idx_to_layer(layer_idx)
    best_entry["layer_name"] = layer.name if layer else "None"
    best_path[layer_idx] = best_entry
    # Update the ptr to the next layer.
    best_OutElementPerClk = best_entry["OutElementPerClk"]

    # get the next node from the graph
    # TODO(lishanok@): extend the code to non-sequential model where there are
    # multiple next layers.
    layer_idx = graph.get_next_nodes(layer_idx)[0]

  # best_path stores the best hw param combination and cost for each layer.
  return best_path, best_accumlative_cost


def update_cur_best_choices(
    cur_best_choices: List[Any], OutElementPerClk: int,
    prev_OutElementPerClk: int, cur_layer_cost: float,
    accumulative_cost: float, choice: Choice):
  """Update the cur_best_choices dict.

  At each layer, different choices of unroll factors will generate a
  prev_OutElementPerClk value. Some of the choices might generate the same
  prev_OutElementPerClk. So for each pre_OutElementPerClk, we only store
  the best choice which has the min cost.
  """

  entry = cur_best_choices.get(prev_OutElementPerClk, None)
  existing_accumulative_cost = entry["acc_cost"] if entry else np.inf
  logging.debug("...............cost of cur_best_choices [%d]: %.2f",
                prev_OutElementPerClk, existing_accumulative_cost)
  if accumulative_cost < existing_accumulative_cost:
    # Stores the best choice and its cost for the given
    # prev_OutElementPerClk. We also store the ptr to next layer's
    # OutElementPerClk for future backtracking purpose.
    cur_best_choices[prev_OutElementPerClk] = {
        "choice": choice.__str__(),
        "cur_cost": cur_layer_cost,
        "acc_cost": accumulative_cost,
        "OutElementPerClk": OutElementPerClk}
    logging.debug(
        "...............Find better cost! Update cur_best_choices[%d]: %s",
        prev_OutElementPerClk, cur_best_choices[prev_OutElementPerClk])


def get_ComputeInElementPerClk(layer_type, cin_unroll,
                               cout_unroll, kh_unroll, kw_unroll):
  if layer_type in ["QConv2D", "QDense"]:
    return cin_unroll * kh_unroll * kw_unroll
  elif layer_type in ["QDepthwiseConv2D", "QAveragePooling2D", "MaxPooling2D"]:
    return cout_unroll * kh_unroll * kw_unroll
  elif layer_type in ["QGlobalAveragePooling2D", "GlobalMaxPooling2D",
                      "UpSampling2D"]:
    return cout_unroll
  elif layer_type in ["Concatenate"]:
    return cin_unroll


def get_InElementPerClk_base(ComputInElementPerClk, kh_unroll, kw_unroll):
  return int(ComputInElementPerClk / (kh_unroll * kw_unroll))


def get_pe_throughput(layer_type, cin_unroll, cout_unroll, kh_unroll, kw_unroll,
                      input_channel, output_channel, kernel_height,
                      kernel_width):
  """Calculate compute throughput for the given unroll factors."""
  if layer_type in ["QConv2D", "QDense"]:
    return 1.0 * cin_unroll * cout_unroll * kh_unroll * kw_unroll / (
        input_channel * output_channel * kernel_height * kernel_width)
  elif layer_type in ["QDepthwiseConv2D", "QAveragePooling2D", "MaxPooling2D",
                      "UpSampling2D"]:
    return 1.0 * cout_unroll * kh_unroll * kw_unroll / (
        output_channel * kernel_height * kernel_width)
  elif layer_type in ["QGlobalAveragePooling2D", "GlobalMaxPooling2D",
                      "Concatenate"]:
    return 1.0 * cout_unroll / output_channel
  else:
    raise ValueError(f"Unspported layer type: {layer_type}")


def get_target_throughputs(layer, target_out_throughput):
  """Update throughput for a given layer."""

  # For layer that do not change the number of inference pixels,
  # throughput remains the same. For layers that decrease or increase the
  # number of inference pixels, the target throughput needs to update
  # accordingly.

  def multiply_elements_except_none(my_tuple):
    # Convert None values to np.nan and then use np.nanprod to calculate
    # the product
    return np.nanprod([x if x is not None else np.nan for x in my_tuple])

  if layer:
    input_size = multiply_elements_except_none(layer.input_shape[:-1])
    output_size = multiply_elements_except_none(layer.output_shape[:-1])
    target_in_throughput = target_out_throughput * input_size / output_size
  else:
    target_in_throughput = target_out_throughput

  # Per new design, target_pe_throughput equals to target_out_throughput.
  target_pe_throughput = target_out_throughput
  return target_in_throughput, target_pe_throughput


def calc_hw_params(graph, target_OutElementPerClk, target_out_throughput,
                   input_quantizer_bits,
                   compute_to_memory_max_ratio=4,
                   memory_to_unroll_max_ratio=4,
                   mode=CostMode.ACE):
  """Calculate HW params that minimizes total cost.

  Args:
    graph: DivideConquerGraph Object. Model graph.
    target_OutElementPerClk: Int. Target number of elements per clock
      cycle that the hardware needs to output.
    target_out_throughput: Float. Target number of inferences per clock
      cycle that the hardware needs to make.
    input_quantizer_bits: Int. Model's input quantizer bits.
    compute_to_memory_max_ratio: Int. Max allowed ratio between
      ComputeOutElement and OutElement
    memory_to_unroll_max_ratio: Int. Max allowed ratio between
      InElementPerClk and CinUnroll
    mode: CostMode. The mode to calculate per layer cost. Default is ACE.

  Returns:
    best_path: Dict. Stores the best hw param value at each layer and their
      irrespective cost.
    best_cost: Float. The best global cost of the entire model.
  """

  # Paths stores the best choices for every layer.
  # For the layer_idx, for each OutElementPerClk, we can calculate the best hw
  # param choice. We store all these best choices, each choice will
  # correspond to one OutElementPerClk key. Path therefore has the format:
  # {layer: {OutElementPerClk: (choice, cost, downstream_OutElementPerClk)}}
  paths = {}

  # We start the computation from the last node.
  layer_idx = graph.get_last_node()

  # Store the hw choices for the last node (a dummy node) for the sake
  # of completion.
  paths[layer_idx] = {
      target_OutElementPerClk: {
          "choice": Choice().__str__(),
          "cur_cost": 0,
          "acc_cost": 0,
          "OutElementPerClk": -1}}

  logging.debug("====== Extracting HW params combinations per layer =====")

  # The following code calculates cost backward, from last layer to the first.
  while  graph.get_prev_nodes(layer_idx):
    # Find precessor of the layer.
    # TODO(lishanok@): extend this code to multiple prev layers.
    cur_layer_idx = graph.get_prev_nodes(layer_idx)[0]
    cur_layer = graph.idx_to_layer(cur_layer_idx)
    logging.debug("processing layer_idx:%d name:%s type:%s ***",
                  cur_layer_idx, getattr(cur_layer, "name", None),
                  cur_layer.__class__.__name__)

    target_in_throughput, target_pe_throughput = get_target_throughputs(
        cur_layer, target_out_throughput)

    # Previous layer will generate a list of candidates for OutElementPerClk
    # values for the current layer.
    OutElementPerClk_list = list(paths[layer_idx].keys())
    logging.debug("OutElementPerClk_list:%s", OutElementPerClk_list)

    layer_quantizer_bitwidth = graph.get_layer_quantizer_bitwidth(cur_layer)
    layer_mac_count = graph.get_layer_mac_count(cur_layer)
    layer_shapes = graph.get_layer_shapes(cur_layer)

    # TODO(lishanok@): need to extend to multiple input layers, i.e., more
    # than 1 layer will reach graph's first node. We should only exit if all
    # input layers are processed.
    if graph.is_first_node(cur_layer_idx):
      # Computation reaches the 1st node of the graph. We can now find the best
      # path of all OutElementPerClk choices at the first layer.
      set_best_global_cost_in_paths(
          OutElementPerClk_list, paths, layer_idx, cur_layer_idx,
          layer_quantizer_bitwidth, layer_mac_count, layer_shapes, mode)
      break

    # Get layer-related information
    input_channel = qtools_util.get_layer_info(cur_layer, "input_channel")
    output_channel = qtools_util.get_layer_info(cur_layer, "output_channel")
    kernel_height = qtools_util.get_layer_info(cur_layer, "kernel_height")
    kernel_width = qtools_util.get_layer_info(cur_layer, "kernel_width")
    layer_type = qtools_util.get_layer_info(cur_layer, "layer_type")
    output_channel_divisors = qtools_util.find_divisors(output_channel)

    logging.debug("input_channel: %d, output_channel: %d, kernel_height: %d, "
                  "kernel_width: %d, weight_quantizer_bits: %d",
                  input_channel, output_channel, kernel_height, kernel_width,
                  layer_quantizer_bitwidth["weight_bits"])

    cur_best_choices = {}
    for OutElementPerClk in OutElementPerClk_list:
      logging.debug("...OutElementPerClk: %d", OutElementPerClk)

      # Pass through OutElementPerClk and cost for non-essential layers.
      if layer_type in ["QBatchNormalization", "QActivation", "Dropout",
                        "Reshape", "Activation", "ZeroPadding2D"]:
        logging.debug("...... Passing through layer_type: %s with 0 cost",
                      layer_type)

        # Update the best choices dict with only 1 key-value pair. By
        # considering current light-computation layer in the graph
        # as a pass-through node, we set layer cost=0, and set the predecessor
        # node's OutElementPerClk the same as current node's OutElementPerClk.
        update_cur_best_choices(
            cur_best_choices, OutElementPerClk=OutElementPerClk,
            prev_OutElementPerClk=OutElementPerClk, cur_layer_cost=0,
            accumulative_cost=paths[layer_idx][OutElementPerClk]["acc_cost"],
            choice=Choice())

        # Exit current iteration since there is no design param to explore
        # for these layer types.
        continue

      # For each of the possible OutElementPerClk values provided by the next
      # layer, we derive possible HW params choices of the current layer.
      for ComputeOutElementPerClk in get_valid_candidates(
          OutElementPerClk, compute_to_memory_max_ratio):
        logging.debug("......ComputeOutElementPerClk: %d",
                      ComputeOutElementPerClk)

        l = OutElementPerClk / ComputeOutElementPerClk
        cout_unroll = ComputeOutElementPerClk

        # cout_unroll needs to be a divisor of output_channels
        if cout_unroll not in output_channel_divisors:
          continue

        logging.debug(
            ".........OutElementPerClk / ComputeOutElementPerClk = %.2f,"
            "cout_unroll=%.2f", l, cout_unroll)
        # Find valid unroll values that meet pe throughput requirement.
        valid_unrolls = get_valid_unrolls(cur_layer, cout_unroll,
                                          target_pe_throughput)
        if not valid_unrolls:
          # Skip if no valid unroll values are found.
          logging.debug(".........No valid unroll values found!")
          continue

        for (cin_unroll, kh_unroll, kw_unroll) in valid_unrolls:
          # Check throughput requirement of each combination of unroll values.
          logging.debug(".........cin_unroll: %d, kh_unroll: %d, kw_unroll: %d",
                        cin_unroll, kh_unroll, kw_unroll)
          ComputInElementPerClk = get_ComputeInElementPerClk(
              layer_type, cin_unroll=cin_unroll, cout_unroll=cout_unroll,
              kh_unroll=kh_unroll, kw_unroll=kw_unroll)

          # InElementPerClk = k*ComputeInElementPerClk/(kh_unroll * kw_unroll)
          # TODO(lishanok@): Confirm if it works for Concatenate layer.
          InElementPerClk_base = get_InElementPerClk_base(
              ComputInElementPerClk=ComputInElementPerClk, kh_unroll=kh_unroll,
              kw_unroll=kw_unroll)
          for InElementPerClk in get_valid_candidates(
              InElementPerClk_base, memory_to_unroll_max_ratio):
            # With given cin_unroll, check throughput requirement of each
            # possible candidate of InElementPerClk.
            logging.debug("............InElementPerClk: %d", InElementPerClk)
            k = cin_unroll / InElementPerClk
            # prev_OutElementPerClk is the predecessor node's OutElementPerClk
            prev_OutElementPerClk = InElementPerClk

            if is_bufferThru_greater_than_targetThru(
                layer_type=layer_type, InElementPerClk=InElementPerClk,
                OutElementPerClk=OutElementPerClk, input_channel=input_channel,
                output_channel=output_channel, kernel_height=kernel_height,
                kernel_width=kernel_width,
                target_out_throughput=target_out_throughput,
                target_in_throughput=target_in_throughput):
              # If valid unroll values meet buffer throughput requirements,
              # comput cost.
              # cost = current layer's cost + total of downstream layers' cost.
              # Since we derive cost iteratively starting from the last layer,
              # paths already store the total cost of the downstream layers.
              cur_layer_cost = get_per_layer_cost(
                  layer_quantizer_bitwidth, layer_mac_count, layer_shapes,
                  cin_unroll, cout_unroll, kh_unroll, kw_unroll,
                  InElementPerClk, OutElementPerClk, mode)
              accumulative_cost = (
                  cur_layer_cost + paths[layer_idx][OutElementPerClk][
                      "acc_cost"])

              logging.debug("...............Buf throughput is good! "
                            "Accumulative_cost: %.2f", accumulative_cost)

              # Each choice is a hw param combination.
              choice = Choice(l, k, cin_unroll, cout_unroll, kh_unroll,
                              kw_unroll)

              update_cur_best_choices(cur_best_choices, OutElementPerClk,
                                      prev_OutElementPerClk, cur_layer_cost,
                                      accumulative_cost, choice)

    if not cur_best_choices:
      logging.error("Cannot find any valid HW choice for layer %s! Exit!",
                    cur_layer.name)
      return {}, None

    logging.debug("=======================")

    # Store the best choices of hw params for the current layer. Proceed to
    # the previous layer.
    paths[cur_layer_idx] = cur_best_choices
    layer_idx = cur_layer_idx
    # Predicessor node's OutBuf throughput is sucessor node's InBuf throughput.
    target_out_throughput = target_in_throughput

  return backtrack(graph, paths)


def estimate_model_cost(
    model: tf.keras.Model,
    input_quantizer_bits: int = 8,
    target_OutElementPerClk: int = 10,
    target_out_throughput: float = 1.0,
    compute_to_memory_max_ratio: int = 4,
    memory_to_unroll_max_ratio: int = 4,
    mode: CostMode = CostMode.ACE):
  """Main function to divide and conquer cost modeling.

  Args:
    model: QKeras model.
    input_quantizer_bits: Model's input quantizer bits.
    target_OutElementPerClk: Target number of elements per clock
      cycle that the hardware needs to output.
    target_out_throughput: Target number of inferences per clock
      cycle that the hardware needs to make.
    compute_to_memory_max_ratio: Max allowed ratio between
      ComputeOutElement and OutElement
    memory_to_unroll_max_ratio: Max allowed ratio between
      InElementPerClk and CinUnroll
    mode: The mode to calculate per layer cost.

  Returns:
    best_path: Dict. Stores the best hw param value at each layer and their
      irrespective cost.
    best_cost: Float. The best global cost of the entire model.
  """

  logging.info("Estimating model design params and cost...")
  # Generate graph
  graph = DivideConquerGraph(model)
  # Call the main function to generate optimal HW configs for all layers
  best_path, best_cost = calc_hw_params(
      graph=graph, target_OutElementPerClk=target_OutElementPerClk,
      target_out_throughput=target_out_throughput,
      input_quantizer_bits=input_quantizer_bits,
      compute_to_memory_max_ratio=(
          compute_to_memory_max_ratio),
      memory_to_unroll_max_ratio=(
          memory_to_unroll_max_ratio),
      mode=mode
  )

  logging.info("best_design_params: %s", best_path)

  return (best_path, best_cost)
