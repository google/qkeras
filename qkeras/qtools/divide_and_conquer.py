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
from typing import List, Any, Union

import numpy as np
import tensorflow as tf

from qkeras import quantizers
from qkeras.qtools import qgraph
from qkeras.qtools import qtools_util


class CostMode(enum.Enum):
    NAIVE = 1
    ML_PE_AREA = 2
    ML_PE_BW_AREA = 3


# pylint: disable=invalid-name
class DivideConquerGraph:
  """This class creates model graph structure and methods to access layers."""

  def __init__(self, model: tf.keras.Model,
               source_quantizers: quantizers.BaseQuantizer = None):
    self._model = model
    self._source_quantizer_list = source_quantizers or [
        quantizers.quantized_bits(8, 0, 1)]

    (self._graph, self._source_quantizer_list) = qgraph.CreateGraph(
        model, source_quantizers, "quantized_bits(8, 0, 1)")

    # Propagate output quantizer info into the graph edges.
    qgraph.GraphPropagateActivationsToEdges(self._graph)

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
                      target_throughput: int):
  """Get valid unroll values where resulting throughput>=Target throughput."""

  input_channel = qtools_util.get_layer_info(layer, "input_channel")
  output_channel = qtools_util.get_layer_info(layer, "output_channel")
  kernel_height = qtools_util.get_layer_info(layer, "kernel_height")
  kernel_width = qtools_util.get_layer_info(layer, "kernel_width")

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
        # Caculate computation throughput.
        pe_throughput = cin_unroll * cout_unroll * kh_unroll * kw_unroll / (
            input_channel * output_channel * kernel_height * kernel_width)

        if pe_throughput >= target_throughput:
          # Save the valid combination of unroll factors to valid_unrolls.
          valid_unrolls.append((cin_unroll, kh_unroll, kw_unroll))

  return valid_unrolls


def get_per_layer_cost(mac_bitwidth, cin_unroll, cout_unroll, kh_unroll,
                       kw_unroll, InElementPerClk, OutElementPerClk,
                       mode):
  # Area for a single layer, includes both PE and memory Bandwidth
  # TODO(lishanok@): needs a better cost modeling function. For now we simplify
  # it to the number of multipliers + interface bitwidth.
  assert mode == CostMode.NAIVE, "Only CostMode.NAIVE is supported for now."

  pe_area = mac_bitwidth * cin_unroll * cout_unroll * kh_unroll * kw_unroll
  memory_bw = InElementPerClk * OutElementPerClk
  return pe_area + memory_bw


def get_valid_candidates(input_value, output_to_input_ratio_max):
  candidate_list = qtools_util.find_divisors(input_value)
  # Add the other scenario where ComputeElementPerClk is multiple
  # of ElementPerClk.
  if output_to_input_ratio_max >= 2:
    candidate_list += [input_value * x for x in list(
        range(2, output_to_input_ratio_max+1))]

  return candidate_list


def is_bufferThru_greater_than_targetThru(
    InElementPerClk: int, OutElementPerClk: int, input_channel: int,
    output_channel: int, kernel_height: int, kernel_width: int,
    is_upsampled: bool, target_throughput: float):
  """Verify whether the resulting buffer throughput > target throughput."""

  # Calculate throughput of input buffer.
  InBuf_throughput = InElementPerClk / input_channel
  # Calculate throughput of output buffer.
  if is_upsampled:
    OutBuf_throughput = OutElementPerClk / (
        output_channel * kernel_height *kernel_width)
  else:
    OutBuf_throughput = OutElementPerClk / output_channel

  logging.debug(
      "...............InBuf_throughput: %.2f OutBuf_throughput: %.2f",
      InBuf_throughput, OutBuf_throughput)

  # Valid unroll values must meet buffer throughput requirements.
  return (InBuf_throughput >= target_throughput and
          OutBuf_throughput >= target_throughput)


def set_best_global_cost_in_paths(
    OutElementPerClk_list, paths, layer_idx, cur_layer_idx,
    input_quantizer_bits, mode):
  """Find the best global cost of the entire model and update the paths dict.

  Args:
    OutElementPerClk_list: list of OutElementPerClk for the current layer.
    paths: Dict that contains the choices that each layer has.
    layer_idx: The index value of the current layer's predecessor.
    cur_layer_idx: current layer's index value.
    input_quantizer_bits: Input quantizer bits to the model.
    mode: mode to calculate cost per layer.

  Returns:
    None.
  """

  def calculate_cost(OutElementPerClk):
    cur_layer_cost = get_per_layer_cost(
        input_quantizer_bits, 0, 0, 0, 0, 0, OutElementPerClk, mode)
    accumulative_cost = cur_layer_cost + paths[layer_idx][OutElementPerClk][2]
    return (cur_layer_cost, accumulative_cost, OutElementPerClk)

  cost_and_values = list(map(calculate_cost, OutElementPerClk_list))

  layer_cost, min_accumulative_cost, best_OutElementPerClk = (
      min(cost_and_values, key=lambda x: x[1]))

  # For the initial node, we find the best path which contains a sentinel
  # choice, cost with that path, and the chosen OutElementPerClk
  # that will point to the corresponding choice of the following layer.
  paths[cur_layer_idx] = {
      best_OutElementPerClk: (Choice().__str__(), layer_cost,
                              min_accumulative_cost, best_OutElementPerClk)}


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
  best_path[layer_idx] = best_entry
  best_OutElementPerClk = best_entry[3]
  best_accumlative_cost = best_entry[2]

  layer_idx = graph.get_next_nodes(layer_idx)[0]
  # Given the best choice of 1st layer, find the best choice for all following
  # layers by backtracking.
  while not graph.is_last_node(layer_idx):
    # Find current layer's best choice from the ptr (ie. best_OutElementPerClk)
    # stored in the best choice of the previous layer.
    best_entry = paths[layer_idx][best_OutElementPerClk]
    best_path[layer_idx] = best_entry
    # Update the ptr to the next layer.
    best_OutElementPerClk = best_entry[3]

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
  existing_accumulative_cost = entry[2] if entry else np.inf
  logging.debug("...............cost of cur_best_choices [%d]: %.2f",
                prev_OutElementPerClk, existing_accumulative_cost)
  if accumulative_cost < existing_accumulative_cost:
    # Stores the best choice and its cost for the given
    # prev_OutElementPerClk. We also store the ptr to next layer's
    # OutElementPerClk for future backtracking purpose.
    cur_best_choices[prev_OutElementPerClk] = (
        choice.__str__(), cur_layer_cost, accumulative_cost, OutElementPerClk)
    logging.debug(
        "...............Find better cost! Update cur_best_choices[%d]: %s",
        prev_OutElementPerClk, cur_best_choices[prev_OutElementPerClk])


def calc_hw_params(graph, target_OutElementPerClk, target_throughput,
                   input_quantizer_bits,
                   compute_to_memory_max_ratio=4,
                   memory_to_unroll_max_ratio=4,
                   mode=CostMode.NAIVE):
  """Calculate HW params that minimizes total cost.

  Args:
    graph: DivideConquerGraph Object. Model graph.
    target_OutElementPerClk: Int. Target number of elements per clock
      cycle that the hardware needs to output.
    target_throughput: Float. Target number of inferences per clock
      cycle that the hardware needs to make.
    input_quantizer_bits: Int. Model's input quantizer bits.
    compute_to_memory_max_ratio: Int. Max allowed ratio between
      ComputOutElement and OutElement
    memory_to_unroll_max_ratio: Int. Max allowed ratio between
      InElementPerClk and CinUnroll
    mode: CostMode. The mode to calculate per layer cost. Default is NAIVE.

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

  # Store the hw choices for the last node (a fake node) for the sake
  # of completion.
  paths[layer_idx] = {target_OutElementPerClk: (
      Choice().__str__(), 0, 0, -1)}

  logging.debug("====== Extracting HW params combinations per layer =====")

  # The following code calculates cost backward, from last layer to the first.
  while  graph.get_prev_nodes(layer_idx):
    # Find precessor of the layer.
    # TODO(lishanok@): extend this code to multiple prev layers.
    cur_layer_idx = graph.get_prev_nodes(layer_idx)[0]
    cur_layer = graph.idx_to_layer(cur_layer_idx)
    logging.debug("processing layer_idx:%d layer:%s",
                  cur_layer_idx, getattr(cur_layer, "name", None))

    # Previous layer will generate a list of candidates for OutElementPerClk
    # values for the current layer.
    OutElementPerClk_list = list(paths[layer_idx].keys())
    logging.debug("OutElementPerClk_list:%s", OutElementPerClk_list)

    # TODO(lishanok@): need to extend to multiple input layers, i.e., more
    # than 1 layer will reach graph's first node. We should only exit if all
    # input layers are processed.
    if graph.is_first_node(cur_layer_idx):
      # Computation reaches the 1st node of the graph. We can now find the best
      # path of all OutElementPerClk choices at the first layer.
      set_best_global_cost_in_paths(OutElementPerClk_list, paths, layer_idx,
                                    cur_layer_idx, input_quantizer_bits, mode)
      break

    # Get layer-related information
    input_channel = qtools_util.get_layer_info(cur_layer, "input_channel")
    output_channel = qtools_util.get_layer_info(cur_layer, "output_channel")
    kernel_height = qtools_util.get_layer_info(cur_layer, "kernel_height")
    kernel_width = qtools_util.get_layer_info(cur_layer, "kernel_width")
    quantizer_bits = qtools_util.get_layer_info(cur_layer, "quantizer_bits")

    logging.debug("input_channel: %d, output_channel: %d, kernel_height: %d, "
                  "kernel_width: %d, quantizer_bits: %d", input_channel,
                  output_channel, kernel_width, kernel_width, quantizer_bits)

    cur_best_choices = {}
    for OutElementPerClk in OutElementPerClk_list:
      logging.debug("...OutElementPerClk: %d", OutElementPerClk)

      # For each of the possible OutElementPerClk values provided by the next
      # layer, we derive possible HW params choices of the current layer.
      for ComputeOutElementPerClk in get_valid_candidates(
          OutElementPerClk, compute_to_memory_max_ratio):
        logging.debug("......ComputeOutElementPerClk: %d",
                      ComputeOutElementPerClk)
        l = OutElementPerClk / ComputeOutElementPerClk
        cout_unroll = ComputeOutElementPerClk

        # Find valid unroll values that meet pe throughput requirement.
        valid_unrolls = get_valid_unrolls(cur_layer, cout_unroll,
                                          target_throughput)
        if len(valid_unrolls) == 0:
          # Skip if no valid unroll values are found.
          continue

        for (cin_unroll, kh_unroll, kw_unroll) in valid_unrolls:
          # Check throughput requirement of each combination of unroll values.
          logging.debug(".........cin_unroll: %d, kh_unroll: %d, kw_unroll: %d",
                        cin_unroll, kh_unroll, kw_unroll)

          for InElementPerClk in get_valid_candidates(
              cin_unroll, memory_to_unroll_max_ratio):
            # With given cin_unroll, check throughput requirement of each
            # possible candidate of InElementPerClk.

            # InElementPerClk*k=ComputeInElementPerClk/(kh_unroll * kw_unroll)
            # ==> InElementPerClk=cin_unroll/k
            logging.debug("............InElementPerClk: %d", InElementPerClk)
            k = cin_unroll / InElementPerClk
            prev_OutElementPerClk = InElementPerClk

            is_upsampled = qtools_util.is_upsampled(cur_layer)
            if is_bufferThru_greater_than_targetThru(
                InElementPerClk, OutElementPerClk, input_channel,
                output_channel, kernel_height, kernel_width, is_upsampled,
                target_throughput):
              # If valid unroll values meet buffer throughput requirements,
              # comput cost.
              # cost = current layer's cost + total of downstream layers' cost.
              # Since we derive cost iteratively starting from the last layer,
              # paths already store the total cost of the downstream layers.
              cur_layer_cost = get_per_layer_cost(
                  quantizer_bits, cin_unroll, cout_unroll, kh_unroll,
                  kw_unroll, InElementPerClk, OutElementPerClk, mode)
              accumulative_cost = (
                  cur_layer_cost + paths[layer_idx][OutElementPerClk][1])

              logging.debug("...............Buf throughput is good! "
                            "Accumulative_cost: %.2f", accumulative_cost)

              # Each choice is a hw param combination.
              choice = Choice(l, k, cin_unroll, cout_unroll, kh_unroll,
                              kw_unroll)

              update_cur_best_choices(cur_best_choices, OutElementPerClk,
                                      prev_OutElementPerClk, cur_layer_cost,
                                      accumulative_cost, choice)

    logging.debug("=======================")

    # Store the best choices of hw params for the current layer. Proceed to
    # the previous layer.
    paths[cur_layer_idx] = cur_best_choices
    layer_idx = cur_layer_idx

  return backtrack(graph, paths)


def estimate_model_cost(
    model: tf.keras.Model,
    input_quantizer_bits: int = 8,
    target_OutElementPerClk: int = 10,
    target_throughput: float = 1.0,
    compute_to_memory_max_ratio: int = 4,
    memory_to_unroll_max_ratio: int = 4,
    mode: CostMode = CostMode.NAIVE):
  """Main function to divide and conquer cost modeling.

  Args:
    model: QKeras model.
    input_quantizer_bits: Model's input quantizer bits.
    target_OutElementPerClk: Target number of elements per clock
      cycle that the hardware needs to output.
    target_throughput: Target number of inferences per clock
      cycle that the hardware needs to make.
    compute_to_memory_max_ratio: Max allowed ratio between
      ComputOutElement and OutElement
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
      target_throughput=target_throughput,
      input_quantizer_bits=input_quantizer_bits,
      compute_to_memory_max_ratio=(
          compute_to_memory_max_ratio),
      memory_to_unroll_max_ratio=(
          memory_to_unroll_max_ratio),
      mode=mode
  )

  logging.info("best_design_params: %s", best_path)
  logging.info("best_cost: %.2f", best_cost)

  return (best_path, best_cost)
