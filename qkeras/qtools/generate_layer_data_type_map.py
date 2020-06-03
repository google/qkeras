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
"""Generates MAC, input and output datatype for a qkeras model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import networkx as nx
from qkeras.qtools import qgraph
from qkeras.qtools import qtools_util
from qkeras.qtools import quantized_operators
from qkeras.qtools.quantized_operators import quantizer_factory as quantizer_factory_module
from qkeras.qtools.settings import cfg


class TagMissingError(ValueError):
  pass


LayerDataType = collections.namedtuple(
    "LayerDataType",
    [
        "input_quantizer_list",
        "multiplier",
        "accumulator",

        "weight_quantizer",
        "w_shapes",

        "bias_quantizer",
        "b_shapes",

        "output_quantizer",
        "output_shapes",

        "operation_count",
    ],
)

QKERAS_LAYERS = [
    "QDense",
    "QConv1D",
    "QConv2D",
    "QDepthwiseConv2D",
    "QSeparableConv2D",
    "QOctaveConv2D",
]

KERAS_LAYERS = [
    "Dense",
    "Conv1D",
    "Conv2D",
    "DepthwiseConv2D",
    "SeparableConv2D",
    "OctaveConv2D",
]


def update_output_quantizer_in_graph(graph, node_id, quantizer_factory,
                                     new_quantizer, for_reference):
  """update the edge with output quantizer type."""

  node = graph.nodes[node_id]
  qkeras_output_quantizer = node["out_quantizer"]

  # if existing graph doesn't have a valid output quantizer
  # update graph with the new quantizer
  if (for_reference or not qkeras_output_quantizer or
      not quantizer_factory.is_quantizer_supported(qkeras_output_quantizer)):
    qkeras_output_quantizer = new_quantizer
    qgraph.GraphUpdateEdge(graph, node_id, qkeras_output_quantizer)

  # if activation specified, convert activation quantizer to qtools quantizer
  # if activation not secified, convert the new quantizer to qtools quantizer
  output_quantizer = quantizer_factory.make_quantizer(qkeras_output_quantizer)

  # output_quantizer is used for updating dictionary in json
  return output_quantizer


def generate_layer_data_type_map(graph, source_quantizer_list, is_inference,
                                 keras_quantizer=None, keras_accumulator=None,
                                 for_reference=False, debug=False):
  """main funciton to generate datatype for each layer.

  For each type of layer, this function calculates the sizes and minimum
  number of bits required to represent the parameters and variables (e.g.,
  weights, bias, multiplier and accumulator - MAC, etc.) embedded in
  these layers.

  Args:
    graph: input graph that traverses the model
    source_quantizer_list: a list of quantizers for model inputs
    is_inference: whether model is pre-trained with weights available
    keras_quantizer: default quantizer used to quantize un-quantized layers
    keras_accumulator: default MAC quantizer to quantize un-quantized layers
    for_reference: whether to generate a map for a baseline model
    debug: whether to print debug messages

  Returns:
    a result containing the following fields:
    source_quantizer_list similar as input
    output_layers: names of the layers that are output layers
    input_layers: names of the layers that are input_layers,
    layer_data_type_map: data type map of each layer
  """

  quantizer_factory = quantizer_factory_module.QuantizerFactory()
  layer_data_type_map = collections.OrderedDict()

  # get the output layers

  output_layers = []
  input_layers = []
  predecessors = list(graph.predecessors(qgraph.SINK))
  successors = list(graph.successors(qgraph.SOURCE))

  for u in predecessors:
    if u == qgraph.SOURCE or u == qgraph.SINK:
      continue
    output_layers.append(graph.nodes[u]["layer"][0])

  for u in successors:
    if u == qgraph.SOURCE or u == qgraph.SINK:
      continue
    input_layers.append(graph.nodes[u]["layer"][0])

  for node_id in nx.topological_sort(graph):
    node = graph.nodes[node_id]
    node_type = node["type"][-1]
    layer = node["layer"][0]
    is_input_layer = layer in input_layers

    w_shapes = None
    b_shapes = None
    output_shapes = None

    if hasattr(layer, "output_shape"):
      output_shapes = layer.output_shape

    if hasattr(layer, "get_weights"):
      weights = layer.get_weights()
      if len(weights) != 0:
        w_shapes = layer.get_weights()[0].shape
        b_shapes = weights[0].shape[-1]

    if debug:
      print("########")
      if layer is not None:
        print(layer.name)
      else:
        print("None")

    # deal with keras layer or lack of input quantizer in qkeras layer
    input_qe_list = qtools_util.get_input_quantizers_advanced(
        graph, node_id, is_input_layer, quantizer_factory, cfg)

    if input_qe_list and node_id != qgraph.SINK:
      input_quantizer_list = []
      for node in input_qe_list:
        input_quantizer_list.append(node[0])

      # calculate number of operations (multiplication/accumulation)
      (_, edge_0) = input_qe_list[0]
      input_shape = edge_0["shape"]
      # for merge layers, all input_shape are identical
      operation_count = qtools_util.get_operation_count(
          layer, input_shape)

    # Merge Layers with multiple inputs
    if qtools_util.is_merge_layers(layer):
      # output_shapes = layer.output_shape
      merge_factory = quantized_operators.MergeFactory()
      merge_quantizer = merge_factory.make_quantizer(
          input_qe_list, layer.__class__.__name__)

      if hasattr(layer, "get_quantizers"):
        # QMerge layer from future
        qkeras_quantizer = layer.get_quantizers()[0]
        merge_quantizer.output = quantizer_factory.make_quantizer(
            qkeras_quantizer)
      else:
        # merge layer is a Keras layer
        if for_reference:
          merge_quantizer.output = quantizer_factory.make_default_quantizer(
              mode=cfg.default_interm_quantizer)

          if keras_accumulator:
            # gate_factor and gate_bits remain the same as previously
            # calculated; only change output quantizer as the keras_accumulator
            merge_quantizer.output = quantizer_factory.make_default_quantizer(
                mode=keras_accumulator)

      output_quantizer = update_output_quantizer_in_graph(
          graph, node_id, quantizer_factory, merge_quantizer.output,
          for_reference)

      layer_data_type_map[layer] = LayerDataType(
          input_quantizer_list,
          merge_quantizer,
          None,

          None,
          None,

          None,
          None,

          output_quantizer,
          output_shapes,

          operation_count
      )

    # pooling/reshape/flatten/UpSampling1D/2D/3D
    elif (qtools_util.is_shape_alternation_layers(layer) or
          "UpSampling" in layer.__class__.__name__):
      input_quantizer = input_quantizer_list[0]

      # output quantizer
      output_quantizer = update_output_quantizer_in_graph(
          graph, node_id, quantizer_factory, input_quantizer, for_reference)

      layer_data_type_map[layer] = LayerDataType(
          input_quantizer_list,
          None,
          None,

          None,
          None,

          None,
          None,

          output_quantizer,
          output_shapes,

          operation_count
      )

    # if Quantized Activation layer
    elif node_type in ["QActivation", "Activation"]:

      if for_reference or not hasattr(layer, "quantizer"):
        # Keras activation layer -> use default_interm_quantizer
        layer_quantizer = quantizer_factory.make_default_quantizer(
            mode=cfg.default_interm_quantizer)

        if keras_quantizer:
          layer_quantizer = quantizer_factory.make_default_quantizer(
              mode=keras_quantizer)
      else:
        layer_quantizer = layer.quantizer

        if not quantizer_factory.is_quantizer_supported(layer_quantizer):
          raise TagMissingError(
              "Unsupported activation quantizer {} on this layer: {}".format(
                  layer_quantizer, layer))

        if not layer_quantizer:
          layer_quantizer = quantizer_factory.make_default_quantizer(
              mode=cfg.default_interm_quantizer)

      output_quantizer = update_output_quantizer_in_graph(
          graph, node_id, quantizer_factory, layer_quantizer, for_reference)

      layer_data_type_map[layer] = LayerDataType(
          input_quantizer_list,
          None,
          None,
          None,
          w_shapes,
          None,
          b_shapes,
          output_quantizer,
          output_shapes,
          operation_count
      )

    elif node_type in ["QBatchNormalization", "BatchNormalization"]:

      (input_quantizer, _) = input_qe_list[0]

      # qkeras layers might be mixed with keras layers
      if for_reference or not hasattr(layer, "get_quantizers"):
        # Keras BatchNorm layer mixed with quantized model
        # -> no reference mode
        gamma_quantizer = quantizer_factory.make_default_quantizer(
            mode=cfg.default_interm_quantizer)
        beta_quantizer = quantizer_factory.make_default_quantizer(
            mode=cfg.default_interm_quantizer)
        mean_quantizer = quantizer_factory.make_default_quantizer(
            mode=cfg.default_interm_quantizer)
        variance_quantizer = quantizer_factory.make_default_quantizer(
            mode=cfg.default_interm_quantizer)

        if keras_quantizer:
          gamma_quantizer = quantizer_factory.make_default_quantizer(
              mode=keras_quantizer)
          beta_quantizer = quantizer_factory.make_default_quantizer(
              mode=keras_quantizer)
          mean_quantizer = quantizer_factory.make_default_quantizer(
              mode=keras_quantizer)
          variance_quantizer = quantizer_factory.make_default_quantizer(
              mode=keras_quantizer)
      else:
        (qkeras_gamma_quantizer, qkeras_beta_quantizer, qkeras_mean_quantizer,
         qkeras_variance_quantizer) = layer.get_quantizers()

        if not qkeras_beta_quantizer:
          beta_quantizer = quantizer_factory.clone_quantizer(input_quantizer)
        else:
          beta_quantizer = quantizer_factory.make_quantizer(
              qkeras_beta_quantizer)

        if not qkeras_mean_quantizer:
          mean_quantizer = quantizer_factory.clone_quantizer(input_quantizer)
        else:
          mean_quantizer = quantizer_factory.make_quantizer(
              qkeras_mean_quantizer)

        if not qkeras_variance_quantizer:
          variance_quantizer = quantizer_factory.make_default_quantizer(
              mode=cfg.default_interm_quantizer)
        else:
          # if gamma is float, convert to input_quantizer
          variance_quantizer = quantizer_factory.make_quantizer(
              qkeras_variance_quantizer)

        if not qkeras_gamma_quantizer:
          gamma_quantizer = quantizer_factory.make_default_quantizer(
              mode=cfg.default_interm_quantizer)
        else:
          gamma_quantizer = quantizer_factory.make_quantizer(
              qkeras_gamma_quantizer)

      # during inference, gamma, beta and variance are constants
      # if they are po2 quantizers, we need to modify their bits
      # with actual values and also update graph with the
      # corresponding output_quantizer on the edge
      if is_inference:
        weights = qtools_util.get_weights(layer)
        # if no scale(gamma), num_weights --
        # if no center(beta_quantizer) num_weights --
        num_weights = 4
        if not layer.scale:
          num_weights -= 1
        if not layer.center:
          num_weights -= 1

        if layer.scale and gamma_quantizer.is_po2:
          gamma_quantizer.update_inference_values(weights[0])
        if variance_quantizer.is_po2:
          variance_quantizer.update_inference_values(
              weights[num_weights-1])

      qbn = quantized_operators.QBNFactory()
      qbn.make_quantizer(
          input_quantizer, gamma_quantizer, beta_quantizer,
          mean_quantizer, variance_quantizer, layer.scale, layer.center
      )

      def set_output(op, output):
        if op:
          op.output = output

      if for_reference or not hasattr(layer, "get_quantizers"):
        set_output(
            qbn.internal_divide_quantizer,
            quantizer_factory.make_default_quantizer(
                mode=cfg.default_interm_quantizer))

        set_output(
            qbn.internal_multiplier,
            quantizer_factory.make_default_quantizer(
                mode=cfg.default_interm_quantizer))

        set_output(
            qbn.internal_accumulator,
            quantizer_factory.make_default_quantizer(
                mode=cfg.default_interm_quantizer))

        set_output(
            qbn.internal_output,
            quantizer_factory.make_default_quantizer(
                mode=cfg.default_interm_quantizer))

        if keras_accumulator:
          set_output(
              qbn.internal_divide_quantizer,
              quantizer_factory.make_default_quantizer(mode=keras_accumulator))

          set_output(
              qbn.internal_multiplier,
              quantizer_factory.make_default_quantizer(mode=keras_accumulator))

          set_output(
              qbn.internal_accumulator,
              quantizer_factory.make_default_quantizer(mode=keras_accumulator))

          set_output(
              qbn.internal_output.output,
              quantizer_factory.make_default_quantizer(mode=keras_accumulator))

      layer_quantizer = qbn.internal_output.output

      output_quantizer = update_output_quantizer_in_graph(
          graph, node_id, quantizer_factory, layer_quantizer, for_reference)

      gamma_range = None
      if hasattr(layer, "gamma_range"):
        gamma_range = layer.gamma_range

      beta_range = None
      if hasattr(layer, "beta_range"):
        beta_range = layer.beta_range

      if not layer.center:
        qbn.beta_quantizer = None

      if not layer.scale:
        qbn.gamma_quantizer = None

      layer_data_type_map[layer] = {
          "input_quantizer_list": input_quantizer_list,
          "gamma_quantizer": gamma_quantizer,
          "beta_quantizer": beta_quantizer,
          "mean_quantizer": mean_quantizer,
          "variance_quantizer": variance_quantizer,
          "gamma_range": gamma_range,
          "beta_range": beta_range,
          "internal_divide_quantizer": qbn.internal_divide_quantizer,
          "internal_multiplier": qbn.internal_multiplier,
          "internal_accumulator": qbn.internal_accumulator,
          "output_quantizer": output_quantizer,
          "output_shapes": input_shape,
          "operation_count": operation_count
      }

    # if qdense, qconv, qpool, qoctave
    elif node_type in QKERAS_LAYERS or node_type in KERAS_LAYERS:

      (input_quantizer, _) = input_qe_list[0]

      if for_reference or not hasattr(layer, "get_quantizers"):
        # for_reference: force all quantizers to keras_quantizer
        weight_quantizer = quantizer_factory.make_default_quantizer(
            mode=cfg.default_interm_quantizer)
        bias_quantizer = quantizer_factory.make_default_quantizer(
            mode=cfg.default_interm_quantizer)

        if keras_quantizer:
          weight_quantizer = quantizer_factory.make_default_quantizer(
              mode=keras_quantizer)
          bias_quantizer = quantizer_factory.make_default_quantizer(
              mode=keras_quantizer)
      else:
        # qkeras layer
        qkeras_weight_quantizer = layer.get_quantizers()[0]
        qkeras_bias_quantizer = layer.get_quantizers()[1]

        if not quantizer_factory.is_quantizer_supported(
            qkeras_weight_quantizer):
          raise TagMissingError(
              "Unsupported weight quantizer {} on this layer: {}".format(
                  qkeras_weight_quantizer, layer))

        if not quantizer_factory.is_quantizer_supported(
            qkeras_bias_quantizer):
          raise TagMissingError(
              "Unsupported bias quantizer {} on this layer: {}".format(
                  qkeras_bias_quantizer, layer))

        weight_quantizer = quantizer_factory.make_quantizer(
            qkeras_weight_quantizer)
        bias_quantizer = quantizer_factory.make_quantizer(
            qkeras_bias_quantizer)

      # TODO(lishanok): during inference, if weight and bias is po2,
      #  need to update corresponding quantizer type with min and max
      #  of the constant values
      if is_inference:
        weights = qtools_util.get_weights(layer)
        if weight_quantizer.is_po2:
          weight_quantizer.update_inference_values(weights[0])

        if bias_quantizer.is_po2:
          bias_quantizer.update_inference_values(weights[1])

      multiplier_factory = quantized_operators.MultiplierFactory()
      multiplier = multiplier_factory.make_multiplier(
          weight_quantizer, input_quantizer)

      weights = layer.get_weights()
      kernel = weights[0]

      # if layer.use_bias:
      #  bias = weights[1]

      accumulator_factory = quantized_operators.AccumulatorFactory()
      accumulator = accumulator_factory.make_accumulator(
          kernel.shape, multiplier)

      if not layer.use_bias:
        bias_quantizer = None

      if debug:
        print(layer.name or "None")
        print("weight_quantizer:", weight_quantizer.bits)
        print("input_quantizer:", input_quantizer.bits)
        print("multiplier_quantizer:", multiplier.output.bits)
        print("multiplier_gate_bits:", multiplier.gate_bits)
        print("accumulator:", accumulator.output.bits)

      if for_reference or not hasattr(layer, "get_quantizers"):
        accumulator.output = quantizer_factory.make_default_quantizer(
            mode=cfg.default_interm_quantizer)
        multiplier.output = quantizer_factory.make_default_quantizer(
            mode=cfg.default_interm_quantizer)

        if keras_accumulator:
          accumulator.output = quantizer_factory.make_default_quantizer(
              mode=keras_accumulator)
          multiplier.output = quantizer_factory.make_default_quantizer(
              mode=keras_accumulator)

      layer_quantizer = accumulator.output
      output_quantizer = update_output_quantizer_in_graph(
          graph, node_id, quantizer_factory, layer_quantizer, for_reference)

      layer_data_type_map[layer] = LayerDataType(
          input_quantizer_list,
          multiplier,
          accumulator,
          weight_quantizer,
          w_shapes,
          bias_quantizer,
          b_shapes,
          output_quantizer,
          output_shapes,
          operation_count
      )

    elif node_type:
      # any other unsupported layer types -> pass the input quantizer
      # type to output in qraph
      (input_quantizer, _) = input_qe_list[0]

      if for_reference and keras_quantizer:
        input_quantizer = quantizer_factory.make_default_quantizer(
            mode=keras_quantizer)

      update_output_quantizer_in_graph(
          graph, node_id, quantizer_factory, input_quantizer, for_reference)

      layer_data_type_map[layer] = LayerDataType(
        input_quantizer_list,
        None,
        None,

        None,
        None,

        None,
        None,

        output_quantizer,
        output_shapes,

        operation_count
      )

  result = {"source_quantizer_list": source_quantizer_list,
            "output_layers": output_layers,
            "input_layers": input_layers,
            "layer_data_type_map": layer_data_type_map}

  return result

