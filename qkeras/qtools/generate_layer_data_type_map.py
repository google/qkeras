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
import collections
import copy
import numpy as np
import sys

import networkx as nx
from qkeras.qtools import qgraph
from qkeras.qtools import qtools_util
from qkeras.qtools import quantized_operators
from qkeras.qtools.quantized_operators import quantizer_factory as quantizer_factory_module
from qkeras.qtools.settings import cfg
from qkeras.qtools.quantized_operators import adder_factory
from qkeras.qtools.quantized_operators.fused_bn_factory import FusedBNFactory

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
]

KERAS_LAYERS = [
    "Dense",
    "Conv1D",
    "Conv2D",
    "DepthwiseConv2D",
]


def get_bn_quantizers(layer, quantizer_factory, cfg, keras_quantizer,
                      input_quantizer, is_inference, for_reference,
                      model_weights_already_quantized):
  """Extract quantizers from a given batchnorm layer."""

  # QKeras layers might be mixed with keras layers.
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
    inverse_quantizer = quantizer_factory.make_default_quantizer(
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
      inverse_quantizer = quantizer_factory.make_default_quantizer(
          mode=keras_quantizer)
  else:
    (qkeras_gamma_quantizer, qkeras_beta_quantizer,
     qkeras_mean_quantizer, qkeras_variance_quantizer,
     qkeras_inverse_quantizer) = layer.get_quantizers()

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
      # If variance is float, convert to input_quantizer.
      variance_quantizer = quantizer_factory.make_quantizer(
          qkeras_variance_quantizer)

    if not qkeras_gamma_quantizer:
      gamma_quantizer = quantizer_factory.make_default_quantizer(
          mode=cfg.default_interm_quantizer)
    else:
      gamma_quantizer = quantizer_factory.make_quantizer(
          qkeras_gamma_quantizer)

    if not qkeras_inverse_quantizer:
      inverse_quantizer = quantizer_factory.make_default_quantizer(
          mode=cfg.default_interm_quantizer)
    else:
      inverse_quantizer = quantizer_factory.make_quantizer(
          qkeras_inverse_quantizer)

  # During inference, gamma, beta and variance are constants
  # if they are po2 quantizers, we need to modify their bits
  # with actual values and also update graph with the
  # corresponding output_quantizer on the edge.
  if is_inference:
    weights = qtools_util.get_weights(
        layer, model_weights_already_quantized)
    # If no scale(gamma), num_weights --
    # If no center(beta_quantizer) num_weights --
    num_weights = 4
    if not layer.scale:
      num_weights -= 1
    if not layer.center:
      num_weights -= 1

    if (layer.scale and gamma_quantizer is not None and gamma_quantizer.is_po2):
      gamma_quantizer.update_inference_values(weights[0])
    if (variance_quantizer is not None and variance_quantizer.is_po2):
      variance_quantizer.update_inference_values(
          weights[num_weights-1])

  return (gamma_quantizer, beta_quantizer, mean_quantizer, variance_quantizer,
          inverse_quantizer)


def update_output_quantizer_in_graph(graph, node_id, quantizer_factory,
                                     new_quantizer, for_reference):
  """update the edge with output quantizer type."""

  node = graph.nodes[node_id]
  qkeras_output_quantizer = node["out_quantizer"]

  # If existing graph doesn't have a valid output quantizer
  # update graph with the new quantizer
  if (for_reference or not qkeras_output_quantizer or
      not quantizer_factory.is_quantizer_supported(qkeras_output_quantizer)):
    qkeras_output_quantizer = new_quantizer
    qgraph.GraphUpdateEdge(graph, node_id, qkeras_output_quantizer)

  # If activation specified, convert activation quantizer to qtools quantizer
  # If activation not secified, convert the new quantizer to qtools quantizer
  output_quantizer = quantizer_factory.make_quantizer(qkeras_output_quantizer)

  # Output_quantizer is used for updating dictionary in json
  return output_quantizer


def generate_layer_data_type_map(
    graph, source_quantizer_list, is_inference,
    keras_quantizer=None, keras_accumulator=None,
    for_reference=False, debug=False,
    model_weights_already_quantized=True,
    hw_weight_dict=None):
  """main funciton to generate datatype for each layer.

  For each type of layer, this function calculates the sizes and minimum
  number of bits required to represent the parameters and variables (e.g.,
  weights, bias, multiplier and accumulator - MAC, etc.) embedded in
  these layers.

  Args:
    graph: input graph that traverses the model
    source_quantizer_list: a list of quantizers for model inputs
    is_inference: whether model is pre-trained with weights available
    keras_quantizer: default quantizer used to quantize weights and bias
    keras_accumulator: default MAC quantizer to quantize multiplier,
      accumulator and output
    for_reference: whether to generate a map for a baseline model
    debug: whether to print debug messages
    model_weights_already_quantized: bool. If model weights are already
      quantized, no need to apply quantizer to weights here in this function.
    hw_weight_dict: weight dictonary for hardware inference. For example, fused
      bn op inference in hardware will need additional fused weights, which
      can be extracted from this dictionary. This dictionary is the output from
      utils.py/model_save_quantized_weights function.

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
    qkeras_weight_quantizer = None

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

    # Deals with keras layer or lack of input quantizer in qkeras layer.
    input_qe_list = qtools_util.get_input_quantizers_advanced(
        graph, node_id, is_input_layer, quantizer_factory, cfg)

    if input_qe_list and node_id != qgraph.SINK:
      input_quantizer_list = []
      for node in input_qe_list:
        input_quantizer_list.append(node[0])

      # Calculates number of operations (multiplication/accumulation).
      # Previously Merge layers's inputs all have the same shape, however, in
      # MobilenetV3 we found that there is shape broadcast in the keras
      # Multiply layer. Therefore we use the shape with max size as the
      # input shape
      if len(input_qe_list) > 0:
        maxsize = -1
        max_id = 0
        for (idx, item) in enumerate(input_qe_list):
          shape = item[1]["shape"]
          size = np.prod(shape[1:])
          if size > maxsize:
            maxsize = size
            max_id = idx
        input_shape = input_qe_list[max_id][1]["shape"]
      else:
        (_, edge_0) = input_qe_list[0]
        input_shape = edge_0["shape"]

      operation_count = qtools_util.get_operation_count(
          layer, input_shape)

    # Merges layers with multiple inputs.
    if qtools_util.is_merge_layers(layer):

      # merge_factory.make_quantizer automatically calculates the merge output
      # quantizer bitwidth according to input quantizer type.
      merge_factory = quantized_operators.MergeFactory()
      merge_quantizer = merge_factory.make_quantizer(
          input_qe_list, layer.__class__.__name__)

      if for_reference:
        # The for_reference option overwrites the auto-calculated merge output
        # quantizer
        if keras_accumulator:
          # gate_factor and gate_bits remain the same as previously
          # calculated; only change output quantizer as the keras_accumulator
          merge_quantizer.output = quantizer_factory.make_default_quantizer(
              mode=keras_accumulator)
        else:
          merge_quantizer.output = quantizer_factory.make_default_quantizer(
              mode=cfg.default_interm_quantizer)

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

    # MaxPooling/reshape/flatten/UpSampling1D/2D/3D
    elif (qtools_util.is_shape_alternation_layers(layer) or
          "UpSampling" in layer.__class__.__name__):
      input_quantizer = input_quantizer_list[0]

      # Output quantizer
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

    # AveragePooling and GlobalAveragePooling
    elif layer.__class__.__name__ in [
        "AveragePooling2D", "AvgPool2D", "GlobalAvgPool2D",
        "GlobalAveragePooling2D", "QAveragePooling2D",
        "QGlobalAveragePooling2D"]:
      (input_quantizer, _) = input_qe_list[0]
      qtools_average_quantizer = None
      # This is a hack. We don't want to implement a new accumulator class
      # just for averagpooling. So we re-use accumulator type in conv/dense
      # layers which need multiplier and kernel as input parameters.
      # In order to do so, we fake a multiplier which treat the pool_size as
      # the kernel. since kernel needs 4 dimension, k_h, k_w, C_in, C_out,
      # we set the last two dimension as [1, 1]
      if layer.__class__.__name__ in ["AveragePooling2D", "AvgPool2D",
                                      "QAveragePooling2D"]:
        pool_size = tuple(list(layer.pool_size) + [1, 1])
      else:
        pool_size = tuple(list(input_shape)[1:-1] + [1, 1])

      # Automatically calculates the accumulator bitwidth according to input
      # quantizer type for both quantized pooling and regular pooling layers
      multiplier_factory = quantized_operators.MultiplierFactory()
      fake_multiplier = multiplier_factory.make_multiplier(
          input_quantizer, input_quantizer)
      fake_multiplier.output = input_quantizer
      accumulator_factory = quantized_operators.AccumulatorFactory()
      accumulator = accumulator_factory.make_accumulator(
          pool_size, fake_multiplier, use_bias=False)

      # For quantized pooling layers, we also need to consider the division
      # precision, which is controlled by the average quantizer
      if layer.__class__.__name__ in ["QAveragePooling2D",
                                      "QGlobalAveragePooling2D"]:
        # For the quantized layer, there is an average_quantizer used for
        # the inverse of division operation.
        qkeras_average_quantizer = layer.get_quantizers()[0]
        qtools_average_quantizer = quantizer_factory.make_quantizer(
            qkeras_average_quantizer)
        multiplier = multiplier_factory.make_multiplier(
            accumulator.output, qtools_average_quantizer)
      else:
        multiplier = None
      if debug:
        print("accumulator:", accumulator.output.bits)

      # Re-calcualte accumulator/multiplier type when it's using
      # for_reference option
      if for_reference:
        if keras_accumulator:
          # If keras_accumulator exists, use keras_accumulator as multiplier
          # or accumulator type
          if multiplier:
            # Quantized layers need to define multiplier type
            multiplier.output = quantizer_factory.make_default_quantizer(
                mode=keras_accumulator)
          accumulator.output = quantizer_factory.make_default_quantizer(
              mode=keras_accumulator)
        else:
          # If user didn't provide keras_accumulator, use the default settings
          # in cfg to define multiplier/accumulator type
          if multiplier:
            multiplier.output = quantizer_factory.make_default_quantizer(
                mode=cfg.default_interm_quantizer)
          accumulator.output = quantizer_factory.make_default_quantizer(
              mode=cfg.default_interm_quantizer)
        layer_quantizer = accumulator.output

      # set the output quantizer
      if layer.__class__.__name__ in ["QAveragePooling2D",
                                      "QGlobalAveragePooling2D"]:
        # If is quantized layer, last operation is multiply (averaging).
        layer_quantizer = multiplier.output
      else:
        layer_quantizer = accumulator.output
      output_quantizer = update_output_quantizer_in_graph(
          graph, node_id, quantizer_factory, layer_quantizer, for_reference)

      layer_data_type_map[layer] = {
          "input_quantizer_list": input_quantizer_list,
          "average_quantizer": qtools_average_quantizer,
          "pool_sum_accumulator": accumulator,
          "pool_avg_multiplier": multiplier,
          "output_quantizer": output_quantizer,
          "output_shapes": output_shapes,
          "operation_count": operation_count
      }

    # If it's a Quantized Activation layer.
    elif node_type in ["QActivation", "QAdaptiveActivation", "Activation"]:

      if for_reference or not hasattr(layer, "quantizer"):
        # Keras activation layer -> use default_interm_quantizer
        layer_quantizer = quantizer_factory.make_default_quantizer(
            mode=cfg.default_interm_quantizer)

        if keras_accumulator:
          layer_quantizer = quantizer_factory.make_default_quantizer(
              mode=keras_accumulator)
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
      # If this batchnorm layer needs to be fused with the previous layer,
      # we pass the input quantizer type as the output type in qraph.

      (input_quantizer, _) = input_qe_list[0]

      if  (hw_weight_dict is not None and
           hw_weight_dict[layer.name]["enable_bn_fusing"]):
        if for_reference and keras_accumulator and not is_input_layer:
          input_quantizer = quantizer_factory.make_default_quantizer(
              mode=keras_accumulator)
        output_quantizer = update_output_quantizer_in_graph(
            graph, node_id, quantizer_factory, input_quantizer, for_reference)
        layer_data_type_map[layer] = {
          "input_quantizer_list": input_quantizer_list,
          "output_quantizer": output_quantizer,
          "output_shapes": input_shape,
          "operation_count": operation_count
        }
      else:
        (gamma_quantizer, beta_quantizer, mean_quantizer, variance_quantizer,
         _) = get_bn_quantizers(layer, quantizer_factory, cfg, keras_quantizer,
                                input_quantizer, is_inference, for_reference,
                                model_weights_already_quantized)

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

        layer_quantizer = qbn.internal_output.output
        output_quantizer = update_output_quantizer_in_graph(
            graph, node_id, quantizer_factory, layer_quantizer, for_reference)
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
    # If qdense, qconv, qpool, qoctave
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

      # TODO(lishanok): During inference, if weight and bias is po2,
      #  need to update corresponding quantizer type with min and max
      #  of the constant values.
      if is_inference:
        weights = qtools_util.get_weights(
            layer, model_weights_already_quantized)
        if weight_quantizer.is_po2:
          weight_quantizer.update_inference_values(weights[0])

        if bias_quantizer.is_po2:
          bias_quantizer.update_inference_values(weights[1])

      multiplier_factory = quantized_operators.MultiplierFactory()
      multiplier = multiplier_factory.make_multiplier(
          weight_quantizer, input_quantizer)

      enable_bn_fusing = (
          hw_weight_dict is not None and hw_weight_dict.get(layer.name, None)
          and hw_weight_dict[layer.name].get("enable_bn_fusing", None))

      if enable_bn_fusing and qkeras_weight_quantizer:
        # When conv layer is fused wiht bn, multiplier bitwidth is ajusted by
        # kernel quantizer scale values (for auto_po2 type of quantizer only).
        # For conv layer without fusing, multiplier bitwidth is not adjusted
        # even if auto_po2 is used in quantizer. Instead, we directly adjusted
        # the accumulator and store that in fused_accumulator.
        qtools_util.adjust_multiplier_for_auto_po2(
            multiplier, qkeras_weight_quantizer)

      weights = layer.get_weights()
      kernel = weights[0]

      kernel_shape = kernel.shape
      # depthwise_kernel_shape = kernel_size + (input_dim, depth_multiplier)
      # When computing accumulator bitwidth for dw conv2d layer, we do not
      # need to count the last two dimensions
      if node_type in ["QDepthwiseConv2D", "DepthwiseConv2D"]:
        kernel_shape = kernel.shape[:-2] + (1, 1)

      kernel_accumulator_factory = quantized_operators.AccumulatorFactory()
      # Sets use_bias=False so that the accumulator doesn't account for bias
      # bitwdith.
      kernel_accumulator = kernel_accumulator_factory.make_accumulator(
          kernel_shape, multiplier, use_bias=False)

      if not layer.use_bias:
        bias_quantizer = None
        accumulator = kernel_accumulator
      else:
        # Add bias quantizer bitwidth to the overall accumulator
        bias_accumulator_instance = adder_factory.IAdder()
        accumulator = bias_accumulator_instance.make_quantizer(
            kernel_accumulator.output, bias_quantizer)
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

      if enable_bn_fusing:
        bn_layer_name = hw_weight_dict[layer.name]["fused_bn_layer_name"]
        successor_ids = list(graph.successors(node_id))
        bn_layer = graph.nodes[successor_ids[0]]["layer"][0]
        assert bn_layer.name == bn_layer_name, (
            "Batchnorm layer in the graph has different name from hw_weight"
            f"_dict: {layer.name} vs {bn_layer_name}. Check both places to "
            "ensure they are matching.")

        # Add additional datatype for bn fused weights
        (gamma_quantizer, beta_quantizer, mean_quantizer, variance_quantizer,
         inverse_quantizer) = get_bn_quantizers(
             bn_layer, quantizer_factory, cfg, keras_quantizer, input_quantizer,
             is_inference, for_reference, model_weights_already_quantized)

        qkeras_inverse_quantizer = bn_layer.inverse_quantizer_internal
        fused_bn = FusedBNFactory()
        fused_bn.make_quantizer(
            prev_output_quantizer=kernel_accumulator.output,
            prev_bias_quantizer=bias_quantizer,
            beta_quantizer=beta_quantizer,
            mean_quantizer=mean_quantizer,
            inverse_quantizer=inverse_quantizer,
            use_beta=bn_layer.center,
            use_bias=layer.use_bias,
            qkeras_inverse_quantizer=qkeras_inverse_quantizer
        )
        if for_reference or not hasattr(layer, "get_quantizers"):
          fused_bn.internal_accumulator.output = (
              quantizer_factory.make_default_quantizer(
                  mode=cfg.default_interm_quantizer))
          if keras_accumulator:
            fused_bn.internal_accumulator.output = (
                quantizer_factory.make_default_quantizer(
                    mode=keras_accumulator))
          fused_bn.internal_output.output = fused_bn.internal_accumulator.output

        layer_quantizer = fused_bn.internal_accumulator.output
        output_quantizer = update_output_quantizer_in_graph(
            graph, node_id, quantizer_factory, layer_quantizer, for_reference)
        layer_data_type_map[layer] = {
            "input_quantizer_list": input_quantizer_list,
            "multiplier": multiplier,
            "accumulator": accumulator,
            "weight_quantizer": weight_quantizer,
            "w_shapes": w_shapes,
            "bias_quantizer": bias_quantizer,
            "b_shapes": b_shapes,
            "bn_inverse_quantizer": inverse_quantizer,
            "bn_mean_quantizer": mean_quantizer,
            "bn_beta_quantizer": beta_quantizer,
            "fused_accumulator": fused_bn.internal_accumulator,
            "output_quantizer": output_quantizer,
            "output_shapes": output_shapes,
            "operation_count": operation_count
        }
      else:
        # Correct accumulator bitwith with the scale values from
        # auto-po2 type of quantizers and store them in fused_accumulator.
        if (
            hasattr(qkeras_weight_quantizer, "__str__") and
            ("quantized_bits" in qkeras_weight_quantizer.__str__() or
             "quantized_linear" in qkeras_weight_quantizer.__str__()) and
            qkeras_weight_quantizer.alpha == "auto_po2"):
          fused_accumulator = qtools_util.adjust_accumulator_for_auto_po2(
              layer, multiplier, qkeras_weight_quantizer, bias_quantizer)
        else:
          fused_accumulator = accumulator

        layer_quantizer = accumulator.output
        output_quantizer = update_output_quantizer_in_graph(
            graph, node_id, quantizer_factory, layer_quantizer, for_reference)

        layer_data_type_map[layer] = {
            "input_quantizer_list": input_quantizer_list,
            "multiplier": multiplier,
            "accumulator": accumulator,
            "weight_quantizer": weight_quantizer,
            "w_shapes": w_shapes,
            "bias_quantizer": bias_quantizer,
            "b_shapes": b_shapes,
            "fused_accumulator": fused_accumulator,
            "output_quantizer": output_quantizer,
            "output_shapes": output_shapes,
            "operation_count": operation_count
        }
    elif node_type in ["QConv2DBatchnorm", "QDepthwiseConv2DBatchnorm"]:
      # Datatype for Folded Conv/DepthwiseConv layer
      # TODO(lishanok): Add additional support for Folded Dense layer
      (input_quantizer, _) = input_qe_list[0]
      if for_reference or not hasattr(layer, "get_quantizers"):
        # For_reference: force all quantizers to keras_quantizer.
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
        # QKeras layer
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

        if qkeras_bias_quantizer:
          bias_quantizer = quantizer_factory.make_quantizer(
              qkeras_bias_quantizer)
        else:
          bias_quantizer = None

      # TODO(lishanok): During inference, if weight and bias is po2,
      #  need to update corresponding quantizer type with min and max
      #  of the constant values
      if is_inference:
        weights = qtools_util.get_weights(
            layer, model_weights_already_quantized)
        if weight_quantizer.is_po2:
          weight_quantizer.update_inference_values(weights[0])

        if bias_quantizer and bias_quantizer.is_po2:
          bias_quantizer.update_inference_values(weights[1])

      multiplier_factory = quantized_operators.MultiplierFactory()
      multiplier = multiplier_factory.make_multiplier(
          weight_quantizer, input_quantizer)
      if qkeras_weight_quantizer:
        qtools_util.adjust_multiplier_for_auto_po2(
            multiplier, qkeras_weight_quantizer)
      weights = layer.get_weights()
      kernel = weights[0]

      accumulator_factory = quantized_operators.AccumulatorFactory()
      accumulator = accumulator_factory.make_accumulator(
          kernel.shape, multiplier, use_bias=True if bias_quantizer else False)

      if not bias_quantizer:
        # Set bias the same as accumulator type.
        bias_quantizer = copy.deepcopy(accumulator.output)
        if not accumulator.output.is_floating_point:
          # For fixed point accumulator, needs to add 1 to its bits to avoid
          # possible satuation.
          accumulator.output.bits += 1
          accumulator.output.int_bits += 1
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
      # Any other unsupported layer types -> pass the input quantizer
      # type to output in qraph
      print(f"[WARNING] QTools cannot parse {node_type}. The input quatnizer"
            " of this layer is directly passed through to the output!",
            file=sys.stderr)

      (input_quantizer, _) = input_qe_list[0]

      if for_reference and keras_accumulator and not is_input_layer:
        input_quantizer = quantizer_factory.make_default_quantizer(
            mode=keras_accumulator)

      output_quantizer = update_output_quantizer_in_graph(
          graph, node_id, quantizer_factory, input_quantizer, for_reference)

      layer_data_type_map[layer] = LayerDataType(input_quantizer_list, None,
                                                 None, None, None, None, None,
                                                 output_quantizer,
                                                 output_shapes, operation_count)

  result = {
      "source_quantizer_list": source_quantizer_list,
      "output_layers": output_layers,
      "input_layers": input_layers,
      "layer_data_type_map": layer_data_type_map
  }

  return result
