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
"""Utility functions for folding batchnorm with qconv/qdense layers."""

import networkx as nx
from tensorflow.keras.models import Model

from .qconvolutional import QConv2D
from .qtools import qgraph


def replace_layers(src_layer, inputs, replace_to):
  """Replace a source layer with a different layer type and set the weights.

  Args:
    src_layer: keras/qkeras layer type
    inputs: tensor type. input of the src layer
    replace_to: string. replace src layer with the new layer type.

  Returns:
    the new layer and its output tensor.
  """

  # get layer config from the composite layer
  config = src_layer.get_config()

  # set layer config for QConv2D layer by first creating a tmp
  # QConv2D object and generate template for its config
  if replace_to == "QConv2D":
    new_layer = QConv2D(filters=1, kernel_size=(2, 2))
    new_layer_cfg = new_layer.get_config()
  else:
    # TODO(lishanok): will extend to other layer types in the future
    assert ValueError, "%s is not supported!" % replace_to

  # set qconv2d config according to the values in the composite layer
  for (key, _) in new_layer_cfg.items():
    new_layer_cfg[key] = config[key]

  # in case use_bias is False in the composite layer,
  #  we need to set it True because we have folded bias
  new_layer_cfg["use_bias"] = True

  # create a qconv2d layer from config and replace old layer with it
  if replace_to == "QConv2D":
    new_layer = QConv2D.from_config(new_layer_cfg)
  outputs = new_layer(inputs)

  # transfer weights from composite layer to the target layer
  folded_kernel_quantized = src_layer.folded_kernel_quantized.numpy()
  folded_bias_quantized = src_layer.folded_bias_quantized.numpy()
  new_layer.set_weights([folded_kernel_quantized, folded_bias_quantized])

  return (new_layer, outputs)


def convert_folded_model_to_normal_seq(model):
  """Convert a sequential model with batchnorm folded layer to a normal model.

  Replace the folded layers with a normal qconv/qdense layer.
  Set the weights in the normal layer with the folded weights
  in the folded layer.

  we need to convert a folded model to a normal model before we pass it
  to hardware.

  Note: this function only supports sequential model and is therefore
  deprecated. A newer version, convert_folded_model_to_normal(), which
  supports both sequential and non-sequential models, replaces this function.

  Arguments:
    model: model with folded layers.

  Returns:
    A model that replaces folded layers (e.g., QConv2DBatchnorm) with normal
      qkeras layers (e.g., QConv2D). This model can be passed on to hardware
      generator (zpm) so that hardware doesn't see batch normalization
      parameters.
  """

  layer_list = list(model.layers)
  x = layer_list[0].output

  for i in range(1, len(layer_list)):
    layer = layer_list[i]

    if layer.__class__.__name__ not in ["QConv2DBatchnorm"]:
      x = layer_list[i](x)

    else:
      # get layer config from the composite layer
      config = layer.get_config()

      # set layer config for QConv2D layer by first creating a tmp
      # QConv2D object and generate template for its config
      qconv2d = QConv2D(filters=1, kernel_size=(2, 2))
      qconv2d_cfg = qconv2d.get_config()

      # set qconv2d config according to the values in the composite layer
      for (key, _) in qconv2d_cfg.items():
        qconv2d_cfg[key] = config[key]

      # in case use_bias is False in the composite layer,
      #  we need to set it True because we have folded bias
      qconv2d_cfg["use_bias"] = True

      # create a qconv2d layer from config and replace old layer with it
      qconv2d = QConv2D.from_config(qconv2d_cfg)
      x = qconv2d(x)

      # transfer weights from composite layer to the qconv2d layer
      for weight in layer.weights:
        if "folded_kernel_quantized" in weight.name:
          folded_kernel_quantized = weight.numpy()
        elif "folded_bias_quantized" in weight.name:
          folded_bias_quantized = weight.numpy()
      qconv2d.set_weights([folded_kernel_quantized, folded_bias_quantized])

  return Model(inputs=model.inputs, outputs=x)


def convert_folded_model_to_normal(fold_model):
  """Convert a model with batchnorm folded layer to a normal model.

  "Normal" here refers to a model with separate qconv/qdense and batchnorm
  layer. This function replace the folded layers with a normal qconv/qdense
  layer. It also sets the weights in the normal layer with the folded weights
  in the folded layer. Model architecture could be either sequential or
  non-sequential.

  Arguments:
    fold_model: keras object, model with folded layers.

  Returns:
    A model that replaces folded layers (e.g., QConv2DBatchnorm) with normal
      qkeras layers (e.g., QConv2D). This model can be passed on to hardware
      generator so that hardware doesn't see batch normalization
      parameters.
  """

  # Use "quantized_bits(8, 0, 1)" as the default input quantizer. The
  # exact value of the input quantizer has no impact on the function of
  # convert_folded_model_to_normal. It's only used as a placeholder.
  (graph, _) = qgraph.GenerateGraphFromModel(
      fold_model, "quantized_bits(8, 0, 1)", "quantized_bits(8, 0, 1)")

  qgraph.GraphAddSingleSourceSingleSink(graph)
  qgraph.GraphRemoveNodeWithNodeType(graph, "InputLayer")
  qgraph.GraphPropagateActivationsToEdges(graph)

  model_outputs = []
  x = model_inputs = fold_model.inputs

  for node_id in nx.topological_sort(graph):
    layer_input_tensors = []
    node = graph.nodes[node_id]
    is_output_layer = False
    layer = node["layer"][0]
    if layer:
      # get layer input tensors from graph edge
      for parent_node_id in graph.predecessors(node_id):
        edge = graph.edges[(parent_node_id, node_id)]
        input_tensor = edge["tensor"]
        layer_input_tensors.append(input_tensor)

      # replace composite layer with "normal" layer
      # TODO(lishanok): extend to QDense types
      if layer.__class__.__name__ in ["QConv2DBatchnorm"]:
        assert len(layer_input_tensors) == 1
        (_, x) = replace_layers(
            layer, layer_input_tensors[0].deref(), "QConv2D")

      else:
        # for other layers that do not need to be replaced, we simply call the
        # layer to get output tensor
        if len(layer_input_tensors) == 1:
          layer_input_tensors = layer_input_tensors[0].deref()
        else:
          layer_input_tensors = [t.deref() for t in layer_input_tensors]

        x = layer(layer_input_tensors)

      # replace edge tensors between the replaced layer and successor layers
      for u, v in graph.edges(node_id):
        # u is current layer node, v is successor layer node
        # graph[u][v] is the edge between the two nodes
        # Replace the tensor on this edge so that the input tensor for the
        # successor layer can be updated accordingly.
        graph[u][v]["tensor"] = x.ref()

        if v == -2:
          # is output layer:
          model_outputs.append(x)
          break

  return Model(inputs=model_inputs, outputs=model_outputs)
