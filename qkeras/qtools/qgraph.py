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
"""Creates networkx graph from a model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import networkx as nx
import tensorflow.keras.backend as K

from qkeras.qtools.quantized_operators import quantizer_factory as quantizer_factory_module
from qkeras.qtools.settings import cfg
from tensorflow.keras.layers import InputLayer

SOURCE = -1
SINK = -2


class WrongInputQuantizerError(ValueError):
  pass


def GraphRemoveNode(graph, v):
  """Removes node "v" from u -> v -> w, connecting u -> w."""

  incoming = [u for u in graph.predecessors(v) if u != v]
  outgoing = [w for w in graph.successors(v) if w != v]

  # add incoming edges
  for u in incoming:
    for w in outgoing:
      in_attr = graph[u][v]
      out_attr = graph[v][w]

      assert list(in_attr["shape"]) == list(out_attr["shape"])

      graph.add_edges_from([(u, w, out_attr)])

  graph.remove_node(v)


def GraphRemoveNodeWithNodeType(graph, node_type):
  """Removes node with attribute node_type, reconnecting network."""

  nodes_to_remove = [v for v in graph.nodes
                     if graph.nodes[v]["type"][-1] == node_type]

  for v in nodes_to_remove:

    GraphRemoveNode(graph, v)


def  GraphAddHiddenInputLayer(model, graph, input_quantizer_map):
  """For Keras Sequential model api, input layer is hidden. Need to add it."""

  node_id = -1
  for (u, _) in graph.nodes.items():
    if u >= node_id:
      node_id = u
    if u == SOURCE or u == SINK:
      continue

    if graph.nodes[u]["type"][-1] == "InputLayer":
      return

  # determine a node id for the newly added input layer
  node_id += 1

  # find the first layer of the sequential model
  first_layer_nodes = []
  for u in graph.nodes:
    if u == SOURCE or u == SINK:
      continue
    predecessors = list(graph.predecessors(u))
     # find the first layer which doesn't have a parent
    if not predecessors:
      first_layer_nodes.append(u)
  assert len(first_layer_nodes) == 1
  # since it is a sequential model, there is only one first layer
  v_id = first_layer_nodes[0]

  # create a input layer node
  node_type = "InputLayer"
  input_shape = model.layers[0].input_shape
  layer = InputLayer(input_shape=input_shape[1:])
  o_shape = input_shape
  node = (node_id, {"layer": [layer], "type": [node_type],
                    "out_quantizer": None})
  graph.add_nodes_from([node])

  # insert input_quantizers on the edge between input layer and its next layer
  for (a, _) in input_quantizer_map.items():
    edge = (node_id, v_id, {
        "shape": [o_shape], "tensor": a,
        "quantizer": input_quantizer_map[a]})

  graph.add_edges_from([edge])


def GraphAddSingleSourceSingleSink(graph):

  """Connects graph to source and sink nodes."""

  edge_list = []

  for u in graph.nodes:

    if u == SOURCE or u == SINK:
      continue

    if graph.nodes[u]["type"][-1] == "InputLayer":
      # If the layer has multiple nodes, you can use get_output_at(node_index)
      tensor = graph.nodes[u]["layer"][-1].get_output_at(0)
      # if tf 1.0+, we can do tensor.shape with the same effect
      shape = tuple(tensor.get_shape().as_list())
      shape = [shape]

      edge_list.append((SOURCE, u, {
          "shape": shape, "tensor": tensor, "quantizer": None}))

    if graph.out_degree(u) == 0:
      tensor = graph.nodes[u]["layer"][-1].get_output_at(0)
      shape = tensor.shape

      edge_list.append((u, SINK, {
          "shape": shape, "tensor": tensor, "quantizer": None}))

  graph.add_edges_from(edge_list)


def GenerateGraphFromModel(model, input_quantizers,
                           default_source_quantizer):
  """Generates single source, single sink graph from model."""

  # node represents layers with attributes [layer, type(class_name)]
  # edge represents the tensor flowing between two layers,
  # attributes is [tensor, output_shape, QA(activation quantizer]

  # input_quantizers are tagged on the edge between input
  # layer and the following layer

  # generate a list of input quantizers
  input_quantizer_list = []
  quantizer_factory = quantizer_factory_module.QuantizerFactory()
  if input_quantizers is None:
    logging.warning(
        "************ SOURCE has no quantizer type."
        " Use default quantizer instead")

    for _ in range(len(model.inputs)):
      input_quantizer_list.append(
          quantizer_factory.make_default_quantizer(
              mode=default_source_quantizer))
  else:
    if len(model.inputs) == len(input_quantizers):
      for quantizer in input_quantizers:
        input_quantizer_list.append(quantizer_factory.make_quantizer(
            quantizer))
    # pass a single quantizer which will be used for all q list.
    elif not isinstance(input_quantizers, list):
      for _ in range(len(model.inputs)):
        input_quantizer_list.append(quantizer_factory.make_quantizer(
            input_quantizers))
    else:
      raise WrongInputQuantizerError(
          "ERROR: Numer of input (%d) must be the same as number of source"
          " quantizers (%d)"%(len(model.inputs), len(input_quantizers)))

  # dict that map input_tensor to its quantizer
  input_quantizer_map = {}
  for (idx, tensor) in enumerate(model.inputs):
    input_quantizer_map[tensor.experimental_ref()] = input_quantizer_list[idx]

  graph = nx.DiGraph()

  source = SOURCE
  sink = SINK

  node_list = [
      (source, {"layer": [None], "type": [None], "out_quantizer": None}),
      (sink, {"layer": [None], "type": [None], "out_quantizer": None})
  ]

  graph.add_nodes_from(node_list)

  node_list = []

  for i, layer in enumerate(model.layers):

    node_type = layer.__class__.__name__

    node = (i, {"layer": [layer], "type": [node_type], "out_quantizer": None})
    node_list.append(node)

  node_dict = {layer: i for i, layer in enumerate(model.layers)}

  graph.add_nodes_from(node_list)

  # nodes = tensors
  in_nodes = {}
  out_nodes = {}
  for layer in model.layers:
    i_list = layer.input
    if not isinstance(layer.input, list):
      i_list = [i_list.experimental_ref()]
    else:
      i_list = [tmp.experimental_ref() for tmp in i_list]

    for i in i_list:
      # dict: tensor -> layers have this tensor as input
      if i not in in_nodes.keys():
        in_nodes[i] = [layer]
      else:
        in_nodes[i].append(layer)

    o_list = layer.output
    if not isinstance(layer.output, list):
      o_list = [o_list.experimental_ref()]
    else:
      o_list = [tmp.experimental_ref() for tmp in o_list]

    for o in o_list:
      # dict: tensor -> layer have this tensor as output
      if o not in out_nodes.keys():
        out_nodes[o] = [layer]
      else:
        out_nodes[o].append(layer)

  # union of all tensors; non-redundant
  attr_set = set(in_nodes.keys()) | set(out_nodes.keys())

  # add edges. we want edges annotated with tensors and shapes

  edge_list = []

  for a in attr_set:
    # for a given tensor a, find the layer u that outputs this tensor
    # and the layer v that has this tensor as input
    u_list = out_nodes.get(a, [None])
    v_list = in_nodes.get(a, [None])

    for u in u_list:
      for v in v_list:
        if not u or not v:
          continue

        o_shape = u.output_shape

        # layer -> layer_id
        u_id = node_dict[u]
        v_id = node_dict[v]

        # insert input_quantizers on the edge between
        # input layer and its next layer
        if a in input_quantizer_map.keys():
          edge_list.append((u_id, v_id, {
              "shape": o_shape, "tensor": a,
              "quantizer": input_quantizer_map[a]}))
        else:
          edge_list.append((u_id, v_id, {
              "shape": o_shape, "tensor": a,
              "quantizer": None}))

  graph.add_edges_from(edge_list)
  GraphAddHiddenInputLayer(model, graph, input_quantizer_map)

  return (graph, input_quantizer_list)


def GraphGetInputs(graph):

  """Returns edges SOURCE->u that are inputs."""

  successors = list(graph.successors(SOURCE))

  input_tensors = []

  for u in successors:

    if u == SOURCE or u == SINK:
      continue

    input_tensors.append(graph[SOURCE][u])

  return input_tensors


def GraphGetOutputs(graph):

  """Returns edges u->SINK that are outputs."""

  predecessors = list(graph.predecessors(SINK))

  output_tensors = []

  for u in predecessors:

    if u == SOURCE or u == SINK:
      continue

    output_tensors.append(graph[u][SINK])

  return output_tensors


def GraphPropagateActivationsToEdges(graph, debug=False):
  """Traverses graph and move activations to edges.

  1.If current dense/conv layer is specified with QA:
    outgoing edge (output data type) will be QA type
  2.If current dense/conv layer has no QA:
    default type (float32) is used as output
  3.If current layer is QA layer:
    float32 is used by default as output type on the edge

  Args:
    graph: graph to inject activations to.
    debug: debug mode

  Returns:
    None
  """

  scheduler = list(nx.topological_sort(graph))

  for vertex in scheduler[1:-1]:
    # get rid of source and sink vertex
    if debug:
      print("########### GraphPropagateActivationsToEdges ############")
      print("vertex:", vertex)

    for u, v in graph.edges(vertex):
      # u=vertex, v: outgoing edge vertex

      if debug:
        print("  outgoing ->", v, graph.nodes[v]["layer"][0].name)

      layer = graph.nodes[u]["layer"][0]
      result = None
      # if current layer has no QA specified
      if not hasattr(layer, "activation"):
        result = None
      else:
        activation_name = layer.activation.__name__ if hasattr(
            layer.activation, "__name__") else None
        q_activation_class_name = layer.activation.__class__.__name__ if hasattr(
            layer.activation, "__class__") else None

        if debug:
          print("  layer type:", layer.__class__.__name__)
          print("  activation object:", layer.activation)
          print("  activation_name:", activation_name)
          print("  q_activation_class_name:", q_activation_class_name)

        # if current layer is QA
        if (graph.nodes[u]["type"][0] in ["QActivation"] or
            graph.nodes[u]["type"][0] in ["QAdaptiveActivation"]):
          result = layer.quantizer

        # if current layer is not QA layer but has QA specified within
        elif hasattr(layer, "activation"):
          if activation_name == "linear":
            result = None
          else:
            result = layer.activation

      if debug:
        print("  {}->{}: {}".format(u, v, result))

      graph[u][v]["quantizer"] = result
      # all edge_quantizer is the same for all edges starting
      # from current vertex to different nodes
      graph.nodes[vertex]["out_quantizer"] = result


def PrintGraph(graph, msg=""):
  """Print graph structure."""

  print()
  print(msg)
  print()
  print("nodes:",
        [(u, graph.nodes[u]["layer"][
            0].name if graph.nodes[u]["layer"][0] is not None else "",
          graph.nodes[u]["type"]) for u in graph.nodes])
  print()
  print("edges:",
        [(u, v, graph[u][v]["shape"],
          graph[u][v]["quantizer"]) for u, v in graph.edges])


def CreateGraph(model, input_quantizers=None,
                default_source_quantizer=cfg.default_source_quantizer,
                debug=False):
  """create graph."""

  K.set_image_data_format("channels_last")

  (graph, source_quantizer_list) = GenerateGraphFromModel(
      model, input_quantizers, default_source_quantizer)
  GraphAddSingleSourceSingleSink(graph)
  GraphRemoveNodeWithNodeType(graph, "Dropout")
  GraphRemoveNodeWithNodeType(graph, "InputLayer")

  scheduler = list(nx.topological_sort(graph))

  if debug:
    for vertex in scheduler[1:-1]:
      for _, v in graph.edges(vertex):
        if v == SINK:
          continue
        print("... calling", graph.nodes[v][
            "layer"][0].name, graph.nodes[v]["type"])

  return (graph, source_quantizer_list)


def GraphUpdateEdge(graph, node_id, quantizer_on_edge):
  """update the graph edges outgoing from node_id with new quantizer."""

  for u, v in graph.edges(node_id):
    graph[u][v]["quantizer"] = quantizer_on_edge
