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
import copy
import json
import tempfile
import types

import numpy as np
import os
import six
import re
import networkx as nx
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import initializers
from tensorflow.python.keras.layers.core import TFOpLambda

from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json

from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from tensorflow_model_optimization.python.core.sparsity.keras import prune_registry
from tensorflow_model_optimization.python.core.sparsity.keras import prunable_layer

from .qlayers import Clip
from .qconv2d_batchnorm import QConv2DBatchnorm
from .qdepthwiseconv2d_batchnorm import QDepthwiseConv2DBatchnorm
from .qlayers import QActivation
from .qlayers import QAdaptiveActivation
from .qpooling import QAveragePooling2D
from .qlayers import QDense
from .qlayers import QInitializer
from .qconvolutional import QConv1D
from .qconvolutional import QConv2D
from .qconvolutional import QConv2DTranspose
from .qrecurrent import QSimpleRNN
from .qrecurrent import QSimpleRNNCell
from .qrecurrent import QLSTM
from .qrecurrent import QLSTMCell
from .qrecurrent import QGRU
from .qrecurrent import QGRUCell
from .qrecurrent import QBidirectional
from .qconvolutional import QSeparableConv1D
from .qconvolutional import QSeparableConv2D
from .qconvolutional import QDepthwiseConv2D
from .qnormalization import QBatchNormalization
from .qpooling import QGlobalAveragePooling2D
from .qtools import qgraph
from .quantizers import binary
from .quantizers import bernoulli
from .quantizers import get_weight_scale
from .quantizers import quantized_bits
from .quantizers import quantized_relu
from .quantizers import quantized_ulaw
from .quantizers import quantized_tanh
from .quantizers import quantized_sigmoid
from .quantizers import quantized_po2
from .quantizers import quantized_relu_po2
from .quantizers import stochastic_binary
from .quantizers import stochastic_ternary
from .quantizers import ternary

from .safe_eval import safe_eval

REGISTERED_LAYERS = [
    "QActivation",
    "QAdaptiveActivation",
    "QDense",
    "QConv1D",
    "QConv2D",
    "QSeparableConv1D",
    "QSeparableConv2D",
    "QDepthwiseConv2D",
    "QConv2DTranspose",
    "QSimpleRNN",
    "QLSTM",
    "QGRU",
    "QBidirectional",
    "QBatchNormalization",
    "QConv2DBatchnorm",
    "QDepthwiseConv2DBatchnorm",
    "QAveragePooling2D",
    "QGlobalAveragePooling2D",
]


#
# Model utilities: before saving the weights, we want to apply the quantizers
#
def model_save_quantized_weights(model, filename=None):
  """Quantizes model for inference and save it.

  Takes a model with weights, apply quantization function to weights and
  returns a dictionary with quantized weights.

  User should be aware that "po2" quantization functions cannot really
  be quantized in meaningful way in Keras. So, in order to preserve
  compatibility with inference flow in Keras, we do not covert "po2"
  weights and biases to exponents + signs (in case of quantize_po2), but
  return instead (-1)**sign*(2**round(log2(x))). In the returned dictionary,
  we will return the pair (sign, round(log2(x))).

  Arguments:
    model: model with weights to be quantized.
    filename: if specified, we will save the hdf5 containing the quantized
      weights so that we can use them for inference later on.

  Returns:
    dictionary containing layer name and quantized weights that can be used
    by a hardware generator.

  """

  saved_weights = {}

  print("... quantizing model")
  for layer in model.layers:
    if hasattr(layer, "get_quantizers"):
      weights = []
      signs = []

      if any(isinstance(layer, t) for t in [
          QConv2DBatchnorm, QDepthwiseConv2DBatchnorm]):
        qs = layer.get_quantizers()
        ws = layer.get_folded_weights()
      elif any(isinstance(layer, t) for t in [QSimpleRNN, QLSTM, QGRU]):
        qs = layer.get_quantizers()[:-1]
        ws = layer.get_weights()
      else:
        qs = layer.get_quantizers()
        ws = layer.get_weights()

      has_sign = False

      for quantizer, weight in zip(qs, ws):
        if quantizer:
          weight = tf.constant(weight)
          weight = tf.keras.backend.eval(quantizer(weight))

        # If quantizer is power-of-2 (quantized_po2 or quantized_relu_po2),
        # we would like to process it here.
        #
        # However, we cannot, because we will loose sign information as
        # quanized_po2 will be represented by the tuple (sign, log2(abs(w))).
        #
        # In addition, we will not be able to use the weights on the model
        # any longer.
        #
        # So, instead of "saving" the weights in the model, we will return
        # a dictionary so that the proper values can be propagated.

        weights.append(weight)

        has_sign = False
        if quantizer:
          if isinstance(quantizer, six.string_types):
            q_name = quantizer
          elif hasattr(quantizer, "__name__"):
            q_name = quantizer.__name__
          elif hasattr(quantizer, "name"):
            q_name = quantizer.name
          elif hasattr(quantizer, "__class__"):
            q_name = quantizer.__class__.__name__
          else:
            q_name = ""
        if quantizer and ("_po2" in q_name):
          # Quantized_relu_po2 does not have a sign
          if isinstance(quantizer, quantized_po2):
            has_sign = True
          sign = np.sign(weight)
          # Makes sure values are -1 or +1 only
          sign += (1.0 - np.abs(sign))
          weight = np.round(np.log2(np.abs(weight)))
          signs.append(sign)
        else:
          signs.append([])

      saved_weights[layer.name] = {"weights": weights}
      if has_sign:
        saved_weights[layer.name]["signs"] = signs

      if not any(isinstance(layer, t) for t in [
          QConv2DBatchnorm, QDepthwiseConv2DBatchnorm]):
        layer.set_weights(weights)
      else:
        print(layer.name, " conv and batchnorm weights cannot be seperately"
              " quantized because they will be folded before quantization.")
    else:
      if layer.get_weights():
        print(" ", layer.name, "has not been quantized")

  if filename:
    model.save_weights(filename)

  return saved_weights


def quantize_activation(layer_config, activation_bits):
  """Replaces activation by quantized activation functions."""
  str_act_bits = str(activation_bits)
  # relu -> quantized_relu(bits)
  # tanh -> quantized_tanh(bits)
  # sigmoid -> quantized_sigmoid(bits)
  # more to come later
  if layer_config.get("activation", None) is None:
    return
  if isinstance(layer_config["activation"], six.string_types):
    a_name = layer_config["activation"]
  elif isinstance(layer_config["activation"], types.FunctionType):
    a_name = layer_config["activation"].__name__
  else:
    a_name = layer_config["activation"].__class__.__name__

  if a_name == "linear":
    return
  if a_name == "relu":
    layer_config["activation"] = "quantized_relu(" + str_act_bits + ")"
  elif a_name == "tanh":
    layer_config["activation"] = "quantized_tanh(" + str_act_bits + ")"
  elif a_name == "sigmoid":
    layer_config["activation"] = "quantized_sigmoid(" + str_act_bits + ")"


def get_config(quantizer_config, layer, layer_class, parameter=None):
  """Returns search of quantizer on quantizer_config."""
  quantizer = quantizer_config.get(layer["config"]["name"],
                                   quantizer_config.get(layer_class, None))

  if quantizer is not None and parameter is not None:
    quantizer = quantizer.get(parameter, None)

  return quantizer


def is_TFOpLambda_layer(layer):
  return (isinstance(layer, TFOpLambda) or
          layer.__class__.__name__ == "TFOpLambda")


def get_y_from_TFOpLambda(model_cfg, layer):
  """Get the value of "y" from the TFOpLambda layer's configuration.
  Args:
    model_cfg: dictionary type, model.get_config() output
    layer: a given layer instance

  Return:
    value of "y" for a TFOpLambda layer. 'y' here corresponds to how tensorflow
    stores TFOpLambda layer parameter in serialization. for example,
    TFOpLambda(func), where func is tf.multiply(input_tensor, 3). "y" would be
    the value 3.
  """

  for layer_config in model_cfg["layers"]:
    op_name = layer_config["config"]["name"]
    class_name = layer_config["class_name"]

    # TODO(lishanok): Extend support for other TFOpLambda types when needed
    if op_name == layer.name and  class_name == "TFOpLambda":
      assert ("tf.__operators__.add" in op_name or "tf.math.multiply"
              in op_name), "TFOpLambda layer {} not supported!".format(op_name)
      return layer_config["inbound_nodes"][-1][-1]["y"]

  return None


def convert_to_folded_model(model):
  """Find conv/dense layers followed by bn layers and fold them.

  Args:
    model: input model

  Returns:
    new model without bn layers
    list of layers being folded

  Note: supports sequential and non-sequential model
  """

  fold_model = clone_model(model)
  model_cfg = model.get_config()
  (graph, _) = qgraph.GenerateGraphFromModel(
      fold_model, "quantized_bits(8, 0, 1)", "quantized_bits(8, 0, 1)")

  qgraph.GraphAddSingleSourceSingleSink(graph)
  qgraph.GraphRemoveNodeWithNodeType(graph, "InputLayer")
  qgraph.GraphPropagateActivationsToEdges(graph)

  # Finds the Batchnorm nodes to be deleted and mark them.
  bn_nodes_to_delete = []
  layers_to_fold = []
  for node_id in nx.topological_sort(graph):
    layer_input_tensors = []
    node = graph.nodes[node_id]
    layer = node["layer"][0]
    if layer:
      successor_ids = list(graph.successors(node_id))
      is_single = len(successor_ids) == 1
      successor_layer = graph.nodes[successor_ids[0]]["layer"][0]
      followed_by_bn = (successor_layer.__class__.__name__ ==
                        "BatchNormalization")
      # TODO(lishanok): extend to QDense types
      is_foldable = layer.__class__.__name__ in [
          "Conv2D", "DepthwiseConv2D"
      ] and is_single and followed_by_bn

      if is_foldable:
        # Removes the batchnorm node from the graph.
        bn_nodes_to_delete.append(successor_ids[0])
        layers_to_fold.append(layer.name)

  # Deletes the marked nodes.
  for node_id in bn_nodes_to_delete:
    qgraph.GraphRemoveNode(graph, node_id)

  # Modifies model according to the graph.
  model_outputs = []
  x = model_inputs = fold_model.inputs

  for node_id in nx.topological_sort(graph):
    layer_input_tensors = []
    node = graph.nodes[node_id]

    layer = node["layer"][0]
    if layer:
      # Gets layer input tensors from graph edge.
      for parent_node_id in graph.predecessors(node_id):
        edge = graph.edges[(parent_node_id, node_id)]
        input_tensor = edge["tensor"]
        layer_input_tensors.append(input_tensor)

      # We call the layer to get output tensor.
      if len(layer_input_tensors) == 1:
        layer_input_tensors = layer_input_tensors[0].deref()
      else:
        layer_input_tensors = [t.deref() for t in layer_input_tensors]

      if is_TFOpLambda_layer(layer):
        # TFOpLambda layer requires one extra input: "y"
        y = get_y_from_TFOpLambda(model_cfg, layer)
        x = layer(layer_input_tensors, y)
      else:
        x = layer(layer_input_tensors)

      # Replaces edge tensors between the predecessor and successor
      for u, v in graph.edges(node_id):
        # u is current layer node, v is successor layer node
        # graph[u][v] is the edge between the two nodes
        # Replace the tensor on this edge so that the input tensor for the
        # successor layer can be updated accordingly.
        graph[u][v]["tensor"] = x.ref()

        if v == -2 and x not in model_outputs:
          # When it is output layer, add the output tensor of this layer
          # into model outputs.
          model_outputs.append(x)

  new_model = Model(inputs=model_inputs, outputs=model_outputs)

  return new_model, layers_to_fold


def model_quantize(model,
                   quantizer_config,
                   activation_bits,
                   custom_objects=None,
                   transfer_weights=False,
                   prefer_qadaptiveactivation=False,
                   enable_bn_folding=False):
  """Creates a quantized model from non-quantized model.

  The quantized model translation is based on json interface of Keras,
  which requires a custom_objects dictionary for "string" types.

  Because of the way json works, we pass "string" objects for the
  quantization mechanisms and we perform an eval("string") which
  technically is not safe, but it will do the job.

  The quantizer_config is a dictionary with the following form.
  {
    Dense_layer_name: {
        "kernel_quantizer": "quantizer string",
        "bias_quantizer": "quantizer_string"
    },

    Conv2D_layer_name: {
        "kernel_quantizer": "quantizer string",
        "bias_quantizer": "quantizer_string"
    },

    Activation_layer_name: "quantizer string",

    "QActivation": { "relu": "quantizer_string" },

    "QConv2D": {
        "kernel_quantizer": "quantizer string",
        "bias_quantizer": "quantizer_string"
    },

    "QBatchNormalization": {}
  }

  In the case of "QBidirectional", we can follow the same form as above.
  The specified configuration will be used for both forward and backwards
  layer.
  {
    "Bidirectional" : {
        "kernel_quantizer" : "quantizer string",
        "bias_quantizer" : "quantizer string",
        "recurrent_quantizer" : "quantizer string"
    }
  }

  In the case of "QActivation", we can modify only certain types of
  activations, for example, a "relu". In this case we represent the
  activation name by a dictionary, or we can modify all activations,
  without representhing as a set.

  We right now require a default case in case we cannot find layer name.
  This simplifies the dictionary because the simplest case, we can just
  say:

  {
    "default": {
        "kernel": "quantized_bits(4)",
        "bias": "quantized_bits(4)"
    }
  }

  and this will quantize all layers' weights and bias to be created with
  4 bits.

  Arguments:
    model: model to be quantized
    quantizer_config: dictionary (as above) with quantized parameters
    activation_bits: number of bits for quantized_relu, quantized_tanh,
      quantized_sigmoid
    custom_objects: dictionary following keras recommendations for json
      translation.
    transfer_weights: if true, weights are to be transfered from model to
      qmodel.
    prefer_qadaptiveactivation: Bool. If true, try to use QAdaptiveActivation
      over QActivation whenever possible
    enable_bn_folding: Bool. If true, fold conv/dense layers with
      following batch normalization layers whenever possible. use
      QConv2DBatchnorm for example, to replace conv2d layers

  Returns:
    qmodel with quantized operations and custom_objects.
  """

  if enable_bn_folding:
    # Removes bn layers from the model and find a list of layers to fold.
    model, layers_to_fold = convert_to_folded_model(model)
    if len(layers_to_fold) == 0:
      # If no layers to fold, no need to perform folding.
      enable_bn_folding = False

  if not custom_objects:
    custom_objects = {}

  # Let's make a deep copy to make sure our objects are not shared elsewhere.
  jm = copy.deepcopy(json.loads(model.to_json()))
  custom_objects = copy.deepcopy(custom_objects)
  config = jm["config"]
  layers = config["layers"]

  def quantize_rnn(layer, quantizer_config):
    q_name = "Q" + layer["class_name"]
    # Needs to add kernel, recurrent bias quantizers.
    kernel_quantizer = get_config(
        quantizer_config, layer, q_name, "kernel_quantizer")
    recurrent_quantizer = get_config(
        quantizer_config, layer, q_name, "recurrent_quantizer")
    if layer["config"]['use_bias']:
      bias_quantizer = get_config(
          quantizer_config, layer, q_name, "bias_quantizer")
    else:
      bias_quantizer = None
    state_quantizer = get_config(
            quantizer_config, layer, q_name, "state_quantizer")

    # This is to avoid unwanted transformations.
    if kernel_quantizer is None:
      return

    layer["config"]["kernel_quantizer"] = kernel_quantizer
    layer["config"]["recurrent_quantizer"] = recurrent_quantizer
    layer["config"]["bias_quantizer"] = bias_quantizer
    layer["config"]["state_quantizer"] = state_quantizer

    # If activation is present, add activation here.
    activation = get_config(
        quantizer_config, layer, q_name, "activation_quantizer")
    if activation:
      layer["config"]["activation"] = activation
    else:
      quantize_activation(layer["config"], activation_bits)

    # If recurrent activation is present, add activation here.
    if layer["class_name"] in ["LSTM", "GRU"]:
      recurrent_activation = get_config(
          quantizer_config, layer, q_name, "recurrent_activation_quantizer")
      if recurrent_activation:
        layer["config"]["recurrent_activation"] = recurrent_activation
    layer["class_name"] = q_name

  for layer in layers:
    layer_config = layer["config"]

    # Dense becomes QDense, Conv1D becomes QConv1D etc
    # Activation converts activation functions.

    if layer["class_name"] in [
      "Dense", "Conv1D", "Conv2D", "Conv2DTranspose",
      "SeparableConv1D", "SeparableConv2D"
    ]:
      if (layer["class_name"] in ["Dense", "Conv2D"] and enable_bn_folding and
          layer["name"] in layers_to_fold):
        # Only fold if current layer is followed by BN layer.
        q_name = "Q" + layer["class_name"] + "Batchnorm"
        layer_config["use_bias"] = True  # Folded layers require a bias

        # Sets ema_freeze_delay and folding_mode specific to
        # QDepthwiseConv2DBatchnorm layer config.
        folding_mode = get_config(
            quantizer_config, layer, q_name, "folding_mode")
        layer_config["folding_mode"] = (
            folding_mode if folding_mode else "ema_stats_folding")
        ema_freeze_delay = get_config(
            quantizer_config, layer, q_name, "ema_freeze_delay")
        layer_config["ema_freeze_delay"] = (
            ema_freeze_delay if ema_freeze_delay else None)
      else:
        q_name = "Q" + layer["class_name"]
      # Needs to add kernel/bias quantizers.
      kernel_quantizer = get_config(
          quantizer_config, layer, q_name, "kernel_quantizer")

      if layer_config["use_bias"]:
        bias_quantizer = get_config(
            quantizer_config, layer, q_name, "bias_quantizer")
      else:
        bias_quantizer = None

      if (kernel_quantizer is None and
          q_name == "Q" + layer["class_name"] + "Batchnorm"):
        # Tries none-folded layer quantizer as a back up.
        kernel_quantizer = get_config(
            quantizer_config, layer, "Q" + layer["class_name"],
            "kernel_quantizer")
        bias_quantizer = get_config(
            quantizer_config, layer, "Q" + layer["class_name"],
            "bias_quantizer")

      # This is to avoid unwanted transformations.
      if kernel_quantizer is None:
        continue

      layer["class_name"] = q_name

      layer_config["kernel_quantizer"] = kernel_quantizer
      layer_config["bias_quantizer"] = bias_quantizer

      # If activation is present, add activation here.
      quantizer = get_config(
          quantizer_config, layer, q_name, "activation_quantizer")

      if quantizer:
        layer_config["activation"] = quantizer
      else:
        quantize_activation(layer_config, activation_bits)

    elif layer["class_name"] == "DepthwiseConv2D":
      if enable_bn_folding and layer["name"] in layers_to_fold:
        q_name = "QDepthwiseConv2DBatchnorm"
        layer_config["use_bias"] = True  # Folded layers require a bias

        # Sets ema_freeze_delay and folding_mode specific to
        # QDepthwiseConv2DBatchnorm layers.
        folding_mode = get_config(
            quantizer_config, layer, q_name, "folding_mode")
        layer_config["folding_mode"] = (
            folding_mode if folding_mode else "ema_stats_folding")
        ema_freeze_delay = get_config(
            quantizer_config, layer, q_name, "ema_freeze_delay")
        layer_config["ema_freeze_delay"] = (
            ema_freeze_delay if ema_freeze_delay else None)

      else:
        q_name = "QDepthwiseConv2D"

      # Needs to add kernel/bias quantizers.
      depthwise_quantizer = get_config(quantizer_config, layer, q_name,
                                       "depthwise_quantizer")

      if layer_config["use_bias"]:
        bias_quantizer = get_config(quantizer_config, layer, q_name,
                                    "bias_quantizer")
      else:
        bias_quantizer = None

      if depthwise_quantizer is None and q_name == "QDepthwiseConv2DBatchnorm":
        # Tries none-folded layer quantizer as a back up.
        depthwise_quantizer = get_config(
            quantizer_config, layer, "QDepthwiseConv2D", "depthwise_quantizer")
        bias_quantizer = get_config(
            quantizer_config, layer, "QDepthwiseConv2D", "bias_quantizer")

      # This is to avoid unwanted transformations.
      if depthwise_quantizer is None:
        continue

      layer["class_name"] = q_name

      layer_config["depthwise_quantizer"] = depthwise_quantizer
      layer_config["bias_quantizer"] = bias_quantizer
      # If activation is present, add activation here.
      quantizer = get_config(quantizer_config, layer, q_name,
                             "activation_quantizer",)

      if quantizer:
        layer_config["activation"] = quantizer
      else:
        quantize_activation(layer_config, activation_bits)

    elif layer["class_name"] in ["SimpleRNN", "LSTM", "GRU"]:
      quantize_rnn(layer, quantizer_config)

    elif layer["class_name"] == "Bidirectional":
      forward_layer_quantizer_config = {
          layer_config["layer"]["config"]["name"]:
              get_config(quantizer_config, layer, "QBidirectional")
      }
      quantize_rnn(layer["config"]["layer"], forward_layer_quantizer_config)
      if "backward_layer" in layer_config:
        backward_layer_quantizer_config = {
            layer_config["backward_layer"]["config"]["name"]:
                get_config(quantizer_config, layer, "QBidirectional")
        }
        quantize_rnn(layer["config"]["backward_layer"],
                     backward_layer_quantizer_config)
      layer["class_name"] = "QBidirectional"

    elif layer["class_name"] == "Activation":
      if prefer_qadaptiveactivation:  # Try to find QAdaptiveActivation first
        quantizer = get_config(quantizer_config, layer, "QAdaptiveActivation")
        is_qadaptiveactivation = True
        if quantizer is None:  # Try QActivation as a backup
          quantizer = get_config(quantizer_config, layer, "QActivation")
          is_qadaptiveactivation = False
      else:  # Tries to find QActivation first.
        quantizer = get_config(quantizer_config, layer, "QActivation")
        is_qadaptiveactivation = False
        if quantizer is None:  # Try QAdaptiveActivation as a backup
          quantizer = get_config(quantizer_config, layer, "QAdaptiveActivation")
          is_qadaptiveactivation = True

      # This is to avoid softmax from quantizing in autoq.
      if quantizer is None:
        continue

      # If quantizer exists in dictionary related to this name,
      # use it, otherwise, use normal transformations.

      if not isinstance(quantizer, dict) or quantizer.get(
          layer_config["activation"], None):
        # Only change activation layer if we will use a quantized activation.

        layer["class_name"] = ("QAdaptiveActivation" if is_qadaptiveactivation
                               else "QActivation")
        if isinstance(quantizer, dict):
          quantizer = quantizer[layer_config["activation"]]
        if quantizer:
          if is_qadaptiveactivation:
            assert quantizer.find(",") < 0, \
                "Only integer bits should be defined for QAdaptiveActivation"
            layer_config["total_bits"] = int(re.sub(r"[^\d]", "", quantizer))
            quantizer = re.sub(r"\(.*", "", quantizer)  # remove params
          layer_config["activation"] = quantizer
        else:
          quantize_activation(layer_config, activation_bits)

    # We have to do this because of other instances of ReLU.
    elif layer["class_name"] in ["ReLU", "relu", "LeakyReLU"]:

      quantizer = get_config(quantizer_config, layer, "QActivation")
      # This is to avoid unwanted transformations.
      if quantizer is None:
        continue

      if layer["class_name"] == "LeakyReLU":
        negative_slope = layer["config"]["alpha"]
      elif layer["class_name"] == "relu":
        max_value = layer["config"]["max_value"]
        negative_slope = layer["config"]["alpha"]
        threshold = layer["config"]["threshold"]
      else: # ReLU from mobilenet
        max_value = layer["config"]["max_value"]
        negative_slope = layer["config"]["negative_slope"]
        threshold = layer["config"]["threshold"]

      if negative_slope > 0:
        q_name = "leakyrelu"
      else:
        q_name = "relu"

      # If quantizer exists in dictionary related to this name,
      # use it, otherwise, use normal transformations.

      if not isinstance(quantizer, dict) or quantizer.get(q_name, None):
        # Only change activation layer if we will use a quantized activation.

        layer["class_name"] = "QActivation"

        # Remove relu specific configurations
        # remember that quantized relu's are always upper bounded.

        if layer["class_name"] == "LeakyReLU":
          del layer["config"]["alpha"]
        elif layer["class_name"] == "relu":
          del layer["config"]["max_value"]
          del layer["config"]["alpha"]
          del layer["config"]["threshold"]
        else: # ReLU from mobilenet
          del layer["config"]["max_value"]
          del layer["config"]["negative_slope"]
          del layer["config"]["threshold"]

        if isinstance(quantizer, dict):
          quantizer = quantizer[q_name]
        if quantizer:
          layer["config"]["activation"] = quantizer
        else:
          quantize_activation(layer["config"], activation_bits)

    elif layer["class_name"] == "BatchNormalization":
      # We will assume at least QBatchNormalization or
      # layer name is in dictionary to enable conversion
      # otherwise we will just skip it.
      if (
          layer_config["name"] not in quantizer_config and
          "QBatchNormalization" not in quantizer_config
      ):
        continue

      layer["class_name"] = "QBatchNormalization"
      # Needs to add kernel/bias quantizers.
      gamma_quantizer = get_config(
          quantizer_config, layer, "QBatchNormalization",
          "gamma_quantizer")
      beta_quantizer = get_config(
          quantizer_config, layer, "QBatchNormalization",
          "beta_quantizer")
      mean_quantizer = get_config(
          quantizer_config, layer, "QBatchNormalization",
          "mean_quantizer")
      variance_quantizer = get_config(
          quantizer_config, layer, "QBatchNormalization",
          "variance_quantizer")

      layer_config["gamma_quantizer"] = gamma_quantizer
      layer_config["beta_quantizer"] = beta_quantizer
      layer_config["mean_quantizer"] = mean_quantizer
      layer_config["variance_quantizer"] = variance_quantizer

    elif layer["class_name"] in ["AveragePooling2D", "GlobalAveragePooling2D"]:
      q_name = "Q" + layer["class_name"]
      # Adds the average quanizer to config.
      average_quantizer = get_config(
          quantizer_config, layer, q_name, "average_quantizer")

      # This is to avoid unwanted transformations.
      if average_quantizer is None:
        continue

      layer["class_name"] = q_name

      layer_config["average_quantizer"] = average_quantizer

      # Adds activation to config.
      quantizer = get_config(
          quantizer_config, layer, q_name, "activation_quantizer")

      if quantizer:
        layer_config["activation"] = quantizer
      else:
        quantize_activation(layer_config, activation_bits)

  # We need to keep a dictionary of custom objects as our quantized library
  # is not recognized by keras.

  qmodel = quantized_model_from_json(json.dumps(jm), custom_objects)

  # If transfer_weights is true, we load the weights from model to qmodel.

  if transfer_weights and not enable_bn_folding:
    for layer, qlayer in zip(model.layers, qmodel.layers):
      if layer.get_weights():
        qlayer.set_weights(copy.deepcopy(layer.get_weights()))

  return qmodel


def _add_supported_quantized_objects(custom_objects):
  """Map all the quantized objects."""
  custom_objects["QInitializer"] = QInitializer
  custom_objects["QDense"] = QDense
  custom_objects["QConv1D"] = QConv1D
  custom_objects["QConv2D"] = QConv2D
  custom_objects["QConv2DTranspose"] = QConv2DTranspose
  custom_objects["QSimpleRNNCell"] = QSimpleRNNCell
  custom_objects["QSimpleRNN"] = QSimpleRNN
  custom_objects["QLSTMCell"] = QLSTMCell
  custom_objects["QLSTM"] = QLSTM
  custom_objects["QGRUCell"] = QGRUCell
  custom_objects["QGRU"] = QGRU
  custom_objects["QBidirectional"] = QBidirectional
  custom_objects["QDepthwiseConv2D"] = QDepthwiseConv2D
  custom_objects["QSeparableConv1D"] = QSeparableConv1D
  custom_objects["QSeparableConv2D"] = QSeparableConv2D
  custom_objects["QActivation"] = QActivation
  custom_objects["QAdaptiveActivation"] = QAdaptiveActivation
  custom_objects["QBatchNormalization"] = QBatchNormalization
  custom_objects["Clip"] = Clip
  custom_objects["quantized_bits"] = quantized_bits
  custom_objects["bernoulli"] = bernoulli
  custom_objects["stochastic_ternary"] = stochastic_ternary
  custom_objects["ternary"] = ternary
  custom_objects["stochastic_binary"] = stochastic_binary
  custom_objects["binary"] = binary
  custom_objects["quantized_relu"] = quantized_relu
  custom_objects["quantized_ulaw"] = quantized_ulaw
  custom_objects["quantized_tanh"] = quantized_tanh
  custom_objects["quantized_sigmoid"] = quantized_sigmoid
  custom_objects["quantized_po2"] = quantized_po2
  custom_objects["quantized_relu_po2"] = quantized_relu_po2
  

  custom_objects["QConv2DBatchnorm"] = QConv2DBatchnorm
  custom_objects["QDepthwiseConv2DBatchnorm"] = QDepthwiseConv2DBatchnorm

  custom_objects["QAveragePooling2D"] = QAveragePooling2D
  custom_objects["QGlobalAveragePooling2D"] = QGlobalAveragePooling2D


def clone_model(model, custom_objects=None):
  """Clones model with custom_objects."""
  if not custom_objects:
    custom_objects = {}

  # Makes a deep copy to make sure our objects are not shared elsewhere.
  custom_objects = copy.deepcopy(custom_objects)

  _add_supported_quantized_objects(custom_objects)

  json_string = model.to_json()
  qmodel = quantized_model_from_json(json_string, custom_objects=custom_objects)
  qmodel.set_weights(model.get_weights())

  return qmodel

  config = {
      "class_name": model.__class__.__name__,
      "config": model.get_config(),
  }
  clone = tf.keras.models.model_from_config(
      config, custom_objects=custom_objects)
  clone.set_weights(model.get_weights())
  return clone


def quantized_model_from_json(json_string, custom_objects=None):
  if not custom_objects:
    custom_objects = {}

  # Makes a deep copy to make sure our objects are not shared elsewhere.
  custom_objects = copy.deepcopy(custom_objects)

  _add_supported_quantized_objects(custom_objects)

  qmodel = model_from_json(json_string, custom_objects=custom_objects)

  return qmodel


def load_qmodel(filepath, custom_objects=None, compile=True):
  """Loads quantized model from Keras's model.save() h5 file.

  Arguments:
      filepath: one of the following:
          - string, path to the saved model
          - h5py.File or h5py.Group object from which to load the model
          - any file-like object implementing the method `read` that returns
          `bytes` data (e.g. `io.BytesIO`) that represents a valid h5py file
          image.
      custom_objects: Optional dictionary mapping names (strings) to custom
          classes or functions to be considered during deserialization.
      compile: Boolean, whether to compile the model after loading.

  Returns:
      A Keras model instance. If an optimizer was found as part of the saved
      model, the model is already compiled. Otherwise, the model is uncompiled
      and a warning will be displayed. When `compile` is set to False, the
      compilation is omitted without any warning.
  """

  if not custom_objects:
    custom_objects = {}

  # Makes a deep copy to make sure our objects are not shared elsewhere.
  custom_objects = copy.deepcopy(custom_objects)

  _add_supported_quantized_objects(custom_objects)

  qmodel = tf.keras.models.load_model(filepath, custom_objects=custom_objects,
                                      compile=compile)
  return qmodel


def print_model_sparsity(model):
  """Prints sparsity for the pruned layers in the model."""

  def _get_sparsity(weights):
    return 1.0 - np.count_nonzero(weights) / float(weights.size)

  print("Model Sparsity Summary ({})".format(model.name))
  print("--")
  for layer in model.layers:
    if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
      prunable_weights = layer.layer.get_prunable_weights()
    elif isinstance(layer, prunable_layer.PrunableLayer):
      prunable_weights = layer.get_prunable_weights()
    elif prune_registry.PruneRegistry.supports(layer):
      weight_names = prune_registry.PruneRegistry._weight_names(layer)
      prunable_weights = [getattr(layer, weight) for weight in weight_names]
    else:
      prunable_weights = None
    if prunable_weights:
      print("{}: {}".format(
          layer.name, ", ".join([
              "({}, {})".format(weight.name,
                  str(_get_sparsity(K.get_value(weight))))
              for weight in prunable_weights
          ])))
  print("\n")


def get_model_sparsity(model, per_layer=False, allow_list=None):
  """Calculates the sparsity of the model's weights and biases.

  Quantizes the model weights using model_save_quantized_weights (but does not
    save the quantized weights) before calculating the proportion of weights and
    biases set to zero.

  Arguments:
      model: The model to use to calculate sparsity. Assumes that this is a
          QKeras model with trained weights.
      per_layer: If to return a per-layer breakdown of sparsity
      allow_list: A list of layer class names that sparsity will be calculated
        for. If set to None, a default list will be used.

  Returns:
      A float value representing the proportion of weights and biases set to
      zero in the quantized model. If per_layer is True, it also returns a
      per-layer breakdown of model sparsity formatted as a list of tuples in the
      form (<layer name>, <sparsity proportion>)
  """
  # Checks if to use a default list of allowed layers to calculate sparsity.
  if allow_list is None:
    allow_list = [
        "QDense", "Dense", "QConv1D", "Conv1D", "QConv2D", "Conv2D",
        "QDepthwiseConv2D", "DepthwiseConv2D",
        "QSeparableConv1D", "SeparableConv1D",
        "QSeparableConv2D", "SeparableConv2D", "QOctaveConv2D",
        "QSimpleRNN", "RNN", "QLSTM", "QGRU",
        "QConv2DTranspose", "Conv2DTranspose",
        "QConv2DBatchnorm", "QDepthwiseConv2DBatchnorm",
    ]

  # Quantizes the model weights for a more accurate sparsity calculation.
  model_save_quantized_weights(model)

  # Calculates the sparsity layer by layer.
  layer_sparsity = []
  total_sparsity = 0.
  all_weights = []
  for layer in model.layers:
    if hasattr(layer, "quantizers") and layer.__class__.__name__ in allow_list:
      if layer.__class__.__name__ in [
          "QConv2DBatchnorm", "QDepthwiseConv2DBatchnorm"]:
        weights_to_examine = layer.get_folded_weights()
      else:
        weights_to_examine = layer.get_weights()

      layer_weights = []
      for weight in weights_to_examine:
        try:
          weight_numpy = weight.ravel()
        except AttributeError:
          # In case of EagerTensor.
          weight_numpy = weight.numpy().ravel()
        layer_weights.append(weight_numpy)
        all_weights.append(weight_numpy)
      layer_weights = np.concatenate(layer_weights)
      layer_sparsity.append((layer.name, np.mean(layer_weights == 0)))

  if len(all_weights) > 0:
    # Average the sparsity for the entire model.
    all_weights = np.concatenate(all_weights)
    total_sparsity = np.mean(all_weights == 0)
  if per_layer:
    return (total_sparsity, layer_sparsity)
  else:
    return total_sparsity


def quantized_model_debug(model, X_test, plot=False, plt_instance=None):
  """Debugs and plots model weights and activations.

  Args:
    model: The QKeras model to debug
    X_test: The sample data to use to give to model.predict
    plot: Bool. If to plot the results.
    plt_instance: A matplotlib.pyplot instance used to plot in an IPython
      environment.
  """
  assert (plt_instance and plot) or not plot, (
      "plt_instance is required if plt is True")

  outputs = []
  output_names = []

  for layer in model.layers:
    if layer.__class__.__name__ in REGISTERED_LAYERS:
      output_names.append(layer.name)
      outputs.append(layer.output)

  model_debug = Model(inputs=model.inputs, outputs=outputs)

  y_pred = model_debug.predict(X_test)

  print("{:30} {: 8.4f} {: 8.4f}".format(
      "input", np.min(X_test), np.max(X_test)))

  for n, p in zip(output_names, y_pred):
    layer = model.get_layer(n)
    if (layer.__class__.__name__ in "QActivation" or
        layer.__class__.__name__ in "QAdaptiveActivation"):
      alpha = get_weight_scale(layer.activation, p)
    else:
      alpha = 1.0
    print(
        "{:30} {: 8.4f} {: 8.4f}".format(n, np.min(p / alpha),
                                         np.max(p / alpha)),
        end="")
    if alpha != 1.0:
      print(" a[{: 8.4f} {:8.4f}]".format(np.min(alpha), np.max(alpha)))
    if plot and layer.__class__.__name__ in [
        "QConv1D", "QConv2D", "QConv2DTranspose", "QDense", "QActivation",
        "QAdaptiveActivation", "QSimpleRNN", "QLSTM", "QGRU", "QBidirectional",
        "QSeparableConv1D", "QSeparableConv2D"
    ]:
      plt_instance.hist(p.flatten(), bins=25)
      plt_instance.title(layer.name + "(output)")
      plt_instance.show()
    alpha = None

    if layer.__class__.__name__ not in [
        "QConv2DBatchnorm", "QDepthwiseConv2DBatchnorm"]:
      weights_to_examine = layer.get_weights()
    else:
      weights_to_examine = layer.get_folded_weights()

    for i, weights in enumerate(weights_to_examine):
      if hasattr(layer, "get_quantizers") and layer.get_quantizers()[i]:
        weights = K.eval(layer.get_quantizers()[i](K.constant(weights)))
        if i == 0 and layer.__class__.__name__ in [
            "QConv1D", "QConv2D", "QConv2DTranspose", "QDense",
            "QSimpleRNN", "QLSTM", "QGRU",
            "QSeparableConv1D", "QSeparableConv2D",
            "QConv2DBatchnorm", "QDepthwiseConv2DBatchnorm"
        ]:
          alpha = get_weight_scale(layer.get_quantizers()[i], weights)
          # if alpha is 0, let's remove all weights.
          alpha_mask = (alpha == 0.0)
          weights = np.where(alpha_mask, weights * alpha, weights / alpha)
          if plot:
            plt_instance.hist(weights.flatten(), bins=25)
            plt_instance.title(layer.name + "(weights)")
            plt_instance.show()
      print(" ({: 8.4f} {: 8.4f})".format(np.min(weights), np.max(weights)),
            end="")
    if alpha is not None and isinstance(alpha, np.ndarray):
      print(" a({: 10.6f} {: 10.6f})".format(
          np.min(alpha), np.max(alpha)), end="")
    print("")


def quantized_model_dump(model,
                         x_test,
                         output_dir=None,
                         layers_to_dump=[]):
  """Dumps tensors of target layers to binary files.

  Arguments:
    model: QKeras model object.
    x_test: numpy type, test tensors to generate output tensors.
    output_dir: a string for the directory to hold binary data.
    layers_to_dump: a list of string, specified layers by layer
      customized name.
  """
  outputs = []
  y_names = []

  if not output_dir:
    with tempfile.TemporaryDirectory() as output_dir:
      print("temp dir", output_dir)

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print("create dir", output_dir)

  for layer in model.layers:
    if not layers_to_dump or layer.name in layers_to_dump:
      y_names.append(layer.name)
      outputs.append(layer.output)

  # Gather the tensor outputs from specified layers at layers_to_dump.
  model_debug = Model(inputs=model.inputs, outputs=outputs)
  y_pred = model_debug.predict(x_test)

  # Dumps tensors to files.
  for name, tensor_data in zip(y_names, y_pred):
    filename = os.path.join(output_dir, name + ".bin")
    print("writing the layer output tensor to ", filename)
    with open(filename, "w") as fid:
      tensor_data.astype(np.float32).tofile(fid)
