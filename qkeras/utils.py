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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import json
import tempfile
import types

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import os
import six
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras import initializers

from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json

from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from tensorflow_model_optimization.python.core.sparsity.keras import prune_registry
from tensorflow_model_optimization.python.core.sparsity.keras import prunable_layer

from .qlayers import Clip
from .qlayers import QActivation
from .qpooling import QAveragePooling2D
from .qlayers import QDense
from .qlayers import QInitializer
from .qconvolutional import QConv1D
from .qconvolutional import QConv2D
from .qrecurrent import QSimpleRNN
from .qrecurrent import QSimpleRNNCell
from .qrecurrent import QLSTM
from .qrecurrent import QLSTMCell
from .qrecurrent import QGRU
from .qrecurrent import QGRUCell
from .qrecurrent import QBidirectional
from .qconvolutional import QDepthwiseConv2D
from .qnormalization import QBatchNormalization
from .quantizers import binary
from .quantizers import bernoulli
from .quantizers import get_weight_scale
from .quantizers import quantized_bits
from .quantizers import quantized_relu
from .quantizers import quantized_ulaw
from .quantizers import quantized_tanh
from .quantizers import quantized_po2
from .quantizers import quantized_relu_po2
from .quantizers import stochastic_binary
from .quantizers import stochastic_ternary
from .quantizers import ternary
from .safe_eval import safe_eval


REGISTERED_LAYERS = [
    "QActivation", "Activation",
    "QDense", "QConv1D", "QConv2D", "QDepthwiseConv2D",
    "QSimpleRNN", "QLSTM", "QGRU", "QBidirectional",
    "QBatchNormalization"
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
      for quantizer, weight in zip(layer.get_quantizers(), layer.get_weights()):
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
          # quantized_relu_po2 does not have a sign
          if isinstance(quantizer, quantized_po2):
            has_sign = True
          sign = np.sign(weight)
          # make sure values are -1 or +1 only
          sign += (1.0 - np.abs(sign))
          weight = np.round(np.log2(np.abs(weight)))
          signs.append(sign)
        else:
          signs.append([])

      saved_weights[layer.name] = {"weights": weights}
      if has_sign:
        saved_weights[layer.name]["signs"] = signs

      layer.set_weights(weights)
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
  #
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


def get_config(quantizer_config, layer, layer_class, parameter=None):
  """Returns search of quantizer on quantizer_config."""
  quantizer = quantizer_config.get(layer["config"]["name"],
                                   quantizer_config.get(layer_class, None))

  if quantizer is not None and parameter is not None:
    quantizer = quantizer.get(parameter, None)

  return quantizer


def model_quantize(model,
                   quantizer_config,
                   activation_bits,
                   custom_objects=None,
                   transfer_weights=False):
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
    activation_bits: number of bits for quantized_relu, quantized_tanh
    custom_objects: dictionary following keras recommendations for json
      translation.
    transfer_weights: if true, weights are to be transfered from model to
      qmodel.

  Returns:
    qmodel with quantized operations and custom_objects.
  """

  if not custom_objects:
    custom_objects = {}

  # let's make a deep copy to make sure our objects are not shared elsewhere
  jm = copy.deepcopy(json.loads(model.to_json()))
  custom_objects = copy.deepcopy(custom_objects)
  config = jm["config"]
  layers = config["layers"]

  def quantize_rnn(layer):
    q_name = "Q" + layer["class_name"]
    # needs to add kernel, recurrent bias quantizers
    kernel_quantizer = get_config(
        quantizer_config, layer, q_name, "kernel_quantizer")
    recurrent_quantizer = get_config(
        quantizer_config, layer, q_name, "recurrent_quantizer")
    bias_quantizer = get_config(
        quantizer_config, layer, q_name, "bias_quantizer")

    # this is to avoid unwanted transformations
    if kernel_quantizer is None:
      return

    layer["class_name"] = q_name

    layer_config["kernel_quantizer"] = kernel_quantizer
    layer_config["recurrent_quantizer"] = recurrent_quantizer
    layer_config["bias_quantizer"] = bias_quantizer

    # if activation is present, add activation here
    activation = get_config(
        quantizer_config, layer, q_name, "activation_quantizer")
    if activation:
      layer_config["activation"] = activation
    else:
      quantize_activation(layer_config, activation_bits)

    # if recurrent activation is present, add activation here
    if layer["class_name"] in ["LSTM", "GRU"]:
      recurrent_activation = get_config(
          quantizer_config, layer, q_name, "recurrent_activation_quantizer")
      if recurrent_activation:
        layer_config["recurrent_activation"] = recurrent_activation

  for layer in layers:
    layer_config = layer["config"]

    # Dense becomes QDense
    # Activation converts activation functions

    if layer["class_name"] == "Dense":
      # needs to add kernel/bias quantizers
      kernel_quantizer = get_config(
          quantizer_config, layer, "QDense", "kernel_quantizer")
      bias_quantizer = get_config(
          quantizer_config, layer, "QDense", "bias_quantizer")

      # this is to avoid unwanted transformations
      if kernel_quantizer is None:
        continue

      layer["class_name"] = "QDense"

      layer_config["kernel_quantizer"] = kernel_quantizer
      layer_config["bias_quantizer"] = bias_quantizer

      # if activation is present, add activation here
      quantizer = get_config(
          quantizer_config, layer, "QDense", "activation_quantizer")

      if quantizer:
        layer_config["activation"] = quantizer
        custom_objects[quantizer] = safe_eval(quantizer, globals())
      else:
        quantize_activation(layer_config, activation_bits)

    elif layer["class_name"] in ["Conv1D", "Conv2D"]:
      q_name = "Q" + layer["class_name"]
      # needs to add kernel/bias quantizers
      kernel_quantizer = get_config(
          quantizer_config, layer, q_name, "kernel_quantizer")

      bias_quantizer = get_config(
          quantizer_config, layer, q_name, "bias_quantizer")

      # this is to avoid unwanted transformations
      if kernel_quantizer is None:
        continue

      layer["class_name"] = q_name

      layer_config["kernel_quantizer"] = kernel_quantizer
      layer_config["bias_quantizer"] = bias_quantizer

      # if activation is present, add activation here
      quantizer = get_config(
          quantizer_config, layer, q_name, "activation_quantizer")

      if quantizer:
        layer_config["activation"] = quantizer
      else:
        quantize_activation(layer_config, activation_bits)

    elif layer["class_name"] == "DepthwiseConv2D":
      # needs to add kernel/bias quantizers
      depthwise_quantizer = get_config(quantizer_config, layer,
          "QDepthwiseConv2D", "depthwise_quantizer")
      bias_quantizer = get_config(quantizer_config, layer,
          "QDepthwiseConv2D", "bias_quantizer")

      # this is to avoid unwanted transformations
      if depthwise_quantizer is None:
        continue

      layer["class_name"] = "QDepthwiseConv2D"

      layer_config["depthwise_quantizer"] = depthwise_quantizer
      layer_config["bias_quantizer"] = bias_quantizer
      # if activation is present, add activation here
      quantizer = get_config(quantizer_config, layer,
          "QDepthwiseConv2D", "activation_quantizer",)

      if quantizer:
        layer_config["activation"] = quantizer
      else:
        quantize_activation(layer_config, activation_bits)

    elif layer["class_name"] in ["SimpleRNN", "LSTM", "GRU"]:
      quantize_rnn(layer)

    elif layer['class_name'] == 'QBidirectional':
      quantize_rnn(layer['layer'])
      if "backward_layer" in layer:
        quantize_rnn(layer['backward_layer'])

    elif layer["class_name"] == "Activation":
      quantizer = get_config(quantizer_config, layer, "QActivation")
      # this is to avoid softmax from quantizing in autoq
      if quantizer is None:
        continue

      # if quantizer exists in dictionary related to this name,
      # use it, otherwise, use normal transformations

      if not isinstance(quantizer, dict) or quantizer.get(
          layer_config["activation"], None):
        # only change activation layer if we will use a quantized activation

        layer["class_name"] = "QActivation"
        if isinstance(quantizer, dict):
          quantizer = quantizer[layer_config["activation"]]
        if quantizer:
          layer_config["activation"] = quantizer
        else:
          quantize_activation(layer_config, activation_bits)

    # we have to do this because of other instances of ReLU
    elif layer["class_name"] in ["ReLU", "relu", "LeakyReLU"]:

      quantizer = get_config(quantizer_config, layer, "QActivation")
      # this is to avoid unwanted transformations
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

      # if quantizer exists in dictionary related to this name,
      # use it, otherwise, use normal transformations

      if not isinstance(quantizer, dict) or quantizer.get(q_name, None):
        # only change activation layer if we will use a quantized activation

        layer["class_name"] = "QActivation"

        # remove relu specific configurations
        # remember that quantized relu's are always upper bounded

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
      # we will assume at least QBatchNormalization or
      # layer name is in dictionary to enable conversion
      # otherwise we will just skip it.
      if (
          layer_config["name"] not in quantizer_config and
          "QBatchNormalization" not in quantizer_config
      ):
        continue

      layer["class_name"] = "QBatchNormalization"
      # needs to add kernel/bias quantizers
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

  # we need to keep a dictionary of custom objects as our quantized library
  # is not recognized by keras.

  qmodel = quantized_model_from_json(json.dumps(jm), custom_objects)

  # if transfer_weights is true, we load the weights from model to qmodel

  if transfer_weights:
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
  custom_objects["QSimpleRNNCell"] = QSimpleRNNCell
  custom_objects["QSimpleRNN"] = QSimpleRNN
  custom_objects["QLSTMCell"] = QLSTMCell
  custom_objects["QLSTM"] = QLSTM
  custom_objects["QGRUCell"] = QGRUCell
  custom_objects["QGRU"] = QGRU
  custom_objects["QBidirectional"] = QBidirectional
  custom_objects["QDepthwiseConv2D"] = QDepthwiseConv2D
  custom_objects["QActivation"] = QActivation
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
  custom_objects["quantized_po2"] = quantized_po2
  custom_objects["quantized_relu_po2"] = quantized_relu_po2


def clone_model(model, custom_objects=None):
  """Clones model with custom_objects."""
  if not custom_objects:
    custom_objects = {}

  # let's make a deep copy to make sure our objects are not shared elsewhere
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

  # let's make a deep copy to make sure our objects are not shared elsewhere
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

  # let's make a deep copy to make sure our objects are not shared elsewhere
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

def quantized_model_debug(model, X_test, plot=False):
  """Debugs and plots model weights and activations."""
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
    if layer.__class__.__name__ == "QActivation":
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
      "QConv1D", "QConv2D", "QDense", "QActivation", 
      "QSimpleRNN", "QLSTM", "QGRU", "QBidirectional"
    ]:
      plt.hist(p.flatten(), bins=25)
      plt.title(layer.name + "(output)")
      plt.show()
    alpha = None
    for i, weights in enumerate(layer.get_weights()):
      if hasattr(layer, "get_quantizers") and layer.get_quantizers()[i]:
        weights = K.eval(layer.get_quantizers()[i](K.constant(weights)))
        if i == 0 and layer.__class__.__name__ in [
            "QConv1D", "QConv2D", "QDense", "QSimpleRNN", "QLSTM", "QGRU"
        ]:
          alpha = get_weight_scale(layer.get_quantizers()[i], weights)
          # if alpha is 0, let's remove all weights.
          alpha_mask = (alpha == 0.0)
          weights = np.where(alpha_mask, weights * alpha, weights / alpha)
          if plot:
            plt.hist(weights.flatten(), bins=25)
            plt.title(layer.name + "(weights)")
            plt.show()
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
    if layer.__class__.__name__ in ["InputLayer"] + REGISTERED_LAYERS:
      if not layers_to_dump or layer.name in layers_to_dump:
        y_names.append(layer.name)
        outputs.append(layer.output)

  # Gather the tensor outputs from specified layers at layers_to_dump
  model_debug = Model(inputs=model.inputs, outputs=outputs)
  y_pred = model_debug.predict(x_test)

  # dump tensors to files
  for name, tensor_data in zip(y_names, y_pred):
    filename = os.path.join(output_dir, name + ".bin")
    print("writing the layer output tensor to ", filename)
    with open(filename, "w") as fid:
      tensor_data.astype(np.float32).tofile(fid)
