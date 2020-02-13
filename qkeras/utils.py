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
import six

import tensorflow as tf

from tensorflow.keras import initializers
from tensorflow.keras.models import model_from_json

import numpy as np

from .qlayers import QActivation
from .qlayers import QDense
from .qlayers import Clip
from .qconvolutional import QConv1D
from .qconvolutional import QConv2D
from .qconvolutional import QDepthwiseConv2D
from .qpooling import QAveragePooling2D
from .quantizers import quantized_bits
from .quantizers import bernoulli
from .quantizers import stochastic_ternary
from .quantizers import ternary
from .quantizers import stochastic_binary
from .quantizers import binary
from .quantizers import quantized_relu
from .quantizers import quantized_ulaw
from .quantizers import quantized_tanh
from .quantizers import quantized_po2
from .quantizers import quantized_relu_po2
from .qnormalization import QBatchNormalization

from .safe_eval import safe_eval

#
# Model utilities: before saving the weights, we want to apply the quantizers
#

def model_save_quantized_weights(model, filename=None):
  """Quantizes model for inference and save it.

  Takes a model with weights, apply quantization function to weights and
  returns a dictionarty with quantized weights.

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


def quantize_activation(layer_config, custom_objects, activation_bits):
  """Replaces activation by quantized activation functions."""

  str_act_bits = str(activation_bits)

  # relu -> quantized_relu(bits)
  # tanh -> quantized_tanh(bits)
  #
  # more to come later

  if layer_config["activation"] == "relu":
    layer_config["activation"] = "quantized_relu(" + str_act_bits + ")"
    custom_objects["quantized_relu(" + str_act_bits + ")"] = (
        quantized_relu(activation_bits))

  elif layer_config["activation"] == "tanh":
    layer_config["activation"] = "quantized_tanh(" + str_act_bits + ")"
    custom_objects["quantized_tanh(" + str_act_bits + ")"] = (
        quantized_tanh(activation_bits))


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

  custom_objects["QDense"] = QDense
  custom_objects["QConv1D"] = QConv1D
  custom_objects["QConv2D"] = QConv2D
  custom_objects["QDepthwiseConv2D"] = QDepthwiseConv2D
  custom_objects["QAveragePooling2D"] = QAveragePooling2D
  custom_objects["QActivation"] = QActivation

  # just add all the objects from quantizer_config to
  # custom_objects first.

  for layer_name in quantizer_config.keys():

    if isinstance(quantizer_config[layer_name], six.string_types):
      name = quantizer_config[layer_name]
      custom_objects[name] = safe_eval(name, globals())

    else:
      for name in quantizer_config[layer_name].keys():
        custom_objects[quantizer_config[layer_name][name]] = (
            safe_eval(quantizer_config[layer_name][name], globals()))

  for layer in layers:
    layer_config = layer["config"]

    # Dense becomes QDense
    # Activation converts activation functions

    if layer["class_name"] == "Dense":
      layer["class_name"] = "QDense"

      # needs to add kernel/bias quantizers

      kernel_quantizer = quantizer_config.get(
          layer["name"], quantizer_config.get("QDense",
                                              None))["kernel_quantizer"]

      bias_quantizer = quantizer_config.get(
          layer["name"], quantizer_config.get("QDense", None))["bias_quantizer"]

      layer_config["kernel_quantizer"] = kernel_quantizer
      layer_config["bias_quantizer"] = bias_quantizer

      # if activation is present, add activation here

      quantizer = quantizer_config.get(layer["name"],
                                       quantizer_config.get(
                                           "QDense", None)).get(
                                               "activation_quantizer", None)

      if quantizer:
        layer_config["activation"] = quantizer
        custom_objects[quantizer] = safe_eval(quantizer, globals())
      else:
        quantize_activation(layer_config, custom_objects, activation_bits)

    elif layer["class_name"] == "Conv2D":
      layer["class_name"] = "QConv2D"

      # needs to add kernel/bias quantizers

      kernel_quantizer = quantizer_config.get(
          layer["name"], quantizer_config.get("QConv2D", None)).get(
              "kernel_quantizer", None)

      bias_quantizer = quantizer_config.get(
          layer["name"], quantizer_config.get("QConv2D", None)).get(
              "bias_quantizer", None)

      layer_config["kernel_quantizer"] = kernel_quantizer
      layer_config["bias_quantizer"] = bias_quantizer

      # if activation is present, add activation here

      quantizer = quantizer_config.get(layer["name"],
                                       quantizer_config.get(
                                           "QConv2D", None)).get(
                                               "activation_quantizer", None)

      if quantizer:
        layer_config["activation"] = quantizer
        custom_objects[quantizer] = safe_eval(quantizer, globals())
      else:
        quantize_activation(layer_config, custom_objects, activation_bits)

    elif layer["class_name"] == "DepthwiseConv2D":
      layer["class_name"] = "QDepthwiseConv2D"

      # needs to add kernel/bias quantizers

      depthwise_quantizer = quantizer_config.get(
          layer["name"], quantizer_config.get("QDepthwiseConv2D", None)).get(
              "depthwise_quantizer", None)

      bias_quantizer = quantizer_config.get(
          layer["name"], quantizer_config.get("QDepthwiseConv2D", None)).get(
              "bias_quantizer", None)

      layer_config["depthwise_quantizer"] = depthwise_quantizer
      layer_config["bias_quantizer"] = bias_quantizer

      # if activation is present, add activation here

      quantizer = quantizer_config.get(
          layer["name"], quantizer_config.get("QDepthwiseConv2D", None)).get(
              "activation_quantizer", None)

      if quantizer:
        layer_config["activation"] = quantizer
        custom_objects[quantizer] = safe_eval(quantizer, globals())
      else:
        quantize_activation(layer_config, custom_objects, activation_bits)

    elif layer["class_name"] == "Activation":
      quantizer = quantizer_config.get(
          layer["name"], quantizer_config.get("QActivation", None))

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
          custom_objects[quantizer] = safe_eval(quantizer, globals())
        else:
          quantize_activation(layer_config, custom_objects, activation_bits)

    elif layer["class_name"] == "AveragePooling2D":
      layer["class_name"] = "QAveragePooling2D"

      quantizer = quantizer_config.get(layer["name"], None)

      # if quantizer exists in dictionary related to this name,
      # use it, otherwise, use normal transformations

      if quantizer:
        layer_config["activation"] = quantizer
        custom_objects[quantizer] = safe_eval(quantizer, globals())
      else:
        quantize_activation(layer_config, custom_objects, activation_bits)

  # we need to keep a dictionary of custom objects as our quantized library
  # is not recognized by keras.

  qmodel = model_from_json(json.dumps(jm), custom_objects=custom_objects)

  # if transfer_weights is true, we load the weights from model to qmodel

  if transfer_weights:
    for layer, qlayer in zip(model.layers, qmodel.layers):
      if layer.get_weights():
        qlayer.set_weights(copy.deepcopy(layer.get_weights()))

  return qmodel, custom_objects

def _add_supported_quantized_objects(custom_objects):

  # Map all the quantized objects
  custom_objects["QDense"] = QDense
  custom_objects["QConv1D"] = QConv1D
  custom_objects["QConv2D"] = QConv2D
  custom_objects["QDepthwiseConv2D"] = QDepthwiseConv2D
  custom_objects["QAveragePooling2D"] = QAveragePooling2D
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


def quantized_model_from_json(json_string, custom_objects=None):
  if not custom_objects:
    custom_objects = {}

  # let's make a deep copy to make sure our objects are not shared elsewhere
  custom_objects = copy.deepcopy(custom_objects)

  _add_supported_quantized_objects(custom_objects)

  qmodel = model_from_json(json_string, custom_objects=custom_objects)

  return qmodel

def load_qmodel(filepath, custom_objects=None, compile=True):
  """
  Load quantized model from Keras's model.save() h5 file.

  # Arguments:
        filepath: one of the following:
                  - string, path to the saved model
                  - h5py.File or h5py.Group object from which to load the model
                  - any file-like object implementing the method `read` that returns
                  `bytes` data (e.g. `io.BytesIO`) that represents a valid h5py file image.
        custom_objects: Optional dictionary mapping names
                  (strings) to custom classes or functions to be
                  considered during deserialization.
        compile: Boolean, whether to compile the model
                  after loading.
  
  # Returns
        A Keras model instance. If an optimizer was found
        as part of the saved model, the model is already
        compiled. Otherwise, the model is uncompiled and
        a warning will be displayed. When `compile` is set
        to False, the compilation is omitted without any
        warning.
  """

  if not custom_objects:
    custom_objects = {}

  # let's make a deep copy to make sure our objects are not shared elsewhere
  custom_objects = copy.deepcopy(custom_objects)
    
  _add_supported_quantized_objects(custom_objects)
    
  qmodel = tf.keras.models.load_model(filepath, custom_objects=custom_objects, compile=compile)
    
  return qmodel
