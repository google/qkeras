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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import networkx as nx
import tensorflow as tf
from six.moves import range
from tensorflow.keras.models import clone_model
from tensorflow.keras.models import Model
from tensorflow.keras import Input

from .qconvolutional import QConv2D
from .qconvolutional import QDepthwiseConv2D
from .qtools import generate_layer_data_type_map as gen_map
from .qtools import qgraph


def convert_folded_layer_to_unfolded(layer):
  """Replace a source batchnorm folded layer with a non-folded layer.

  Args:
    layer: keras/qkeras layer type. Source layer to be replaced with

  Returns:
    new layer instance
  """

  # get layer config from the composite layer
  config = layer.get_config()
  # set layer config for QConv2D layer by first creating a tmp
  # QConv2D object and generate template for its config
  if layer.__class__.__name__ == "QConv2DBatchnorm":
    new_layer = QConv2D(filters=1, kernel_size=(2, 2), use_bias=True)
  elif layer.__class__.__name__ == "QDepthwiseConv2DBatchnorm":
    new_layer = QDepthwiseConv2D(kernel_size=(2, 2), use_bias=True)
  else:
    # TODO(lishanok): will extend to QDense in the future
    assert ValueError, "%s is not supported!" % layer.__class__.__name__

  new_layer_cfg = new_layer.get_config()

  # set qconv2d config according to the values in the composite layer
  for (key, _) in new_layer_cfg.items():
    if key in config.keys():
      new_layer_cfg[key] = config[key]

  # in case use_bias is False in the composite layer,
  #  we need to set it True because we have folded bias
  new_layer_cfg["use_bias"] = True

  # create a non-folded, e.g., qconv2d layer from config and replace
  # old layer with it
  if layer.__class__.__name__ == "QConv2DBatchnorm":
    new_layer = QConv2D.from_config(new_layer_cfg)
  elif layer.__class__.__name__ == "QDepthwiseConv2DBatchnorm":
    new_layer = QDepthwiseConv2D.from_config(new_layer_cfg)
  else:
    raise ValueError("Unsupported layer conversion {}".format(layer.name))

  return new_layer


def unfold_model(model):
  """Convert a model with batchnorm folded layer to a normal model.

  "Normal" here refers to a model without composite folded layer such as
  QConv2DBatchnorm layer.
  This function replace the folded layers with a normal QConv/QDense
  layer. It aslo sets the weights in the normal layer with the folded weights
  in the folded layer. Model architecture could be either sequential or
  non-sequential.

  Arguments:
    model: keras object, model with folded layers.

  Returns:
    A model that replaces folded layers (e.g., QConv2DBatchnorm) with normal
      qkeras layers (e.g., QConv2D). This model can be passed on to hardware
      generator so that hardware doesn't see batch normalization
      parameters.
  """

  def _convert_folded_layer(layer):
    if layer.__class__.__name__ in [
        "QConv2DBatchnorm", "QDepthwiseConv2DBatchnorm"]:
      new_layer = convert_folded_layer_to_unfolded(layer)
    else:
      new_layer = layer.__class__.from_config(layer.get_config())

    new_layer.build(layer.input_shape)
    return new_layer

  def _clone_weights(src_layer, new_layer):
    if (src_layer.__class__.__name__ == "QConv2DBatchnorm") and (
        new_layer.__class__.__name__ == "QConv2D"):
      src_weights = src_layer.get_folded_weights()
      # transfer weights from folded layer to the target layer
      folded_kernel_quantized = (
          src_weights[0].numpy())
      folded_bias_quantized = (
          src_weights[1].numpy())
      new_layer.set_weights([folded_kernel_quantized, folded_bias_quantized])

    elif (src_layer.__class__.__name__ == "QDepthwiseConv2DBatchnorm") and (
        new_layer.__class__.__name__ == "QDepthwiseConv2D"):
      # transfer weights from folded layer to the target layer
      src_weights = src_layer.get_folded_weights()
      folded_depthwise_kernel_quantized = src_weights[0].numpy()
      folded_bias_quantized = src_weights[1].numpy()
      new_layer.set_weights(
          [folded_depthwise_kernel_quantized, folded_bias_quantized])
    else:
      new_layer.set_weights(src_layer.get_weights())

  inp = Input(shape=model.input_shape[1:])
  cloned_model = clone_model(
      model, input_tensors=inp, clone_function=_convert_folded_layer)

  # replace weights
  for (src_layer, new_layer) in zip(model.layers, cloned_model.layers):
    _clone_weights(src_layer, new_layer)

  return cloned_model


def populate_bias_quantizer_from_accumulator(model, source_quantizers):
  """Populate the bias quantizer from accumulator type.

  When user set bias_quantizer=None for layers(e.g.,
  QConv2DBatchnorm), this function generates the accumulator type of
  the layer MAC op and set it as the bias quantizer.
  Such step is skipped if user provided a specific bias quantizer type.

  Args:
    model: keras/qkeras model object. If the model doesn't contain any batchnorm
      folded layer or if the bias quanizer type in the folded layer is already
      given, no operation needed. Else we generate the bias quantizer type and
      set it in model.

    source_quantizers: list of qkeras quantizers. A list of quantizer types
      for model inputs.

  Returns:
    keras model object
  """
  default_quantizer = "quantized_bits(8, 0, 1)"

  # if source_quantizers is None, CreateGraph will use default_quantizer
  (graph, source_quantizer_list) = qgraph.CreateGraph(
      model, source_quantizers, default_quantizer)
  qgraph.GraphPropagateActivationsToEdges(graph)

  # generate the quantizer types of each layer. For folded layers, if bias
  # quantizer is not given by user, this function will generate the accumulator
  # type and set it as the bias quantizer type.
  is_inference = False
  keras_quantizer = "quantized_bits(8, 0, 1)"
  keras_accumulator = "quantized_bits(8, 0, 1)"
  for_reference = False
  layer_map = gen_map.generate_layer_data_type_map(
      graph, source_quantizer_list, is_inference,
      keras_quantizer, keras_accumulator, for_reference)

  for layer in model.layers:
    # TODO(lishanok): extend to other layer types if necessary
    if layer.__class__.__name__ in [
        "QConv2DBatchnorm", "QDepthwiseConv2DBatchnorm"]:
      if not layer.bias_quantizer:
        # if user didn't specify the bias quantizer, we set it as the
        # MAC accumulator type of the current layer's MAC operation
        qtools_bias_quantizer = layer_map["layer_data_type_map"][
            layer].bias_quantizer

        if tf.is_tensor(qtools_bias_quantizer.int_bits):
          qtools_bias_quantizer.int_bits = (
              qtools_bias_quantizer.int_bits.numpy())

        layer.bias_quantizer = (
            qtools_bias_quantizer.convert_to_qkeras_quantizer())

        layer.bias_quantizer_internal = layer.bias_quantizer
        if layer.__class__.__name__ == "QConv2DBatchnorm":
          layer.quantizers = [layer.kernel_quantizer_internal,
                              layer.bias_quantizer_internal]
        elif layer.__class__.__name__ == "QDepthwiseConv2DBatchnorm":
          layer.quantizers = [layer.depthwise_quantizer_internal,
                              layer.bias_quantizer_internal]
  return model
