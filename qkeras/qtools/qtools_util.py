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
"""utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import sys
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
from qkeras.qtools import quantized_operators


def get_val(feature, key):
  # Return feature[key] or feature.key
  if isinstance(feature, dict):
    val = feature.get(key, None)
  else:
    val = getattr(feature, key, None)
  return val


def is_shape_alternation_layers(layer):
  lname = layer.__class__.__name__
  if lname:
    return "MaxPool" in lname or "Reshape" in lname or "Flatten" in lname
  return False


def is_merge_layers(layer):

  if layer.__class__.__name__ in [
      "Add", "Multiply", "Subtract", "Average", "Maximum", "Minimum",
      "Concatenate", "Dot"]:
    return True
  else:
    return False


def get_input_quantizers(graph, node_id, quantizer_factory, debug=False):
  """get the current layer's input quantizer."""

  # in merge layers, therea are more than 1 input

  output = []
  for parent_node_id in graph.predecessors(node_id):

    edge = graph.edges[(parent_node_id, node_id)]

    if debug:
      print("parent_node_id:", parent_node_id)
      print(edge)

    quantizer_on_edge = edge["quantizer"]
    input_quantizer = quantizer_factory.make_quantizer(quantizer_on_edge)

    output.append((input_quantizer, edge))

  return output


def get_input_quantizers_advanced(graph, node_id,
                                  is_input_layer, quantizer_factory,
                                  cfg, debug=False):
  """get input quantizer, deal with keras layer or lack of input quantizer in qkeras layer."""

  # in merge layers, therea are more than 1 input
  default_source_quantizer = cfg.default_source_quantizer
  default_interm_quantizer = cfg.default_interm_quantizer

  output = []
  for parent_node_id in graph.predecessors(node_id):

    edge = graph.edges[(parent_node_id, node_id)]

    if debug:
      print("parent_node_id:", parent_node_id)
      print(edge)

    quantizer_on_edge = edge["quantizer"]
    input_quantizer = quantizer_factory.make_quantizer(quantizer_on_edge)

    if is_input_layer and not input_quantizer:
      # input layer without input_quantizer specified
      #   ->use default_source_quantizer
      input_quantizer = quantizer_factory.make_default_quantizer(
          mode=default_source_quantizer)
    elif not input_quantizer:
      # if no input quantizer is available
      #   -> use default quantizer from config.json
      input_quantizer = quantizer_factory.make_default_quantizer(
          mode=default_interm_quantizer)

    output.append((input_quantizer, edge))

  return output


def get_operation_count(layer, input_shape):
  """Determines number of multiplier operations in a qkeras layer."""

  # Check if the inputs are a list of Dimensions
  if isinstance(input_shape, list):
    input_shape = input_shape[0]

  operation_count = 0

  if is_merge_layers(layer) or is_shape_alternation_layers(layer):
    operation_count = np.prod(input_shape[1:])

  elif layer.__class__.__name__ in [
      "AveragePooling2D", "AvgPool2D", "GlobalAvgPool2D",
      "GlobalAveragePooling2D", "QGlobalAveragePooling2D"
  ]:

    if hasattr(layer, "pool_size"):
      pool_size = layer.pool_size
    else:
      pool_size = input_shape[1:-1]
    add_ops = np.prod(pool_size)

    output_shape = layer.compute_output_shape(input_shape)
    channels_o = output_shape[-1]

    # total number of add ops
    operation_count = channels_o * add_ops

  elif "UpSampling" in layer.__class__.__name__:
    # UpSampling1D/2D/3D
    output_shape = layer.compute_output_shape(input_shape)
    operation_count = np.prod(output_shape[1:])

  elif ("Activation" in layer.__class__.__name__ or
        "BatchNormalization" in layer.__class__.__name__):
    operation_count = np.prod(input_shape[1:])

  elif layer.__class__.__name__ in ["QConv2D", "Conv2D", "QConv2DBatchnorm"]:

    output_shape = layer.compute_output_shape(input_shape)
    _, _, _, channels_i = input_shape

    _, height_o, width_o, channels_o = output_shape

    weight = layer.get_weights()[0]

    kernel_h, kernel_w, _, _ = weight.shape

    operation_count = (
        height_o * width_o * channels_o * kernel_h * kernel_w * channels_i)

  elif layer.__class__.__name__ in ["QConv1D", "Conv1D"]:
    output_shape = layer.compute_output_shape(input_shape)
    _, _, channels_i = input_shape

    _, time_o, channels_o = output_shape

    weight = layer.get_weights()[0]

    kernel_length, _, _ = weight.shape

    operation_count = (
        time_o * channels_o * kernel_length * channels_i)

  elif layer.__class__.__name__ in ["QDepthwiseConv2D", "DepthwiseConv2D"]:
    output_shape = layer.compute_output_shape(input_shape)
    _, _, _, channels_i = input_shape

    _, height_o, width_o, channels_o = output_shape

    weight_1 = layer.get_weights()[0]

    kernel_h, kernel_w, _, _ = weight_1.shape

    operation_count = (
        kernel_h * kernel_w * height_o * width_o * channels_i)

  elif layer.__class__.__name__ in ["QDense", "Dense"]:
    output_shape = layer.compute_output_shape(input_shape)
    # Find the input and output shapes out of all possible dimensions.
    # Usually, the first shape dimension will be the batch size, and the second
    # shape dimension will be the number of channels. However, if the
    # Dense layer is in Squeeze-and-Excite, the first shape dimension
    # will be the batch size, the second and third shape dimension will be the
    # spatial sizes (should both be 1), and the fourth shape dimensions will
    # be the number of channels
    #
    # Note: asserts have been changed to sum(*shape > 1) <= 1 to avoid the case
    # when the dense layer has an output with shape (None, 1), which results in
    # sum(oshape > 1) = 0.
    ishape = np.array([i for i in input_shape if i is not None])
    assert sum(ishape > 1) <= 1, ("Input Tensor shape in %s has "
                                  "multiple >1 size dims") % layer.name
    size_i = np.max(ishape)

    oshape = np.array([i for i in output_shape if i is not None])
    assert sum(oshape > 1) <= 1, ("Output Tensor shape in %s has " +
                                  "multiple >1 size dims") % layer.name
    size_o = np.max(oshape)

    operation_count = (size_i * size_o)

  else:
    print("operation count for {} is defaulted to 0".format(
        layer))

  return int(operation_count)


def get_weights(layer, model_weights_already_quantized=True):
  """Get layer weights.

  Args:
    layer: given qkeras/keras layer
    model_weights_already_quantized: bool. whether the given layer's weights
      are already quantized. This is necessary because with certain quantizers,
      eg., quantized_bits(alpha="auto_po2"), we cannot quantize the same
      weights more than once, as it will lead to different results.

  Returns:
    Quantized layer weights.
  """

  weights = layer.get_weights()
  out = copy.deepcopy(weights)
  if not model_weights_already_quantized:
    for j, weight in enumerate(weights):
      if hasattr(layer, "get_quantizers") and layer.get_quantizers()[j]:
        out[j] = K.eval(
            layer.get_quantizers()[j](K.constant(weight)))
  return out


def adjust_multiplier_for_auto_po2(multiplier, qkeras_weight_quantizer):
  """Adjust multiplier when weight quantizer is auto_po2 type.

  Multiplier_bits = bits_x + bits_w
  Multiplier_intbits = log2(scale) + intbits_x + intbits_w

  Because we might have different scale for auto_po2 quantizer at different
  output channels, multiplier will have different integer bits at different
  output channel accordingly, which is not desirable in hardware implementation.
  Therefore we set a general multiplier quantizers so that it provides enough
  fractional bits and integer bits for all output channels.
  """
  print("adjust multiplier for auto_po2 ...")
  output_quantizer = multiplier.output
  if (hasattr(qkeras_weight_quantizer, "__str__") and
      ("quantized_bits" in qkeras_weight_quantizer.__str__() or 
       "quantized_linear" in qkeras_weight_quantizer.__str__()) and
      qkeras_weight_quantizer.alpha == "auto_po2"):
    bits = output_quantizer.bits
    int_bits = output_quantizer.int_bits
    if "quantized_bits" in qkeras_weight_quantizer.__str__():
      scale = qkeras_weight_quantizer.scale
    elif "quantized_linear" in qkeras_weight_quantizer.__str__():
      scale = qkeras_weight_quantizer.quantization_scale
    if hasattr(scale, "numpy"):
      scale = qkeras_weight_quantizer.scale
      # if scale doesn't have numpy() function, it means the quantizer has
      # never being called. Therfore we skip the following steps
      scale = scale.numpy()
      if isinstance(scale, np.ndarray):
        scale = np.squeeze(scale)
        max_shift = int(np.log2(np.max(scale)))
        min_shift = int(np.log2(np.min(scale)))
      elif isinstance(scale, (float, np.float32)):
        max_shift = int(np.log2(scale))
        min_shift = max_shift
      else:
        raise ValueError(f"Scale should be either numpy array or float,"
                         f"{type(scale)} is found instead!")

      # In order to set a general quantizer for different output channels,
      # we need to set both fractional bits and integer bits as the max required
      # bits for different output channels
      max_fractional_bits = bits - int_bits - min_shift
      max_int_bits = int_bits + max_shift
      total_bits = max_int_bits + max_fractional_bits

      output_quantizer.bits = total_bits
      output_quantizer.int_bits = max_int_bits
    else:
      print("[WARNING] The weight quantizer is never called even though it has "
            "alpha=auto_po2. In this case we do not adjust the multiplier and "
            "accumulator bit width since we don't know the exact values of "
            "scale", file=sys.stderr)
  elif hasattr(qkeras_weight_quantizer, "alpha") and (
      qkeras_weight_quantizer.alpha == "auto_po2"):
    print("[WARNING] auto_po2 is detected on a non-quantized_bits/"
          "quantized_linear quantizer. "
          "Currently in QTools we do not yet support the auto_po2 with the "
          f" given quantizer type: {type(qkeras_weight_quantizer)}."
          "Therefore we do not adjust the multiplier and accumulator bit width")


def adjust_accumulator_for_auto_po2(
    layer, multiplier, qkeras_weight_quantizer, bias_quantizer):
  """Adjust accumulator when weight quantizer is auto_po2 type."""

  fused_multiplier = copy.deepcopy(multiplier)
  adjust_multiplier_for_auto_po2(fused_multiplier, qkeras_weight_quantizer)
  weights = layer.get_weights()
  kernel = weights[0]

  kernel_shape = kernel.shape
  # depthwise_kernel_shape = kernel_size + (input_dim, depth_multiplier)
  # When computing accumulator bitwidth for dw conv2d layer, we do not
  # need to count the last two dimensions
  if layer.__class__.__name__ in ["QDepthwiseConv2D", "DepthwiseConv2D"]:
    assert kernel_shape[-1] == 1, ("depth_multiplier must be 1, "
                                   f"{kernel_shape[-1]} found instead!")
    kernel_shape = kernel.shape[:-2] + (1, 1)

  kernel_accumulator_factory = quantized_operators.AccumulatorFactory()
  # Sets use_bias=False so that the accumulator doesn't account for bias
  # bitwdith.
  fused_kernel_accumulator = kernel_accumulator_factory.make_accumulator(
      kernel_shape, fused_multiplier, use_bias=False)

  if not layer.use_bias:
    bias_quantizer = None
    fused_accumulator = fused_kernel_accumulator
  else:
    # Add bias quantizer bitwidth to the overall accumulator
    bias_accumulator_instance = quantized_operators.adder_factory.IAdder()
    fused_accumulator = bias_accumulator_instance.make_quantizer(
        fused_kernel_accumulator.output, bias_quantizer)

  return fused_accumulator


def find_divisors(num):
  return [i for i in range(1, num + 1) if num % i == 0]


def get_layer_info(layer: tf.keras.layers.Layer, attr_name: str):

  layer_type = layer.__class__.__name__
  supported_layer_types = ["QConv2D"]
  assert layer_type in supported_layer_types, (
      f"For now only {supported_layer_types} layers are supported. "
      f"Found {layer_type} instead.")

  # Get layer info such as input/output channels, kernel size and quantizers.
  input_channel = layer.input_shape[-1]
  output_channel = layer.output_shape[-1]

  kernel_height, kernel_width = layer.kernel_size if hasattr(
      layer, "kernel_size") else (None, None)

  quantizer_bits = layer.kernel_quantizer.bits
  layer_dict = {
      "layer_type": layer_type,
      "input_channel": input_channel,
      "output_channel": output_channel,
      "kernel_height": kernel_height,
      "kernel_width": kernel_width,
      "quantizer_bits": quantizer_bits
  }
  return layer_dict.get(attr_name, None)


def is_upsampled(layer: tf.keras.layers.Layer):
  # Evaluate if a given layer is doing upsampling.
  return "UpSampling" in layer.__class__.__name__
