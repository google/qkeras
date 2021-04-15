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
"""Definition of quantization package."""

# Some parts of the code were taken from
#
# https://github.com/BertMoons/QuantizedNeuralNetworks-Keras-Tensorflow
#
# and follows several papers.
#
#    https://arxiv.org/pdf/1609.07061.pdf
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import numpy as np
import tensorflow.compat.v1 as tf
from absl import logging

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.models import Model

from .qlayers import QActivation
from .qlayers import QAdaptiveActivation
from .qlayers import QDense
from .qconvolutional import QConv1D
from .qconvolutional import QConv2D
from .qconvolutional import QDepthwiseConv2D
from .qconvolutional import QSeparableConv2D
from .qpooling import QAveragePooling2D
from .quantizers import quantized_bits
from .quantizers import quantized_relu
from .quantizers import quantized_tanh
from .quantizers import quantized_ulaw
from .bn_folding_utils import unfold_model
from .utils import get_model_sparsity


def analyze_accumulator(in_model, x, verbose=False):
  """Analyzes the distribution of weights to specify size of accumulators.

     Computes the maximum number of bits for the accumulator assuming the
     inputs have a distribution given by the dictionary x.

     for each output channel i:
       max_positive_value[i] = sum(w[i]) + bias[i] for the positive weights
       max_negative_value[i] = sum(w[i]) + bias[i] for the negative weights

     max_value = max(
            max_positive_value[i] * positive(x) +
            max_negative_value[i] * negative(x),

         - (max_negative_value[i] * positive(x) +
            max_positive_value[i] * negative(x))
     )

     accumulator_size = ceil( log2( max_value ) )

     x right now is a dictionary of the form:

     { layer_name: (min_value, max_value) }

     in the future, we want to provide a sample and compute this automatically

  Arguments:
    in_model: keras model object, model to be evaluated
    x: dictionary of the form: { layer_name: (min_value, max_value) }
       input distribution
    verbose: boolean, if true, print statistics messages

  Returns:
    dictionary containing { layer_name: accumulator_size }
  """

  # this function converts a folded model to a "normal" model. It replace folded
  # layers (e.g., QConv2dBatchnorm) layer with qconv2d layer whenever possible.
  model = unfold_model(in_model)

  acc_sizes = {}

  for layer in model.layers:
    if (isinstance(layer, QDepthwiseConv2D) or
        isinstance(layer, QConv2D) or
        isinstance(layer, QConv1D) or
        isinstance(layer, QDense)):
      weights = layer.get_weights()
      k = weights[0]
      if layer.use_bias:
        b = weights[1]
      else:
        b = np.zeros((k.shape[-1],), dtype=np.float32)

      all_bits = []
      nbits = []
      for i in range(k.shape[1]):
        # compute sum of positive weights
        npp = np.sum(k[..., i] * (k[..., i] > 0)) + (b[i] > 0) * b[i]

        # compute sum of negative weights
        nnn = np.sum(k[..., i] * (k[..., i] < 0)) + (b[i] < 0) * b[i]

        # largest value is
        #   npp * largest positive - nnn * largest_negative or
        #   nnn * largest_positive - npp * largest_negative

        x_min = x[layer.name][0]
        x_max = x[layer.name][1]

        n1 = npp * (x_max > 0) * x_max + nnn * (x_min < 0) * x_min
        n0 = - (nnn * (x_max > 0) * x_max + npp * (x_min < 0) * x_min)

        if n1 > n0:
          nbits.append(n1)
        else:
          nbits.append(n0)

        all_bits.append((n1, n0))

      max_bits = int(np.ceil(np.log2(max(nbits))))
      acc_sizes[layer.name] = max_bits

      if verbose:
        print()
        print(layer.name, "- input range:", x[layer.name])
        print("  max value:", np.amax(k))
        print("  min value:", np.amin(k))
        print("  most positive sum:", np.amax(np.array(all_bits)[:, 0]))
        print("  most negative sum:", -np.amax(np.array(all_bits)[:, 1]))
        print("  number of bits:", max_bits)

  if verbose:
    print()

  return acc_sizes


def analyze_accumulator_from_sample(
    in_model, x_sample, mode="conservative", verbose=False):
  """Extracts range of inputs of quantized layers from samples."""

  # mode is one of "conservative", "sampled"
  if mode not in ["conservative", "sampled"]:
    ValueError("'mode' has to be 'conservative' or 'sampled'")

  # this function converts a folded model to a "normal" model. It replace folded
  # layers (e.g., QConv2DBatchnorm) layer with qconv2d layer whenever possible.
  model = unfold_model(in_model)

  # get layer names of quantized layers (QDense and QConv2D)
  layer_names = [
      layer.name for layer in model.layers
      if (isinstance(layer, QDepthwiseConv2D) or isinstance(layer, QConv2D) or
          isinstance(layer, QConv1D) or isinstance(layer, QDense))
  ]

  # sampled mode: just apply x_sample and check the outputs

  if mode == "sampled":

    outputs = [
        layer.output for layer in model.layers
        if (isinstance(layer, QDepthwiseConv2D) or isinstance(layer, QConv2D) or
            isinstance(layer, QConv1D) or isinstance(layer, QDense))
    ]

    eval_outputs = Model(inputs=model.inputs, outputs=outputs)

    # predict values for all inputs to quantized layers

    values = eval_outputs.predict(x_sample)

    acc_sizes = {}

    for name, value in zip(layer_names, values):
      max_value = np.amax(np.abs(value))
      if max_value != 0:
        acc_sizes[name] = int(np.ceil(np.log2(max_value)))
      else:
        acc_sizes[name] = 0

    return acc_sizes

  # get inputs of quantized layers (QDense and QConv2D
  # we use Activation("linear") to trick keras and tensorflow
  # to avoid direct connections of inputs and any other
  # artifacts.

  outputs = [
      Activation("linear")(layer.input) for layer in model.layers
      if (isinstance(layer, QDepthwiseConv2D) or isinstance(layer, QConv2D) or
          isinstance(layer, QConv1D) or isinstance(layer, QDense))
  ]

  eval_inputs = Model(inputs=model.inputs, outputs=outputs)

  # predict values for all inputs to quantized layers

  values = eval_inputs.predict(x_sample)

  x_dict = {}

  for name, value in zip(layer_names, values):
    x_dict[name] = (np.amin(value), np.amax(value))

  return analyze_accumulator(model, x_dict, verbose)


def get_quant_mode(quant):
  """Returns the quantizer mode, number of bits and if it is a signed number."""

  #  qb(n)[0] +/-,exp[1] t(-1,0,+1)[2] b(-1,+1)[3] b(0,1)[4]
  #  entry is tuple:
  #    (instance name, mode #(above), number of bits (-1 means check class),
  #     sign bit)

  modes = [
      # depending on the number of bits, quantized_bits may be 2, 2
      ("quantized_bits", 0, -1, 1),
      ("bernoulli", 4, 1, 0),
      ("stochastic_ternary", 2, 2, 1),
      ("ternary", 2, 2, 1),
      ("stochastic_binary", 3, 1, 1),
      ("binary", 3, 1, 1),
      # depending on the number of bits, quantized_relu may be 4, 1
      ("quantized_relu", 0, -1, 0),
      # depending on the number of bits, quantized_tanh may be 2, 2
      ("quantized_ulaw", 0, -1, 1),
      ("quantized_tanh", 0, -1, 1),
      ("quantized_po2", 1, -1, 1),
      ("quantized_relu_po2", 1, -1, 0),
      ("float", 5, 32, 1)
  ]

  for (inst, mode, bits, sign) in modes:
    if not quant or getattr(quant, "__name__", None) == "linear":
      # if quantizer not specified or linear, we use float type
      if inst == "float":
        return (mode, bits, sign)

    elif quant.__class__.__name__ == inst:
      if bits == -1:
        bits = int(quant.bits)
        if (
            isinstance(quant, quantized_bits) or
            isinstance(quant, quantized_tanh) or
            isinstance(quant, quantized_ulaw)):
          if bits == 2 and int(quant.integer) == 1:
            mode = 2
        elif isinstance(quant, quantized_relu):
          if bits == 1 and int(quant.integer) == 1:
            mode = 4
      return (mode, bits, sign)
  raise ValueError("Quantizer {} Not Found".format(quant))


def get_operation_type(layer, output_cache):
  """Checks quantizers around layer and weights to get operation type.

  Determines operator strenght according to the following table.
                                      x
                     qb(n)   +/-,exp  t(-1,0,+1) b(-1,+1) b(0,1) float
      qb(n)            *     << >>,-     ?,-       ?,-       ?    *
      +/-,exp        << >>,-   +         ?,-        ^      ?,-    *
    w t(-1,0,+1)      ?,-     ?,-        ?,^       ?,^      ^     *
      b(-1,+1)        ?,-      ^         ?,^        ^       ^     *
      b(0,1)           ?      ?,-         ^         ^       ^     *
      float           *        *          *        *        *     *

  Arguments:
    layer: layer in Keras to determine the operation strength.
    output_cache: cache of input tensor bit sizes.

  Returns:
    One of "mult", "fmult", "adder", "barrel", "mux", "xor".
    Note: "mult" represents quantized bit multiplier, "fmult" represents
          floating point multiplier.
  """

  wx_table = [
      ["mult", "barrel", "mux", "mux", "mux", "fmult"],
      ["barrel", "adder", "mux", "xor", "mux", "fmult"],
      ["mux", "mux", "mux", "mux", "xor", "fmult"],
      ["mux", "xor", "mux", "xor", "xor", "fmult"],
      ["mux", "mux", "xor", "xor", "xor", "fmult"],
      ["fmult", "fmult", "fmult", "fmult", "fmult", "fmult"],
  ]

  # check if this is a quantized layers (QDense, QConv, QDepthwise)
  if hasattr(layer, "get_quantizers"):
    w_quant = layer.get_quantizers()[0]
    w_mode, w_bits, w_sign = get_quant_mode(w_quant)
    if w_mode == "float":
      logging.warning("%s kernel is unquantized!", layer.name)

    # for the input, get tensor input and search the cache that associates
    # the quantizer with a tensor
    if output_cache.get(layer.input.experimental_ref(), None) is not None:
      x_mode, x_bits, x_sign = get_quant_mode(
          output_cache.get(layer.input.experimental_ref()))
      if x_mode == "float":
        logging.warning("%s input is unquantized!", layer.name)
    else:
      print("cannot determine presently model for {}".format(layer.name))
      return "null", (w_mode, -1), (w_bits, -1), (w_sign, -1)
    mode = wx_table[w_mode][x_mode]
    return mode, (w_mode, x_mode), (w_bits, x_bits), (w_sign, x_sign)

  raise ValueError("Cannot find suitable quantization candidates for {}".format(
      layer.name))


def create_activation_cache(model):
  """Creates an activation cache for the tensors of a model."""

  input_quantizer = quantized_relu(8, 0)

  output_cache = {}

  # If using a Sequential model, the input layer is hidden. Therefore, add the
  # input quantization to the cache if the first layer is not an input layer
  if not isinstance(model.layers[0], InputLayer):
    output_cache[model.layers[0].input.experimental_ref()] = input_quantizer

  # cache graph tensors' activations

  for l in model.layers:
    output_cache[l.output.experimental_ref()] = l
    if isinstance(l, QActivation) or isinstance(l, QAdaptiveActivation) :
      output_cache[l.output.experimental_ref()] = l.quantizer
    elif isinstance(l, InputLayer):
      # assume the input is 8-bit positive value
      output_cache[l.output.experimental_ref()] = input_quantizer
    elif l.__class__.__name__ in [
        "QDense", "QConv2D", "QConv1D", "QDepthwiseConv2D"
    ]:
      output_cache[l.output.experimental_ref()] = l.activation
    else:
      if isinstance(l.input, list):
        # right now, we just get the first one - we assume this is the leading
        # one.
        all_q = [
            output_cache.get(l.input[i].experimental_ref())
            for i in range(len(l.input))
        ]
        q = all_q[0]
      else:
        q = output_cache.get(l.input.experimental_ref(), None)
      output_cache[l.output.experimental_ref()] = q
      if q is None:
        raise ValueError("Unknown operation in {}".format(l.name))

  return output_cache


def extract_model_operations(in_model):
  """Determines types of operations for convolutions."""

  model = unfold_model(in_model)
  cache_q = create_activation_cache(model)
  cache_o = {}

  operations = {}

  for layer in model.layers:

    if layer.__class__.__name__ == "InputLayer":
      continue

    if isinstance(layer.input, list):
      input_shape = [
          cache_o.get(layer.input[i].experimental_ref(),
                      layer.input[i].get_shape())
          for i in range(len(layer.input))
      ]
    else:
      input_shape = cache_o.get(layer.input.experimental_ref(),
                                layer.input.get_shape())

    # Check if the inputs are a list of Dimensions
    if isinstance(input_shape, list):
      # Iterate though all of the input shapes and extract the dimension values
      for i, dim in enumerate(input_shape):
        if isinstance(dim[0], tf.Dimension):
          shape = [None]
          for j in range(1, len(dim)):
            shape.append(dim[j] if isinstance(dim[j], int) else dim[j].value)
          input_shape[i] = tuple(shape)

    output_shape = layer.compute_output_shape(input_shape)

    cache_o[layer.output.experimental_ref()] = output_shape

    if layer.__class__.__name__ not in ["QDense", "QConv2D", "QConv1D",
                                        "QDepthwiseConv2D", "QSeparableConv1D",
                                        "QSeparableConv2D"]:
      continue

    if layer.__class__.__name__ in ["QConv2D"]:

      _, _, _, channels_i = input_shape

      _, height_o, width_o, channels_o = output_shape

      weight = layer.get_weights()[0]


      kernel_h, kernel_w, _, _ = weight.shape

      number_of_operations = (
          height_o * width_o * channels_o * kernel_h * kernel_w * channels_i)

      number_of_weights = (kernel_h * kernel_w * channels_o * channels_i)

      number_of_bias = 0
      if len(layer.get_weights()) > 1:
        number_of_bias = layer.get_weights()[1].shape[0]

      weight_quant, bias_quant = layer.get_quantizers()
      weight_type = get_quant_mode(weight_quant)
      bias_type = get_quant_mode(bias_quant)

      if weight_type[0] == "float":
        logging.warning("%s kernel is unquantized!", layer.name)
      if bias_type[0] == "float":
        logging.warning("%s bias is unquantized!", layer.name)

    elif layer.__class__.__name__ in ["QConv1D"]:

      _, _, channels_i = input_shape

      _, time_o, channels_o = output_shape

      weight = layer.get_weights()[0]

      kernel_length, _, _ = weight.shape

      number_of_operations = (
          time_o * channels_o * kernel_length * channels_i)

      number_of_weights = (kernel_length * channels_o * channels_i)
      number_of_bias = 0
      if len(layer.get_weights()) > 1:
        number_of_bias = layer.get_weights()[1].shape[0]

      weight_quant, bias_quant = layer.get_quantizers()
      weight_type = get_quant_mode(weight_quant)
      bias_type = get_quant_mode(bias_quant)

      if weight_type[0] == "float":
        logging.warning("%s kernel is unquantized!", layer.name)
      if bias_type[0] == "float":
        logging.warning("%s bias is unquantized!", layer.name)

    elif layer.__class__.__name__ in ["QDepthwiseConv2D"]:

      _, _, _, channels_i = input_shape

      _, height_o, width_o, channels_o = output_shape

      weight_1 = layer.get_weights()[0]

      kernel_h, kernel_w, _, _ = weight_1.shape

      number_of_operations = (
          kernel_h * kernel_w * height_o * width_o * channels_i)

      number_of_weights = (kernel_h * kernel_w * channels_o * channels_i)

      number_of_bias = 0
      if len(layer.get_weights()) > 1:
        number_of_bias = layer.get_weights()[1].shape[0]

      weight_quant, bias_quant = layer.get_quantizers()
      weight_type = get_quant_mode(weight_quant)
      bias_type = get_quant_mode(bias_quant)

      if weight_type[0]  == "float":
        logging.warning("%s kernel is unquantized!", layer.name)
      if bias_type[0]  == "float":
        logging.warning("%s bias is unquantized!", layer.name)

    elif layer.__class__.__name__ in ["QSeparableConv1D"]:

      _, _, channels_i = input_shape

      _, time_o, channels_o = output_shape

      weight_1 = layer.get_weights()[0]

      kernel_length, _, _ = weight_1.shape

      number_of_operations = (
          kernel_length * time_o * channels_i + 
          time_o * channels_o)

      number_of_weights = [
        kernel_length * channels_i, 
        channels_o * channels_i]

      number_of_bias = 0
      if len(layer.get_weights()) > 2:
        number_of_bias = layer.get_weights()[2].shape[0]

      depthwise_quant, pointwise_quant, bias_quant = layer.get_quantizers()
      depthwise_type = get_quant_mode(depthwise_quant)
      pointwise_type = get_quant_mode(pointwise_quant)
      weight_type = [depthwise_type, pointwise_type]
      bias_type = get_quant_mode(bias_quant)

      if depthwise_type[0] == "float":
        logging.warning("%s depthwise kernel is unquantized!", layer.name)
      if pointwise_type[0] == "float":
        logging.warning("%s pointwise kernel is unquantized!", layer.name)
      if bias_type[0] == "float":
        logging.warning("%s bias is unquantized!", layer.name)

    elif layer.__class__.__name__ in ["QSeparableConv2D"]:

      _, _, _, channels_i = input_shape

      _, height_o, width_o, channels_o = output_shape

      weight_1 = layer.get_weights()[0]

      kernel_h, kernel_w, _, _ = weight_1.shape

      number_of_operations = (
          kernel_h * kernel_w * height_o * width_o * channels_i + 
          height_o * width_o * channels_o)

      number_of_weights = [
        kernel_h * kernel_w * channels_i,
        channels_o * channels_i]

      number_of_bias = 0
      if len(layer.get_weights()) > 2:
        number_of_bias = layer.get_weights()[2].shape[0]

      depthwise_quant, pointwise_quant, bias_quant = layer.get_quantizers()
      depthwise_type = get_quant_mode(depthwise_quant)
      pointwise_type = get_quant_mode(pointwise_quant)
      weight_type = [depthwise_type, pointwise_type]
      bias_type = get_quant_mode(bias_quant)

      if depthwise_type[0] == "float":
        logging.warning("%s depthwise kernel is unquantized!", layer.name)
      if pointwise_type[0] == "float":
        logging.warning("%s pointwise kernel is unquantized!", layer.name)
      if bias_type[0] == "float":
        logging.warning("%s bias is unquantized!", layer.name)

    elif layer.__class__.__name__ in ["QDense"]:

      _, size_i = input_shape
      _, size_o = output_shape

      number_of_operations = (size_i * size_o)

      number_of_weights = size_i * size_o
      number_of_bias = 0
      if len(layer.get_weights()) > 1:
        number_of_bias = layer.get_weights()[1].shape[0]

      weight_quant, bias_quant = layer.get_quantizers()
      weight_type = get_quant_mode(weight_quant)
      bias_type = get_quant_mode(bias_quant)

      if weight_type[0] == "float":
        logging.warnings("%s kernel is unquantized!", layer.name)
      if bias_type[0] == "float":
        logging.warnings("%s bias is unquantized!", layer.name)

    # "number_of_operations" is tensor_shape.Dimension type
    operations[layer.name] = {
        "type":
            get_operation_type(layer, cache_q),
        "number_of_operations":
            number_of_operations if isinstance(number_of_operations, int) else
            number_of_operations.value,
        "number_of_weights":
            number_of_weights,
            # if isinstance(number_of_weights, int) else number_of_weights.value,
        "number_of_bias":
            number_of_bias,
            # if isinstance(number_of_bias, int) else number_of_bias.value,
        "type_of_weights":
            weight_type,
        "type_of_bias":
            bias_type,
    }

  return operations


def print_qstats(model):
  """Prints quantization statistics for the model."""

  model_ops = extract_model_operations(model)

  ops_table = defaultdict(lambda: 0)

  print("")
  print("Number of operations in model:")
  for name in sorted(model_ops):
    mode, _, sizes, signs = model_ops[name]["type"]
    number = model_ops[name]["number_of_operations"]
    sign = "s" if sum(signs) > 0 else "u"
    op_name = sign + mode + "_" + str(sizes[0]) + "_" + str(sizes[1])
    ops_table[op_name] += number
    print("    {:30}: {:5} ({})".format(str(name), str(number), str(op_name)))

  print("")
  print("Number of operation types in model:")
  for key in sorted(ops_table.keys()):
    if ops_table[key] > 0:
      print("    {:30}: {}".format(key, ops_table[key]))

  print("")
  print("Weight profiling:")
  for name in sorted(model_ops):
    weight_type = model_ops[name]["type_of_weights"]
    n_weights = model_ops[name]["number_of_weights"]
    if isinstance(weight_type, list):
      for i, (w_type, w_number) in enumerate(zip(weight_type, n_weights)):
        _, w_sizes, _ = w_type
        print("    {:30} : {:5} ({}-bit unit)".format(
            str(name) + "_weights_" + str(i), str(w_number), str(w_sizes)))
    else:
      _, w_sizes, _ = weight_type
      print("    {:30} : {:5} ({}-bit unit)".format(
          str(name) + "_weights", str(n_weights), str(w_sizes)))
    _, b_sizes, _ = model_ops[name]["type_of_bias"]
    b_number = model_ops[name]["number_of_bias"]
    print("    {:30} : {:5} ({}-bit unit)".format(
        str(name) + "_bias", str(b_number), str(b_sizes)))

  print("")
  print("Weight sparsity:")
  total_sparsity, per_layer = get_model_sparsity(model, per_layer=True)
  for layer in per_layer:
    print("    {:30} : {:.4f}".format(str(layer[0]), layer[1]))
  print("    " + ("-"*40))
  print("    {:30} : {:.4f}".format("Total Sparsity", total_sparsity))
