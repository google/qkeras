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
"""Tests for various model architectures."""

import json

import numpy as np
import pytest
import tensorflow.keras as keras
import tensorflow as tf

from qkeras import QActivation
from qkeras import QAdaptiveActivation
from qkeras import QBatchNormalization
from qkeras import QConv2D
from qkeras import QDense
from qkeras import quantizers
from qkeras.qtools import interface
from qkeras.qtools import qgraph
from qkeras.qtools import run_qtools
from qkeras.qtools import settings as qtools_settings
from qkeras.qtools.quantized_operators import divider_factory
from qkeras.qtools import generate_layer_data_type_map
from qkeras.utils import model_save_quantized_weights



def qdense_model_fork():
  x = x_in = keras.layers.Input((23,), name="input")
  x = QDense(
      10,
      kernel_quantizer=quantizers.quantized_bits(5, 0, 1),
      bias_quantizer=quantizers.quantized_bits(5, 0, 1),
      activation=quantizers.quantized_po2(3, 1),
      name="qdense_0")(x)
  x = QDense(
      20,
      kernel_quantizer=quantizers.quantized_bits(5, 0, 1),
      bias_quantizer=quantizers.quantized_bits(5, 0, 1),
      activation=quantizers.quantized_relu(6, 2),
      name="qdense_1")(x)
  x = QActivation("quantized_relu(4)", name="QA_2")(x)
  x_1 = QDense(
      30,
      kernel_quantizer=quantizers.binary(),
      bias_quantizer=quantizers.binary(),
      name="qdense_3")(x)
  x_2 = QActivation("quantized_relu(6,2)", name="QA_3")(x)

  model = keras.Model(
      inputs=[x_in], outputs=[x_1, x_2,])
  return model


def qconv_model():
  x = x_in = keras.layers.Input((23, 23, 1), name="input")
  x = QActivation("quantized_relu(4)", name="QA_0")(x)
  x = QConv2D(
      16, 2, 2,
      kernel_quantizer=quantizers.binary(),
      bias_quantizer=quantizers.ternary(),
      name="qconv2d_1")(x)
  x = QConv2D(
      8, 2, 2,
      kernel_quantizer=quantizers.quantized_bits(4, 0, 1),
      bias_quantizer=quantizers.quantized_bits(4, 0, 1),
      activation=quantizers.quantized_relu(6, 2),
      name="qconv2D_2")(x)
  x = QConv2D(
      2, 2, 2,
      kernel_quantizer=quantizers.quantized_bits(4, 0, 1),
      bias_quantizer=quantizers.quantized_bits(4, 0, 1),
      activation=quantizers.quantized_relu(6, 2),
      name="qconv2d_3")(x)
  x = QActivation("quantized_bits(6, 0, 1)", name="QA_4")(x)

  model = keras.Model(
      inputs=[x_in], outputs=[x])
  return model


def po2_qbits_model():
  x = x_in = keras.layers.Input((23, 23, 1), name="input")
  x = QActivation("quantized_relu_po2(3, 2)", name="QA_0")(x)
  x = QConv2D(
      16, 2, 2,
      kernel_quantizer=quantizers.quantized_bits(4, 0, 1),
      bias_quantizer=quantizers.quantized_bits(4, 0, 1),
      name="qconv2d_1")(x)

  model = keras.Model(inputs=[x_in], outputs=[x])
  return model


def float_po2_model():
  x = x_in = keras.layers.Input((23, 23, 1), name="input")
  x = QConv2D(
      16, 2, 2,
      kernel_quantizer=quantizers.quantized_po2(5, 0),
      bias_quantizer=quantizers.quantized_po2(5, 0),
      name="qconv2d_1")(x)
  x = QActivation("quantized_relu_po2(3, 2)", name="QA_0")(x)
  x = QConv2D(
      10, 2, 2,
      kernel_quantizer=quantizers.quantized_bits(5, 2, 1),
      bias_quantizer=quantizers.quantized_bits(5, 2, 1),
      name="qconv2d_0")(x)
  model = keras.Model(
      inputs=[x_in], outputs=[x])

  for layer in model.layers:
    print(layer)
    print(layer.output_shape)
  return model


def qbn_model(
    act="binary(use_01=0)",
    gamma=quantizers.quantized_relu_po2(4, 2),
    variance=quantizers.quantized_relu_po2(4, 2),
    beta=None, mean=None):

  x = x_in = keras.layers.Input((23, 23, 1), name="input")
  x = QActivation(act, name="QA_0")(x)
  x = QBatchNormalization(
      gamma_quantizer=gamma,
      variance_quantizer=variance,
      beta_quantizer=beta,
      mean_quantizer=mean,
      gamma_range=8, beta_range=4, name="qbn_1")(x)

  model = keras.Model(
      inputs=[x_in], outputs=[x])

  return model


def qbn_model_inference():

  x = x_in = keras.layers.Input((23, 23, 1), name="input")
  x = QConv2D(
      4, 2, 23,
      kernel_quantizer=quantizers.quantized_bits(4, 0, 1),
      bias_quantizer=quantizers.quantized_bits(4, 0, 1),
      use_bias=False,
      name="qconv2d_1")(x)
  x = QBatchNormalization(
      mean_quantizer=quantizers.quantized_bits(6, 0, 1),
      gamma_quantizer=None,
      variance_quantizer=None,
      beta_quantizer=quantizers.quantized_bits(6, 0, 1),
      inverse_quantizer=quantizers.quantized_bits(16, 0, 1),
      scale=False,
      center=False,
      gamma_range=8, beta_range=4, name="qbn_2")(x)

  x = QConv2D(
      2, 1, 1,
      kernel_quantizer=quantizers.quantized_bits(3, 0),
      bias_quantizer=quantizers.quantized_bits(3, 2),
      name="qconv2d_3")(x)

  model = keras.Model(inputs=[x_in], outputs=[x])
  hw_weight_dict = model_save_quantized_weights(model)

  return (hw_weight_dict, model)


def add_qmodel(quantizer1, quantizer2, quantizer3):

  # Layer that add a list of inputs.
  # It takes as input a list of tensors, all of the same shape,
  # and returns a single tensor (also of the same shape).

  x1 = input1 = keras.layers.Input((16,), name="input_0")
  x1 = QDense(
      8,
      kernel_quantizer=quantizers.quantized_bits(5, 0, 1),
      bias_quantizer=quantizers.quantized_bits(5, 0, 1),
      activation=quantizer1, name="dense_0")(x1)

  x2 = input2 = keras.layers.Input(shape=(32,), name="input_1")
  x2 = QDense(
      8,
      kernel_quantizer=quantizers.quantized_bits(5, 0, 1),
      bias_quantizer=quantizers.quantized_bits(5, 0, 1),
      activation=quantizer2, name="dense_1")(x2)

  x3 = input3 = keras.layers.Input(shape=(64,), name="input_2")
  x3 = QDense(
      8,
      kernel_quantizer=quantizers.quantized_bits(5, 0, 1),
      bias_quantizer=quantizers.quantized_bits(5, 0, 1),
      activation=quantizer3, name="dense_2")(x3)

  x = keras.layers.add([x1, x2, x3], name="add")

  model = keras.Model(
      inputs=[input1, input2, input3], outputs=[x])

  return model


def multiply_qmodel():

  # element-wise multiply a list of inputs.
  # It takes as input a list of tensors, all of the same shape,
  # and returns a single tensor (also of the same shape).
  x1 = input1 = keras.layers.Input((16,), name="input_0")
  x1 = QDense(
      8,
      kernel_quantizer=quantizers.quantized_bits(5, 0, 1),
      bias_quantizer=quantizers.quantized_bits(5, 0, 1),
      activation=quantizers.quantized_bits(4, 0, 1),
      name="dense_0")(x1)

  x2 = input2 = keras.layers.Input(shape=(32,), name="input_1")
  x2 = QDense(
      8,
      kernel_quantizer=quantizers.quantized_bits(5, 0, 1),
      bias_quantizer=quantizers.quantized_bits(5, 0, 1),
      activation=quantizers.quantized_bits(5, 0, 1),
      name="dense_1")(x2)

  x3 = input3 = keras.layers.Input(shape=(64,), name="input_2")
  x3 = QDense(
      8,
      kernel_quantizer=quantizers.quantized_bits(5, 0, 1),
      bias_quantizer=quantizers.quantized_bits(5, 0, 1),
      activation=quantizers.quantized_bits(6, 0, 1),
      name="dense_2")(x3)

  x = keras.layers.multiply([x1, x2, x3], name="multiply")
  model = keras.Model(
      inputs=[input1, input2, input3], outputs=[x])

  return model


def pooling_qmodel():

  # Average pooling and global average pooling operation for spatial data.
  x = input1 = keras.layers.Input((16, 16, 3), name="input")
  x = keras.layers.AveragePooling2D(pool_size=(2, 2), padding="valid",
                                    name="avg_pooling")(x)
  x = keras.layers.GlobalAveragePooling2D(name="global_avg_pooling")(x)

  model = keras.Model(inputs=[input1], outputs=[x])

  return model


def maximum_qmodel(quantizer1, quantizer2, quantizer3):

  # element-wise maximum/minimum/average of a list of inputs.
  # It takes as input a list of tensors, all of the same shape,
  # and returns a single tensor (also of the same shape).
  x1 = input1 = keras.layers.Input((16,), name="input_0")
  x1 = QDense(
      8,
      kernel_quantizer=quantizers.quantized_bits(5, 0, 1),
      bias_quantizer=quantizers.quantized_bits(5, 0, 1),
      activation=quantizer1, name="qdense_0")(x1)

  x2 = input2 = keras.layers.Input(shape=(32,), name="input_1")
  x2 = QDense(
      8,
      kernel_quantizer=quantizers.quantized_bits(5, 0, 1),
      bias_quantizer=quantizers.quantized_bits(5, 0, 1),
      activation=quantizer2, name="dense_1")(x2)

  x3 = input3 = keras.layers.Input(shape=(64,), name="input_2")
  x3 = QDense(
      8,
      kernel_quantizer=quantizers.quantized_bits(5, 0, 1),
      bias_quantizer=quantizers.quantized_bits(5, 0, 1),
      activation=quantizer3, name="dense_2")(x3)

  x = keras.layers.maximum([x1, x2, x3], name="maximum")
  model = keras.Model(
      inputs=[input1, input2, input3], outputs=[x])

  return model


def concatenate_qmodel(quantizer1, quantizer2, quantizer3):

  # Layer that concatenates a list of inputs.
  # It takes as input a list of tensors, all of the same shape except
  # for the concatenation axis, and returns a single tensor,
  # the concatenation of all inputs..

  x1 = input1 = keras.layers.Input((16, 16, 1), name="input_0")
  x1 = QConv2D(
      16, 2, 2,
      kernel_quantizer=quantizer1,
      bias_quantizer=quantizer1,
      name="conv2d_0")(x1)

  x2 = input2 = keras.layers.Input((16, 16, 1), name="input_1")
  x2 = QConv2D(
      32, 2, 2,
      kernel_quantizer=quantizer2,
      bias_quantizer=quantizer2,
      name="conv2d_1")(x2)

  x3 = input3 = keras.layers.Input((16, 16, 1), name="input_2")
  x3 = QConv2D(
      64, 2, 2,
      kernel_quantizer=quantizer3,
      bias_quantizer=quantizer3,
      name="conv2d_2")(x3)

  x = keras.layers.concatenate([x1, x2, x3], axis=-1, name="concatenate")
  model = keras.Model(inputs=[input1, input2, input3], outputs=[x])

  return model


def run(model, input_quantizers, is_inference=False,
        verbose=False, hw_weight_dict=None):
  (graph, source_quantizer_list) = qgraph.CreateGraph(
      model, input_quantizers)
  # qgraph.PrintGraph(graph)
  qgraph.GraphPropagateActivationsToEdges(graph)

  layer_map = generate_layer_data_type_map.generate_layer_data_type_map(
      graph=graph, source_quantizer_list=source_quantizer_list,
      is_inference=is_inference, hw_weight_dict=hw_weight_dict)

  # interface.print_layer_data_type_map(dict)
  output_dict = interface.map_to_json(layer_map)

  if verbose:
    dict_to_json = json.dumps(output_dict, indent=4)
    print(dict_to_json)

  return output_dict


def test_wrong_input_quantizers():
  input_quantizers = [
      quantizers.quantized_bits(4, 0, 1),
      quantizers.quantized_bits(5, 0, 1),
      quantizers.quantized_bits(6, 0, 1)
  ]
  # INPUT_QUANTIZERS = None
  x1 = input1 = keras.layers.Input((16,), name="input_0")
  x1 = QDense(
      8,
      kernel_quantizer=quantizers.quantized_bits(5, 0, 1),
      bias_quantizer=quantizers.quantized_bits(5, 0, 1),
      name="dense_0")(x1)
  x2 = input2 = keras.layers.Input(shape=(32,), name="input_1")
  x2 = QDense(
      8,
      kernel_quantizer=quantizers.quantized_bits(5, 0, 1),
      bias_quantizer=quantizers.quantized_bits(5, 0, 1),
      name="dense_1")(x2)
  x = keras.layers.add([x1, x2], name="add")

  model = keras.Model(
      inputs=[input1, input2], outputs=[x])

  with pytest.raises(qgraph.WrongInputQuantizerError):
    run(model, input_quantizers)


def test_qbn_inference():
  input_quantizers = [quantizers.quantized_bits(4, 0, 1)]
  (hw_weight_dict, model) = qbn_model_inference()

  dtype_dict = run(model, input_quantizers, is_inference=True,
                   hw_weight_dict=hw_weight_dict)
  multiplier = dtype_dict["qconv2d_1"]["multiplier"]
  accumulator = dtype_dict["qconv2d_1"]["accumulator"]
  output = dtype_dict["qconv2d_1"]["output_quantizer"]
  fused_accumulator = dtype_dict["qconv2d_1"]["fused_accumulator"]

  assert multiplier["quantizer_type"] == "quantized_bits"
  assert multiplier["bits"] == 9
  assert multiplier["int_bits"] == 2
  assert multiplier["is_signed"] == 1
  assert multiplier["op_type"] == "mul"

  assert accumulator["quantizer_type"] == "quantized_bits"
  assert accumulator["bits"] == 11
  assert accumulator["int_bits"] == 4
  assert accumulator["is_signed"] == 1
  assert accumulator["op_type"] == "add"

  assert fused_accumulator["quantizer_type"] == "quantized_bits"
  assert fused_accumulator["bits"] == 28
  assert fused_accumulator["int_bits"] == 5
  assert accumulator["is_signed"] == 1
  assert fused_accumulator["op_type"] == "add"


def test_invalid_denominator_qbn():
  input_quantizers = None
  act = "binary(use_01=0)"
  gamma = quantizers.ternary()
  variance = gamma
  model = qbn_model(
      act=act, gamma=gamma, variance=variance,
      beta=None, mean=None)
  with pytest.raises(divider_factory.UnacceptedQuantizerError):
    run(model, input_quantizers)


def test_conv2d():
  input_quantizers = None

  act = "quantized_bits(6, 0, 1)"
  weight = quantizers.quantized_relu_po2(4, 2)
  x = x_in = keras.layers.Input((23, 23, 1), name="input")
  x = QActivation(act, name="QA_0")(x)
  x = QConv2D(
      16, 2, 2,
      kernel_quantizer=weight,
      bias_quantizer=weight,
      name="qconv2d_1")(x)

  model = keras.Model(inputs=[x_in], outputs=[x])

  dtype_dict = run(model, input_quantizers)
  multiplier = dtype_dict["qconv2d_1"]["multiplier"]
  accumulator = dtype_dict["qconv2d_1"]["accumulator"]
  op_count = dtype_dict["qconv2d_1"]["operation_count"]

  assert multiplier["quantizer_type"] == "quantized_bits"
  assert multiplier["bits"] == 15
  assert multiplier["int_bits"] == 2
  assert multiplier["is_signed"] == 1
  assert multiplier["op_type"] == "shifter"
  assert accumulator["quantizer_type"] == "quantized_bits"
  assert accumulator["bits"] == 18
  assert accumulator["int_bits"] == 5
  assert accumulator["is_signed"] == 1
  assert accumulator["op_type"] == "add"
  assert op_count == 7744


def test_qdense_model_fork():
  input_quantizers = [quantizers.quantized_bits(4, 0, 1)]
  model = qdense_model_fork()
  dtype_dict = run(model, input_quantizers)

  multiplier = dtype_dict["qdense_3"]["multiplier"]
  assert multiplier["quantizer_type"] == "quantized_bits"
  assert multiplier["bits"] == 5
  assert multiplier["int_bits"] == 1
  assert multiplier["is_signed"] == 1
  assert multiplier["op_type"] == "mux"

  accumulator = dtype_dict["qdense_3"]["accumulator"]
  assert accumulator["quantizer_type"] == "quantized_bits"
  assert accumulator["bits"] == 11
  assert accumulator["int_bits"] == 7
  assert accumulator["is_signed"] == 1
  assert accumulator["op_type"] == "add"


def test_util_layers():
  input_quantizers = None  # quantizers.quantized_bits(4, 0, 1)

  act = "quantized_bits(6, 0, 1)"
  x = x_in = keras.layers.Input((24, 24, 1), name="input")
  x = QActivation(act, name="QA_0")(x)
  x = keras.layers.Reshape((12 * 12, 4, 1), name="reshape_1")(x)
  x = keras.layers.MaxPooling2D(
      pool_size=(2, 2), name="maxpooling_2")(x)
  x = keras.layers.Flatten(name="flatten_3")(x)
  x = QDense(
      30,
      kernel_quantizer=quantizers.binary(use_01=1),
      bias_quantizer=quantizers.binary(use_01=1),
      activation=quantizers.quantized_po2(3, 2),
      name="qdense_4")(x)

  model = keras.Model(inputs=[x_in], outputs=[x])
  dtype_dict = run(model, input_quantizers)

  multiplier = dtype_dict["qdense_4"]["multiplier"]
  assert multiplier["quantizer_type"] == "quantized_bits"
  assert multiplier["bits"] == 6
  assert multiplier["int_bits"] == 1
  assert multiplier["is_signed"] == 1
  assert multiplier["op_type"] == "and"

  accumulator = dtype_dict["qdense_4"]["accumulator"]
  assert accumulator["quantizer_type"] == "quantized_bits"
  assert accumulator["bits"] == 15
  assert accumulator["int_bits"] == 10
  assert accumulator["is_signed"] == 1
  assert accumulator["op_type"] == "add"

  output = dtype_dict["qdense_4"]["output_quantizer"]
  assert output["quantizer_type"] == "quantized_po2"
  assert output["bits"] == 3
  assert output["is_signed"] == 1
  assert output["max_value"] == 2


def test_merge_layers():
  input_quantizers = [
      quantizers.quantized_bits(4, 0, 1), quantizers.quantized_bits(5, 0, 1),
      quantizers.quantized_bits(6, 0, 1)]
  model = add_qmodel(
      quantizers.quantized_bits(4, 0, 1), quantizers.quantized_bits(5, 0, 0),
      quantizers.quantized_bits(6, 0, 1))
  dtype_dict = run(model, input_quantizers)
  merge_quantizer = dtype_dict["add"]["Add_quantizer"]
  assert merge_quantizer["quantizer_type"] == "quantized_bits"
  assert merge_quantizer["bits"] == 7
  assert merge_quantizer["int_bits"] == 2
  assert merge_quantizer["is_signed"] == 1

  model = multiply_qmodel()
  dtype_dict = run(model, input_quantizers)
  merge_quantizer = dtype_dict["multiply"]["Multiply_quantizer"]
  assert merge_quantizer["quantizer_type"] == "quantized_bits"
  assert merge_quantizer["bits"] == 15
  assert merge_quantizer["int_bits"] == 1
  assert merge_quantizer["is_signed"] == 1
  assert merge_quantizer["op_type"] == "mul"

  model = maximum_qmodel(
      quantizers.quantized_bits(4, 0, 1), quantizers.quantized_bits(5, 0, 0),
      quantizers.quantized_bits(6, 0, 1))
  dtype_dict = run(model, input_quantizers)
  merge_quantizer = dtype_dict["maximum"]["Maximum_quantizer"]
  assert merge_quantizer["quantizer_type"] == "quantized_bits"
  assert merge_quantizer["bits"] == 6
  assert merge_quantizer["int_bits"] == 1
  assert merge_quantizer["is_signed"] == 1

  model = concatenate_qmodel(
      quantizers.quantized_bits(4, 0, 1), quantizers.quantized_bits(5, 0, 0),
      quantizers.quantized_bits(6, 0, 1))
  dtype_dict = run(model, input_quantizers)
  merge_quantizer = dtype_dict["concatenate"]["Concatenate_quantizer"]
  assert merge_quantizer["quantizer_type"] == "quantized_bits"
  assert merge_quantizer["bits"] == 15
  assert merge_quantizer["int_bits"] == 4
  assert merge_quantizer["is_signed"] == 1


def test_pooling():
  input_quantizers = [quantizers.quantized_bits(8, 0, 1)]
  model = pooling_qmodel()
  dtype_dict = run(model, input_quantizers)

  accumulator = dtype_dict["avg_pooling"]["accumulator"]
  assert accumulator["quantizer_type"] == "quantized_bits"
  assert accumulator["bits"] == 10
  assert accumulator["int_bits"] == 3

  accumulator = dtype_dict["global_avg_pooling"]["accumulator"]
  assert accumulator["quantizer_type"] == "quantized_bits"
  assert accumulator["bits"] == 16
  assert accumulator["int_bits"] == 9


def test_qenergy():
  x = x_in = keras.layers.Input((784,), name="input")
  x = QDense(
      300,
      kernel_quantizer=quantizers.binary(),
      bias_quantizer=quantizers.binary(),
      name="d0")(x)
  x = QActivation("quantized_relu(4,0)", name="d0_qr4")(x)
  x = QDense(100, kernel_quantizer=quantizers.quantized_bits(4, 0, 1),
             bias_quantizer=quantizers.quantized_bits(4, 0, 1),
             name="d1")(x)
  x = QAdaptiveActivation("quantized_relu", 4, name="d1_qr4")(x)
  x = QDense(
      10, kernel_quantizer=quantizers.quantized_bits(4, 0, 1),
      bias_quantizer=quantizers.quantized_bits(4, 0, 1),
      name="d2")(x)
  x = keras.layers.Activation("softmax", name="softmax")(x)

  model = keras.Model(inputs=[x_in], outputs=[x])
  # print(model.summary())

  reference_internal = "int8"
  reference_accumulator = "int32"

  # get reference energy cost
  q = run_qtools.QTools(
      model, process="horowitz",
      source_quantizers=reference_internal,
      is_inference=False, weights_path=None,
      keras_quantizer=reference_internal,
      keras_accumulator=reference_accumulator,
      for_reference=True)

  ref_energy_dict = q.pe(
      weights_on_memory="sram",
      activations_on_memory="sram",
      min_sram_size=8*16*1024*1024,
      rd_wr_on_io=False)
  reference_size = q.extract_energy_sum(
      qtools_settings.cfg.include_energy, ref_energy_dict)

  # get trial energy cost
  q = run_qtools.QTools(
      model, process="horowitz",
      source_quantizers=reference_internal,
      is_inference=False, weights_path=None,
      keras_quantizer=reference_internal,
      keras_accumulator=reference_accumulator,
      for_reference=False)
  trial_energy_dict = q.pe(
      weights_on_memory="sram",
      activations_on_memory="sram",
      min_sram_size=8*16*1024*1024,
      rd_wr_on_io=False)
  trial_size = q.extract_energy_sum(
      qtools_settings.cfg.include_energy, trial_energy_dict)

  # Reference energy number is now updated with keras_accumulator as
  # output quantizer
  tmp = ref_energy_dict["d0"]["energy"]
  assert tmp["inputs"] == pytest.approx(372.77, abs=0.1)
  assert tmp["outputs"] == pytest.approx(570.57, abs=0.1)
  assert tmp["parameters"] == pytest.approx(111975.96, abs=0.1)
  assert tmp["op_cost"] == pytest.approx(70560.0, abs=0.1)

  tmp = ref_energy_dict["d1"]["energy"]
  assert tmp["inputs"] == pytest.approx(570.57, abs=0.1)
  assert tmp["outputs"] == pytest.approx(190.19, abs=0.1)
  assert tmp["parameters"] == pytest.approx(14313.66, abs=0.1)
  assert tmp["op_cost"] == pytest.approx(26500.0, abs=0.1)

  tmp = ref_energy_dict["d2"]["energy"]
  assert tmp["inputs"] == pytest.approx(190.19, abs=0.1)
  assert tmp["outputs"] == pytest.approx(19.02, abs=0.1)
  assert tmp["parameters"] == pytest.approx(483.08, abs=0.1)
  assert tmp["op_cost"] == pytest.approx(883.33, abs=0.1)

  # Trial
  tmp = trial_energy_dict["d0"]["energy"]
  assert tmp["inputs"] == pytest.approx(372.77, abs=0.1)
  assert tmp["outputs"] == pytest.approx(342.34, abs=0.1)
  assert tmp["parameters"] == pytest.approx(13997.95, abs=0.1)
  assert tmp["op_cost"] == pytest.approx(15729.0, abs=0.1)

  tmp = trial_energy_dict["d1"]["energy"]
  assert tmp["inputs"] == pytest.approx(72.27, abs=0.1)
  assert tmp["outputs"] == pytest.approx(110.31, abs=0.1)
  assert tmp["parameters"] == pytest.approx(7158.73, abs=0.1)
  assert tmp["op_cost"] == pytest.approx(3250.0, abs=0.1)

  tmp = trial_energy_dict["d2"]["energy"]
  assert tmp["inputs"] == pytest.approx(26.63, abs=0.1)
  assert tmp["outputs"] == pytest.approx(11.41, abs=0.1)
  assert tmp["parameters"] == pytest.approx(243.44, abs=0.1)
  assert tmp["op_cost"] == pytest.approx(102.08, abs=0.1)

  # print(ref_energy_dict)
  # print(trial_energy_dict)
  assert int(reference_size) == 226629
  assert int(trial_size) == 41070


def test_quntized_reference_energy_same_as_floating_trial():
  # Test if reference energy from quantized model and floating model is the
  # same
  def get_model(quantize=False):
    x1 = input1 = keras.layers.Input((16, 16, 3), name="input_0")
    if quantize:
      x1 = QConv2D(
          16, 2, 2,
          kernel_quantizer=quantizers.quantized_bits(5, 0, 1),
          bias_quantizer=quantizers.quantized_bits(5, 0, 1),
          name="conv_0")(x1)
    else:
      x1 = keras.layers.Conv2D(16, 2, 2, name="conv_0")(x1)

    x2 = input2 = keras.layers.Input(shape=(16, 16, 3), name="input_1")
    if quantize:
      x2 = QConv2D(
          16, 2, 2,
          kernel_quantizer=quantizers.quantized_bits(5, 0, 1),
          bias_quantizer=quantizers.quantized_bits(5, 0, 1),
          name="conv_1")(x2)
    else:
      x2 = keras.layers.Conv2D(16, 2, 2, name="conv_1")(x2)

    x = keras.layers.add([x1, x2], name="add")
    if quantize:
      x = QActivation(activation="quantized_relu(8, 2)", name="relu")(x)
    else:
      x = keras.layers.Activation("relu", name="relu")(x)

    if quantize:
      x = QConv2D(
          2, 2, 2,
          kernel_quantizer=quantizers.quantized_bits(5, 0, 1),
          bias_quantizer=quantizers.quantized_bits(5, 0, 1),
          name="conv_2")(x)
    else:
      x = keras.layers.Conv2D(2, 2, 2, name="conv_2")(x)

    model = keras.Model(inputs=[input1, input2], outputs=[x])
    return model

  def get_qenergy(model, qenergy_config, for_reference):
    q = run_qtools.QTools(
        model, process=qenergy_config["process"],
        source_quantizers=qenergy_config["reference_internal"],
        is_inference=qenergy_config["trained_model"],
        weights_path=None,
        keras_quantizer=qenergy_config["reference_internal"],
        keras_accumulator=qenergy_config["reference_accumulator"],
        for_reference=for_reference)

    # caculate energy of the derived data type map.
    energy_dict = q.pe(
        weights_on_memory=qenergy_config["parameters_on_memory"],
        activations_on_memory=qenergy_config["activations_on_memory"],
        min_sram_size=qenergy_config["min_sram_size"],
        rd_wr_on_io=qenergy_config["rd_wr_on_io"])

    total_energy = q.extract_energy_sum(qtools_settings.cfg.include_energy,
                                        energy_dict)

    return q, total_energy

  qenergy_config = {
      "trained_model": True,
      "delta_p": 8.0,
      "delta_n": 8.0,
      "rate": 2.0,
      "stress": 1.0,
      "process": "horowitz",
      "parameters_on_memory": "sram",
      "activations_on_memory": "sram",
      "rd_wr_on_io": False,
      "min_sram_size": 0,
      "source_quantizers": ["quantizers.quantized_bits(8, 0, 1)"],
      "reference_internal": "int8",
      "reference_accumulator": "int32"
  }

  float_model = get_model(quantize=False)
  quantized_model = get_model(quantize=True)

  _, float_reference_energy = get_qenergy(
      float_model, qenergy_config, for_reference=False)
  _, float_trial_energy = get_qenergy(
      float_model, qenergy_config, for_reference=True)
  _, quantized_reference_energy = get_qenergy(
      quantized_model, qenergy_config, for_reference=True)

  assert float_reference_energy == quantized_reference_energy
  assert float_reference_energy == float_trial_energy


def test_auto_po2():
  def gen_model(img_shape):
    img_input = x = keras.Input(shape=img_shape)
    x = QConv2D(
        filters=5, kernel_size=4, strides=4,
        kernel_quantizer=quantizers.quantized_bits(8, 3, alpha="auto_po2"),
        bias_quantizer=quantizers.quantized_bits(8, 3),
        name="conv")(x)
    x = QActivation(activation=quantizers.quantized_relu(4, 0), name="act")(x)
    x = keras.layers.Flatten(name="flatten")(x)
    x = QDense(5,
               kernel_quantizer=quantizers.quantized_bits(
                   8, 0, alpha="auto_po2"),
               bias_quantizer=quantizers.quantized_bits(8, 3),
               name="dense")(x)
    model = keras.Model(inputs=img_input, outputs=[x])
    return model

  model = gen_model((32, 32, 3,))
  model.compile(loss="mse", run_eagerly=True)
  model.layers[1].quantizers[0].scale = tf.constant(
      [[[[0.0625, 0.0625, 0.0625, 0.0625, 0.03125]]]])
  model.layers[4].quantizers[0].scale = tf.constant([[0.5, 0.5, 1, 0.5, 0.25]])
  input_quantizers = [
      quantizers.quantized_bits(bits=8, integer=0, keep_negative=False)
  ]
  dtype_dict = run(model, input_quantizers)
  multiplier = dtype_dict["conv"]["multiplier"]
  assert multiplier["quantizer_type"] == "quantized_bits"
  # Original multiplier has 16 bits(16=8+8) and 3 int_bits
  # Modified multiplier has bits = max_fractional_bits + max_int_bits
  #                             = bits + max_shift - min_shift
  # max_shift = log2(0.0625) = -4  min_shift=log2(0.03125) = -5
  # Therefore modified multiplier bits = 17
  assert multiplier["bits"] == 17
  # Modified multiplier int_bits = int_bits + max_shift = 3 - 4 = -1
  # Because in datatype map we add int_bits with 1 extra sign bit, therefore
  # we expect to see multiplier["int_bits"] == 0.
  assert multiplier["int_bits"] == 0
  multiplier = dtype_dict["dense"]["multiplier"]
  assert multiplier["quantizer_type"] == "quantized_bits"
  assert multiplier["bits"] == 14
  assert multiplier["int_bits"] == 1


if __name__ == "__main__":
  pytest.main([__file__])
