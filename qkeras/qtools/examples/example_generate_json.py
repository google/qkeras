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
"""Example code to generate weight and MAC sizes in a json file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.keras as keras

from qkeras import QActivation
from qkeras import QDense
from qkeras import quantizers
from qkeras.qtools import run_qtools


def hybrid_model():
  """hybrid model that mixes qkeras and keras layers."""

  x = x_in = keras.layers.Input((784,), name="input")
  x = keras.layers.Dense(300, name="d0")(x)
  x = keras.layers.Activation("relu", name="d0_act")(x)
  x = QDense(100, kernel_quantizer=quantizers.quantized_po2(4),
             bias_quantizer=quantizers.quantized_po2(4),
             name="d1")(x)
  x = QActivation("quantized_relu(4,0)", name="d1_qr4")(x)
  x = QDense(
      10, kernel_quantizer=quantizers.quantized_po2(4),
      bias_quantizer=quantizers.quantized_po2(4),
      name="d2")(x)
  x = keras.layers.Activation("softmax", name="softmax")(x)

  return keras.Model(inputs=[x_in], outputs=[x])


def generate_json(in_model):
  """example to generate data type map for a given model.

  Args:
    in_model: qkeras model object

  Usage:
    input_quantizer_list:
      A list of input quantizers for the model. It could be in the form of:
        1. a list of quantizers, each quantizer for each one of the model inputs
        2. one single quantizer, which will be used for all of the model inputs
        3. None. Default input quantizer defined in config_xxx.py will be used
        for all of the model inputs

    for_reference: get energy for a reference model/trial model
      1. True: get baseline energy for a given model. Use keras_quantizer/keras_
        accumulator (or default_interm_quantizer in config_xxx.py if keras_
        quantizer/keras_accumulator not given) to quantizer all layers in a
        model in order to calculate its energy. It servers the purpose of
        setting up a baseline energy for a given model architecture.
      2. False: get "real" energy for a given model use user-specified
        quantizers. For layers that are not quantized (keras layer) or have no
        user-specified quantizers (qkeras layers without quantizers specified),
        keras_quantizer and keras_accumulator(or default_interm_quantizer in
        config_xxx.py if keras_quantizer/keras_accumulator not given)
        will be used as their quantizers.

     process: technology process to use in configuration (horowitz, ...)

     weights_path: absolute path to the model weights

     is_inference: whether model has been trained already, which is needed to
         compute tighter bounds for QBatchNormalization Power estimation

     Other parameters (defined in config_xxx.py):
       1. "default_source_quantizer" is used as default input quantizer
          if user do not specify any input quantizers,
       2. "default_interm_quantizer": is used as default quantizer for any
          intermediate variables such as multiplier, accumulator, weight/bias
          in a qkeras layer if user do not secifiy the corresponding variable
       3. process_name: energy calculation parameters for different processes.
          "horowitz" is the process we use by default.
       4. "include_energy": what energy to include at each layer
          when calculation the total energy of the entire model.
          "parameters": memory access energy for loading model parameters.
          "inputs": memory access energy to reading inputs
          "outputs": memory access energy for writing outputs
          "op_cost": operation energy for multiplication and accumulation
  """

  input_quantizer_list = [quantizers.quantized_bits(8, 0, 1)]
  reference_internal = "int8"
  reference_accumulator = "int32"

  # generate QTools object which contains model data type map in json format
  q = run_qtools.QTools(
      in_model,
      # energy calculation using a given process
      process="horowitz",
      # quantizers for model inputs
      source_quantizers=input_quantizer_list,
      # training or inference with a pre-trained model
      is_inference=False,
      # path to pre-trained model weights
      weights_path=None,
      # keras_quantizer to quantize weight/bias in non-quantized keras layers
      keras_quantizer=reference_internal,
      # keras_accumulator to quantize MAC in un-quantized keras layers
      keras_accumulator=reference_accumulator,
      # calculating baseline energy or not
      for_reference=False)

  # print data type map
  q.qtools_stats_print()

  # dump the layer data map to a json file
  # json_name = "output.json"
  # q.qtools_stats_to_json(json_name)


if __name__ == "__main__":
  model = hybrid_model()
  model.summary()

  generate_json(model)
