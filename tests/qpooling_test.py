# Copyright 2021 Google LLC
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
"""Test layers from qpooling.py."""
import numpy as np
from numpy.testing import assert_allclose
from numpy.testing import assert_raises
from numpy.testing import assert_equal
import pytest
import logging
import tempfile
import os
import tensorflow.compat.v2 as tf

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.backend import clear_session

from qkeras import QAveragePooling2D
from qkeras import QGlobalAveragePooling2D
from qkeras import quantized_bits
from qkeras import binary
from qkeras import ternary
from qkeras.utils import model_save_quantized_weights
from qkeras.utils import quantized_model_from_json
from qkeras.utils import load_qmodel
from qkeras.utils import model_quantize
from qkeras import print_qstats
from qkeras.qtools import qgraph
from qkeras.qtools import generate_layer_data_type_map
from qkeras.qtools import interface


@pytest.mark.parametrize(
    ('pooling, input_size, pool_size, strides, padding, data_format,'
     'average_quantizer, activation_quantizer,  y'), [
         ('QAveragePooling2D', (4, 4, 3), (2, 2), (2, 2), 'valid',
          'channels_last', quantized_bits(4, 0, 1), quantized_bits(4, 0, 1),
          np.array([[[[0.375, 0.625, 0.375], [0.25, 0.75, 0.5]],
                     [[0.375, 0.25, 0.625], [0.625, 0.5, 0.375]]],
                    [[[0.375, 0.375, 0.5], [0.375, 0.5, 0.625]],
                     [[0.75, 0.625, 0.5], [0.5, 0.5, 0.75]]]]).astype(
                         np.float16)),
         ('QAveragePooling2D', (4, 4, 3), (3, 3), (3, 3), 'valid',
          'channels_last', quantized_bits(4, 0, 1), quantized_bits(4, 0, 1),
          np.array([[[[0.375, 0.625, 0.625]]], [[[0.625, 0.5, 0.625]]]]).astype(
              np.float16)),
         ('QGlobalAveragePooling2D', (4, 4, 3), (2, 2), (2, 2), 'valid',
          'channels_last', quantized_bits(10, 0, 1), quantized_bits(4, 0, 1),
          np.array([[0.5, 0.5, 0.375], [0.5, 0.5, 0.625]]).astype(np.float16)),
         ('QAveragePooling2D', (4, 4, 3), (2, 2), (3, 3), 'valid',
          'channels_last', quantized_bits(4, 0, 1), quantized_bits(4, 0, 1),
          np.array([[[[0.375, 0.625, 0.375]]], [[[0.375, 0.375, 0.5]]]]).astype(
              np.float16)),
         ('QAveragePooling2D', (4, 4, 3), (2, 2), (3, 3), 'same',
          'channels_last', quantized_bits(4, 0, 1), quantized_bits(4, 0, 1),
          np.array([[[[0.375, 0.625, 0.375], [0.375, 0.75, 0.25]],
                     [[0.75, 0.25, 0.375], [0.75, 0.75, 0.25]]],
                    [[[0.375, 0.375, 0.5], [0.25, 0.625, 0.5]],
                     [[0.625, 0.625, 0.5], [0.625, 0.625, 0.875]]]]).astype(
                         np.float16)),
         ('QAveragePooling2D', (4, 4, 3), (2, 2),
          (2, 2), 'valid', 'channels_first', quantized_bits(
              4, 0, 1), quantized_bits(4, 0, 1), None),
     ])
def test_q_average_pooling(pooling, input_size, pool_size, strides, padding,
                           data_format, average_quantizer,
                           activation_quantizer, y):
  """q_average_pooling test utility."""

  np.random.seed(33)

  x = Input(input_size)
  xin = x
  if pooling == 'QAveragePooling2D':
    x = QAveragePooling2D(
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        average_quantizer=average_quantizer,
        activation=activation_quantizer,
        name='qpooling')(x)
  else:
    x = QGlobalAveragePooling2D(
        data_format=data_format,
        average_quantizer=average_quantizer,
        activation=activation_quantizer,
        name='qpooling')(
            x)
  model = Model(inputs=xin, outputs=x)

  # Prints qstats to make sure it works with Conv1D layer
  print_qstats(model)

  size = (2,) + input_size
  inputs = np.random.rand(size[0], size[1], size[2], size[3])

  if data_format == 'channels_first':
    assert_raises(tf.errors.InvalidArgumentError, model.predict, inputs)
  else:
    p = model.predict(inputs).astype(np.float16)
    assert_allclose(p, y, rtol=1e-4)

    # Reloads the model to ensure saving/loading works
    json_string = model.to_json()
    clear_session()
    reload_model = quantized_model_from_json(json_string)
    p = reload_model.predict(inputs).astype(np.float16)
    assert_allclose(p, y, rtol=1e-4)

    # Saves the model as an h5 file using Keras's model.save()
    fd, fname = tempfile.mkstemp(".h5")
    model.save(fname)
    del model  # Delete the existing model

    # Returns a compiled model identical to the previous one
    loaded_model = load_qmodel(fname)

    # Cleans the created h5 file after loading the model
    os.close(fd)
    os.remove(fname)

    # Applys quantizer to weights
    model_save_quantized_weights(loaded_model)
    p = loaded_model.predict(inputs).astype(np.float16)
    assert_allclose(p, y, rtol=1e-4)


def test_qpooling_in_model_quantize():
  input_size = (16, 16, 3)
  pool_size = (2, 2)

  x = Input(input_size)
  xin = x
  x = AveragePooling2D(pool_size=pool_size, name="pooling")(x)
  x = GlobalAveragePooling2D(name="global_pooling")(x)
  model = Model(inputs=xin, outputs=x)

  quantize_config = {
      "QAveragePooling2D": {
          "average_quantizer": "binary",
          "activation_quantizer": "binary"
      },
      "QGlobalAveragePooling2D": {
          "average_quantizer": "quantized_bits(4, 0, 1)",
          "activation_quantizer": "ternary"
      }
  }

  qmodel = model_quantize(model, quantize_config, 4)
  print_qstats(qmodel)
  assert_equal(str(qmodel.layers[1].average_quantizer_internal), "binary()")
  assert_equal(str(qmodel.layers[1].activation), "binary()")
  assert_equal(
      str(qmodel.layers[2].average_quantizer_internal), "quantized_bits(4,0,1)")
  assert_equal(str(qmodel.layers[2].activation), "ternary()")


def test_qpooling_in_qtools():
  input_size = (16, 16, 3)
  pool_size = (2, 2)
  input_quantizers = [quantized_bits(8, 0, 1)]
  is_inference = False

  x = Input(input_size)
  xin = x
  x = QAveragePooling2D(
      pool_size=pool_size,
      average_quantizer=binary(),
      activation=quantized_bits(4, 0, 1),
      name="pooling")(
          x)
  x = QGlobalAveragePooling2D(
      average_quantizer=quantized_bits(4, 0, 1),
      activation=ternary(),
      name="global_pooling")(
          x)
  model = Model(inputs=xin, outputs=x)

  (graph, source_quantizer_list) = qgraph.CreateGraph(
      model, input_quantizers)

  qgraph.GraphPropagateActivationsToEdges(graph)

  layer_map = generate_layer_data_type_map.generate_layer_data_type_map(
      graph, source_quantizer_list, is_inference)

  dtype_dict = interface.map_to_json(layer_map)

  # Checks the QAveragePpooling layer datatype
  multiplier = dtype_dict["pooling"]["pool_avg_multiplier"]
  accumulator = dtype_dict["pooling"]["pool_sum_accumulator"]
  average_quantizer  = dtype_dict["pooling"]["average_quantizer"]
  output = dtype_dict["pooling"]["output_quantizer"]

  assert_equal(multiplier["quantizer_type"], "quantized_bits")
  assert_equal(multiplier["bits"], 10)
  assert_equal(multiplier["int_bits"], 3)
  assert_equal(multiplier["is_signed"], 1)
  assert_equal(multiplier["op_type"], "mux")

  assert_equal(accumulator["quantizer_type"], "quantized_bits")
  assert_equal(accumulator["bits"], 10)
  assert_equal(accumulator["int_bits"], 3)
  assert_equal(accumulator["is_signed"], 1)
  assert_equal(accumulator["op_type"], "add")

  assert_equal(output["quantizer_type"], "quantized_bits")
  assert_equal(output["bits"], 4)
  assert_equal(output["int_bits"], 1)
  assert_equal(output["is_signed"], 1)

  assert_equal(average_quantizer["quantizer_type"], "binary")
  assert_equal(average_quantizer["bits"], 1)
  assert_equal(average_quantizer["int_bits"], 1)
  assert_equal(average_quantizer["is_signed"], 1)

  # Checks the QGlobalAveragePooling layer datatype
  multiplier = dtype_dict["global_pooling"]["pool_avg_multiplier"]
  accumulator = dtype_dict["global_pooling"]["pool_sum_accumulator"]
  average_quantizer  = dtype_dict["global_pooling"]["average_quantizer"]
  output = dtype_dict["global_pooling"]["output_quantizer"]

  assert_equal(multiplier["quantizer_type"], "quantized_bits")
  assert_equal(multiplier["bits"], 13)
  assert_equal(multiplier["int_bits"], 7)
  assert_equal(multiplier["is_signed"], 1)
  assert_equal(multiplier["op_type"], "mul")

  assert_equal(accumulator["quantizer_type"], "quantized_bits")
  assert_equal(accumulator["bits"], 10)
  assert_equal(accumulator["int_bits"], 7)
  assert_equal(accumulator["is_signed"], 1)
  assert_equal(accumulator["op_type"], "add")

  assert_equal(output["quantizer_type"], "ternary")
  assert_equal(output["bits"], 2)
  assert_equal(output["int_bits"], 2)
  assert_equal(output["is_signed"], 1)

  assert_equal(average_quantizer["quantizer_type"], "quantized_bits")
  assert_equal(average_quantizer["bits"], 4)
  assert_equal(average_quantizer["int_bits"], 1)
  assert_equal(average_quantizer["is_signed"], 1)


def test_QAveragePooling_output():
  # Checks if the output of QAveragePooling layer with average_quantizer
  # is correct.
  x = np.ones(shape=(2, 6, 6, 1))
  x[0, 0, :, :] = 0
  x = tf.constant(x)

  y = QAveragePooling2D(
      pool_size=(3, 3),
      strides=3,
      padding="valid",
      average_quantizer="quantized_bits(8, 1, 1)")(x)
  y = y.numpy()
  assert np.all(y == [[[[0.65625], [0.65625]], [[0.984375], [0.984375]]],
                      [[[0.984375], [0.984375]], [[0.984375], [0.984375]]]])


def test_QGlobalAveragePooling_output():
  # Checks if the output of QGlobalAveragePooling layer with average_quantizer
  # is correct.
  x = np.ones(shape=(2, 3, 3, 2))
  x[0, 0, 1, :] = 0
  x = tf.constant(x)
  y = QGlobalAveragePooling2D(average_quantizer="quantized_bits(8, 1, 1)")(x)
  y = y.numpy()
  assert np.all(y == np.array([[0.875, 0.875], [0.984375, 0.984375]]))


if __name__ == "__main__":
  pytest.main([__file__])
