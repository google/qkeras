# ==============================================================================
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

import pytest
from tensorflow.keras.layers import *   # pylint: disable=wildcard-import
from tensorflow.keras.models import Model
from qkeras import *   # pylint: disable=wildcard-import
from qkeras.autoqkeras.forgiving_metrics import ForgivingFactorBits   # pylint: disable=line-too-long
from qkeras.utils import model_quantize


def get_model():
  """Returns sample model."""
  xi = Input((28, 28, 1), name="input")   # pylint: disable=undefined-variable
  x = Conv2D(32, 3, strides=1, padding="same", name="c1")(xi)   # pylint: disable=undefined-variable
  x = BatchNormalization(name="b1")(x)   # pylint: disable=undefined-variable
  x = Activation("relu", name="a1")(x)   # pylint: disable=undefined-variable
  x = MaxPooling2D(2, 2, name="mp1")(x)   # pylint: disable=undefined-variable
  x = QConv2D(32, 3, kernel_quantizer="binary", bias_quantizer="binary",   # pylint: disable=undefined-variable
              strides=1, padding="same", name="c2")(x)
  x = QBatchNormalization(name="b2")(x)   # pylint: disable=undefined-variable
  x = QActivation("binary", name="a2")(x)   # pylint: disable=undefined-variable
  x = MaxPooling2D(2, 2, name="mp2")(x)   # pylint: disable=undefined-variable
  x = QConv2D(32, 3, kernel_quantizer="ternary", bias_quantizer="ternary",   # pylint: disable=undefined-variable
              strides=1, padding="same", activation="binary", name="c3")(x)
  x = Flatten(name="flatten")(x)   # pylint: disable=undefined-variable
  x = Dense(1, name="dense", activation="softmax")(x)   # pylint: disable=undefined-variable

  model = Model(inputs=xi, outputs=x)

  return model


def test_forgiving_factor_bits():
  """Tests forgiving factor bits."""
  delta_p = 8.0
  delta_n = 8.0
  rate = 2.0
  stress = 1.0
  input_bits = 8
  output_bits = 8
  ref_bits = 8

  config = {
      "QDense": ["parameters", "activations"],
      "Dense": ["parameters", "activations"],
      "QConv2D": ["parameters", "activations"],
      "Conv2D": ["parameters", "activations"],
      "DepthwiseConv2D": ["parameters", "activations"],
      "QDepthwiseConv2D": ["parameters", "activations"],
      "Activation": ["activations"],
      "QActivation": ["activations"],
      "QBatchNormalization": ["parameters"],
      "BatchNormalization": ["parameters"],
      "default": ["activations"],
  }

  model = get_model()

  ffb = ForgivingFactorBits(
      delta_p, delta_n, rate, stress,
      input_bits, output_bits, ref_bits,
      config
  )

  cached_result = ffb.compute_model_size(model)
  ref_size = cached_result[0]
  ref_p = cached_result[1]
  ref_a = cached_result[2]

  assert ref_size == 258544
  assert ref_p == 43720
  assert ref_a == 214824


def test_new_forgiving_factor():
  """Tests forgiving factor."""
  delta_p = 8.0
  delta_n = 8.0
  rate = 2.0
  stress = 1.0
  input_bits = 8
  output_bits = 8
  ref_bits = 8

  config = {
      "QDense": ["parameters", "activations"],
      "Dense": ["parameters", "activations"],
      "QConv2D": ["parameters", "activations"],
      "Conv2D": ["parameters", "activations"],
      "DepthwiseConv2D": ["parameters", "activations"],
      "QDepthwiseConv2D": ["parameters", "activations"],
      "Activation": ["activations"],
      "QActivation": ["activations"],
      "QBatchNormalization": ["parameters"],
      "BatchNormalization": ["parameters"],
      "default": ["activations"]
  }

  model = get_model()

  ffb = ForgivingFactorBits(
      delta_p, delta_n, rate, stress,
      input_bits, output_bits, ref_bits,
      config
  )

  cached_result = ffb.compute_model_size(model)
  ref_size = cached_result[0]
  ref_p = cached_result[1]
  ref_a = cached_result[2]
  ref_size_dict = cached_result[3]

  assert ref_size == 258544
  assert ref_p == 43720
  assert ref_a == 214824

  q_dict = {
      "c1": {
          "kernel_quantizer": "binary",
          "bias_quantizer": "quantized_bits(4)"
      }
  }

  q_model = model_quantize(model, q_dict, 4)

  cached_result = ffb.compute_model_size(q_model)
  trial_size_dict = cached_result[3]

  for name in trial_size_dict:
    if name != "c1":
      assert trial_size_dict[name] == ref_size_dict[name]
  assert trial_size_dict["c1"]["parameters"] == 416

if __name__ == "__main__":
  pytest.main([__file__])



