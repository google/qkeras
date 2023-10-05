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

import tempfile
import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
# TODO: Update to new optimizer API
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.utils import to_categorical

from qkeras.autoqkeras import AutoQKerasScheduler

np.random.seed(42)
tf.random.set_seed(42)
tf.config.experimental.enable_op_determinism()


def dense_model():
  """Creates test dense model."""

  x = x_in = Input((4,), name="input")
  x = Dense(20, name="dense_0")(x)
  x = BatchNormalization(name="bn0")(x)
  x = Activation("relu", name="relu_0")(x)
  x = Dense(40, name="dense_1")(x)
  x = BatchNormalization(name="bn1")(x)
  x = Activation("relu", name="relu_1")(x)
  x = Dense(20, name="dense_2")(x)
  x = BatchNormalization(name="bn2")(x)
  x = Activation("relu", name="relu_2")(x)
  x = Dense(3, name="dense")(x)
  x = Activation("softmax", name="softmax")(x)

  model = Model(inputs=x_in, outputs=x)

  # Manually set the weights for each layer. Needed for test determinism.
  for layer in model.layers:
    if isinstance(layer, Dense):
      weights_shape = layer.get_weights()[0].shape
      bias_shape = layer.get_weights()[1].shape
      weights = np.random.RandomState(42).randn(*weights_shape)
      bias = np.random.RandomState(42).randn(*bias_shape)
      layer.set_weights([weights, bias])

  return model

def test_autoqkeras():
  """Tests AutoQKeras scheduler."""

  x_train, y_train = load_iris(return_X_y=True)

  scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
  scaler.fit(x_train)
  x_train = scaler.transform(x_train)

  nb_classes = np.max(y_train) + 1
  y_train = to_categorical(y_train, nb_classes)

  quantization_config = {
      "kernel": {
          "stochastic_ternary": 2,
          "quantized_bits(8,0,1,alpha=1.0)": 8
      },
      "bias": {
          "quantized_bits(4,0,1)": 4
      },
      "activation": {
          "quantized_relu(4,1)": 4
      },
      "linear": {
          "binary": 1
      }
  }

  goal = {
      "type": "energy",
      "params": {
          "delta_p": 8.0,
          "delta_n": 8.0,
          "rate": 2.0,
          "stress": 1.0,
          "process": "horowitz",
          "parameters_on_memory": ["sram", "sram"],
          "activations_on_memory": ["sram", "sram"],
          "rd_wr_on_io": [False, False],
          "min_sram_size": [0, 0],
          "reference_internal": "int8",
          "reference_accumulator": "int32"
      }
  }

  model = dense_model()
  model.summary()
  optimizer = Adam(learning_rate=0.015)
  model.compile(optimizer=optimizer, loss="categorical_crossentropy",
                metrics=["acc"])

  limit = {
      "dense_0": [["stochastic_ternary"], 8, 4],
      "dense": [["quantized_bits(8,0,1,alpha=1.0)"], 8, 4],
      "BatchNormalization": [],
      "Activation": [4]
  }

  run_config = {
      "output_dir": tempfile.mkdtemp(),
      "goal": goal,
      "quantization_config": quantization_config,
      "learning_rate_optimizer": False,
      "transfer_weights": False,
      "mode": "random",
      "seed": 42,
      "limit": limit,
      "tune_filters": "layer",
      "tune_filters_exceptions": "^dense$",
      "max_trials": 1,

      "blocks": [
          "^.*0$",
          "^dense$"
      ],
      "schedule_block": "cost"
  }

  autoqk = AutoQKerasScheduler(model, metrics=["acc"], **run_config)
  autoqk.fit(x_train, y_train, validation_split=0.1, batch_size=150, epochs=4)

  qmodel = autoqk.get_best_model()

  optimizer = Adam(learning_rate=0.015)
  qmodel.compile(optimizer=optimizer, loss="categorical_crossentropy",
                 metrics=["acc"])
  _ = qmodel.fit(x_train, y_train, epochs=5, batch_size=150,
                       validation_split=0.1)

if __name__ == "__main__":
  pytest.main([__file__])

