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
"""Tests automatic conversion of keras model to qkeras."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from qkeras.estimate import print_qstats
from qkeras.utils import model_quantize
from qkeras.utils import quantized_model_dump

x0 = x_in0 = Input((28, 28, 1), name="input0")
x1 = x_in1 = Input((28, 28, 1), name="input1")
x = Concatenate(name="concat")([x0, x1])
x = Conv2D(128, (3, 3), strides=1, name="conv2d_0_m")(x)
x = Activation("relu", name="act0_m")(x)
x = MaxPooling2D(2, 2, name="mp_0")(x)
x = Conv2D(256, (3, 3), strides=1, name="conv2d_1_m")(x)
x = Activation("relu", name="act1_m")(x)
x = MaxPooling2D(2, 2, name="mp_1")(x)
x = Conv2D(128, (3, 3), strides=1, name="conv2d_2_m")(x)
x = Activation("relu", name="act2_m")(x)
x = MaxPooling2D(2, 2, name="mp_2")(x)
x = Flatten()(x)
x = Dense(10, name="dense")(x)
x = Activation("softmax", name="softmax")(x)

model = Model(inputs=[x_in0, x_in1], outputs=[x])
model.summary()

q_dict = {
    "conv2d_0_m": {
        "kernel_quantizer": "binary()",
        "bias_quantizer": "quantized_bits(4,0,1)"
    },
    "conv2d_1_m": {
        "kernel_quantizer": "ternary()",
        "bias_quantizer": "quantized_bits(4,0,1)"
    },
    "act2_m": "quantized_relu(6,2)",
    "QActivation": {
        "relu": "quantized_relu(4,0)"
    },
    "QConv2D": {
        "kernel_quantizer": "quantized_bits(4,0,1)",
        "bias_quantizer": "quantized_bits(4,0,1)"
    },
    "QDense": {
        "kernel_quantizer": "quantized_bits(3,0,1)",
        "bias_quantizer": "quantized_bits(3,0,1)"
    }
}

qmodel = model_quantize(model, q_dict, 4)

qmodel.summary()

print_qstats(qmodel)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test_arr = [x_test[0:10,:], x_test[0:10,:]]

quantized_model_dump(
    qmodel, x_test_arr,
    layers_to_dump=["input0", "input1", "act2_m", "act1_m", "act0_m"])

