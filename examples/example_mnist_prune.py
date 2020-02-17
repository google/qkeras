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
"""Example of mnist model with pruning.
   Adapted from TF model optimization example."""

import tempfile
import numpy as np

import tensorflow.keras.backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import save_model
from tensorflow.keras.utils import to_categorical

from qkeras import QActivation
from qkeras import QDense
from qkeras import QConv2D
from qkeras import quantized_bits
from qkeras.utils import load_qmodel
from qkeras.utils import print_model_sparsity

from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule


batch_size = 128
num_classes = 10
epochs = 12

prune_whole_model = True # Prune whole model or just specified layers


def build_model(input_shape):
    x = x_in = Input(shape=input_shape, name="input")
    x = QConv2D(
        32, (2, 2), strides=(2,2),
        kernel_quantizer=quantized_bits(4,0,1),
        bias_quantizer=quantized_bits(4,0,1),
        name="conv2d_0_m")(x)
    x = QActivation("quantized_relu(4,0)", name="act0_m")(x)
    x = QConv2D(
        64, (3, 3), strides=(2,2),
        kernel_quantizer=quantized_bits(4,0,1),
        bias_quantizer=quantized_bits(4,0,1),
        name="conv2d_1_m")(x)
    x = QActivation("quantized_relu(4,0)", name="act1_m")(x)
    x = QConv2D(
        64, (2, 2), strides=(2,2),
        kernel_quantizer=quantized_bits(4,0,1),
        bias_quantizer=quantized_bits(4,0,1),
        name="conv2d_2_m")(x)
    x = QActivation("quantized_relu(4,0)", name="act2_m")(x)
    x = Flatten()(x)
    x = QDense(num_classes, kernel_quantizer=quantized_bits(4,0,1),
               bias_quantizer=quantized_bits(4,0,1),
               name="dense")(x)
    x = Activation("softmax", name="softmax")(x)

    model = Model(inputs=[x_in], outputs=[x])
    return model


def build_layerwise_model(input_shape, **pruning_params):
    return Sequential([
        prune.prune_low_magnitude(
            QConv2D(
                32, (2, 2), strides=(2,2),
                kernel_quantizer=quantized_bits(4,0,1),
                bias_quantizer=quantized_bits(4,0,1),
                name="conv2d_0_m"),
            input_shape=input_shape,
            **pruning_params),
        QActivation("quantized_relu(4,0)", name="act0_m"),
        prune.prune_low_magnitude(
            QConv2D(
                64, (3, 3), strides=(2,2),
                kernel_quantizer=quantized_bits(4,0,1),
                bias_quantizer=quantized_bits(4,0,1),
                name="conv2d_1_m"),
            **pruning_params),
        QActivation("quantized_relu(4,0)", name="act1_m"),
        prune.prune_low_magnitude(
            QConv2D(
                64, (2, 2), strides=(2,2),
                kernel_quantizer=quantized_bits(4,0,1),
                bias_quantizer=quantized_bits(4,0,1),
                name="conv2d_2_m"),
            **pruning_params),
        QActivation("quantized_relu(4,0)", name="act2_m"),
        Flatten(),
        prune.prune_low_magnitude(
            QDense(
                num_classes, kernel_quantizer=quantized_bits(4,0,1),
                bias_quantizer=quantized_bits(4,0,1),
                name="dense"),
            **pruning_params),
        Activation("softmax", name="softmax")
  ])


def train_and_save(model, x_train, y_train, x_test, y_test):
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"])

    # Print the model summary.
    model.summary()

    # Add a pruning step callback to peg the pruning step to the optimizer's
    # step. Also add a callback to add pruning summaries to tensorboard
    callbacks = [
        pruning_callbacks.UpdatePruningStep(),
        #pruning_callbacks.PruningSummaries(log_dir=tempfile.mkdtemp())
        pruning_callbacks.PruningSummaries(log_dir="/tmp/mnist_prune")
    ]

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    print_model_sparsity(model)

    # Export and import the model. Check that accuracy persists.
    _, keras_file = tempfile.mkstemp(".h5")
    print("Saving model to: ", keras_file)
    save_model(model, keras_file)
    
    print("Reloading model")
    with prune.prune_scope():
        loaded_model = load_qmodel(keras_file)
    score = loaded_model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])


def main():
    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == "channels_first":
      x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
      x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
      input_shape = (1, img_rows, img_cols)
    else:
      x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
      x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
      input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    pruning_params = {
        "pruning_schedule":
            pruning_schedule.ConstantSparsity(0.75, begin_step=2000, frequency=100)
    }
    
    if prune_whole_model:
        model = build_model(input_shape)
        model = prune.prune_low_magnitude(model, **pruning_params)
    else:
        model = build_layerwise_model(input_shape, **pruning_params)

    train_and_save(model, x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    main()