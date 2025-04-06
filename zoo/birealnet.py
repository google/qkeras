# Copyright 2021 Loro Francesco
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__author__ = "Francesco Loro"
__email__ = "francesco.official@gmail.com"
__supervisor__ = "Danilo Pau"
__email__ = "danilo.pau@st.com"

# Download pretrained weight from:
# Birealnet -> https://drive.google.com/file/d/1BuDYhydNAy-sFdvoh24gJbpA0zRDvGVq/view?usp=sharing

import qkeras as q
import tensorflow as tf
import larq as lq
from utils import compare_network, create_random_dataset, dump_network_to_json

# Define path to the pre-trained weights
PATH_BIREALNET = "weights/birealnet_weights.h5"
BIREALNET_NAME = "biRealNet"


class BirealNet:
  """Class to create and load weights of: biRealNet

  Attributes:
        network_name: Name of the network
  """

  def __init__(self):
    self.__weights_path = PATH_BIREALNET
    self.network_name = BIREALNET_NAME

  @staticmethod
  def add_qkeras_residual_block(given_model, filters_num):
    """Adds a sequence of layers to the given model

    Add a sequence of: Activation quantization, Quantized Conv2D

    Args:
      given_model: model where to add the sequence
      filters_num: number of filters for Cov2D
      strides: strides for Conv2D

    Returns:
      Given Model plus the sequence
    """
    x = q.QActivation("binary(alpha=1)")(given_model)
    x = q.QConv2D(filters_num, (3, 3), padding="same",
                  kernel_quantizer="binary(alpha=1)", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    return tf.keras.layers.add([given_model, x])

  @staticmethod
  def add_qkeras_connection_block(given_model, filters_num):
    """Adds a sequence of layers to the given model

    Adds two sequences one of Activation quantization, Quantized Conv2D,
    Batch Normalization the other of Average Pooling2D, Conv2D, BatchNorm

    Args:
      given_model: model where to add the sequence
      filters_num: number of filters for Conv2D

    Returns:
      Given Model plus the sequence
    """
    shortcut = given_model
    shortcut = tf.keras.layers.AvgPool2D(2, strides=2, padding="same")(shortcut)
    shortcut = tf.keras.layers.Conv2D(filters_num, (1, 1),
                                      kernel_initializer="glorot_normal",
                                      use_bias=False)(shortcut)
    shortcut = tf.keras.layers.BatchNormalization(momentum=0.8)(shortcut)
    x = q.QActivation("binary")(given_model)
    x = q.QConv2D(filters_num, (3, 3), strides=(2, 2),
                  padding="same", use_bias=False,
                  kernel_quantizer="binary(alpha=1)")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    return tf.keras.layers.add([x, shortcut])

  @staticmethod
  def add_larq_residual_block(given_model, features):
    """Same method of add_qkeras_residual_block but for a larq network
    """
    x = lq.layers.QuantConv2D(features, (3, 3), padding="same", use_bias=False,
                              input_quantizer="approx_sign",
                              kernel_quantizer=
                              "magnitude_aware_sign",
                              kernel_constraint="weight_clip")(given_model)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    return tf.keras.layers.add([given_model, x])

  @staticmethod
  def add_larq_connection_block(given_model, filters_num):
    """Same method of add_qkeras_connection_block but for a larq network
    """
    shortcut = given_model
    shortcut = tf.keras.layers.AvgPool2D(2, strides=2, padding="same")(shortcut)
    shortcut = tf.keras.layers.Conv2D(filters_num, (1, 1),
                                      kernel_initializer="glorot_normal",
                                      use_bias=False)(shortcut)
    shortcut = tf.keras.layers.BatchNormalization(momentum=0.8)(shortcut)
    x = lq.layers.QuantConv2D(filters_num, (3, 3), strides=(2, 2),
                              padding="same", use_bias=False,
                              input_quantizer="approx_sign",
                              kernel_quantizer=
                              "magnitude_aware_sign",
                              kernel_constraint="weight_clip")(given_model)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    return tf.keras.layers.add([x, shortcut])

  def build(self):
    """Builds the model

    Returns:
      qkeras and larq models
    """
    qkeras_network = self.build_qkeras_birealnet()
    print("\nQKeras network successfully created")
    larq_network = self.build_larq_birealnet()
    print("Larq network successfully created")
    return qkeras_network, larq_network

  def build_qkeras_birealnet(self):
    """Builds the qkeras version of the birealnet

    Returns:
      Qkeras model of the birealnet
    """
    input_layer = tf.keras.Input(shape=(224, 224, 3))
    qkeras_biRealNet = tf.keras.layers.Conv2D(64, (7, 7), strides=2,
                                              padding="same",
                                              use_bias=False)(input_layer)
    qkeras_biRealNet = tf.keras.layers.BatchNormalization(momentum=0.8)(
      qkeras_biRealNet)
    qkeras_biRealNet = tf.keras.layers.MaxPool2D((3, 3), strides=2,
                                                 padding="same") \
      (qkeras_biRealNet)

    for _ in range(0, 4):
      qkeras_biRealNet = self.add_larq_residual_block(qkeras_biRealNet, 64)
    qkeras_biRealNet = self.add_larq_connection_block(qkeras_biRealNet, 128)
    for _ in range(0, 3):
      qkeras_biRealNet = self.add_larq_residual_block(qkeras_biRealNet, 128)
    qkeras_biRealNet = self.add_larq_connection_block(qkeras_biRealNet, 256)
    for _ in range(0, 3):
      qkeras_biRealNet = self.add_larq_residual_block(qkeras_biRealNet, 256)
    qkeras_biRealNet = self.add_larq_connection_block(qkeras_biRealNet, 512)
    for _ in range(0, 3):
      qkeras_biRealNet = self.add_larq_residual_block(qkeras_biRealNet, 512)

    qkeras_biRealNet = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(
      qkeras_biRealNet)
    qkeras_biRealNet = tf.keras.layers.Flatten()(qkeras_biRealNet)
    qkeras_biRealNet = tf.keras.layers.Dense(1000)(qkeras_biRealNet)
    qkeras_biRealNet = tf.keras.layers.Activation("softmax", dtype="float32")(
      qkeras_biRealNet)
    qkeras_biRealNet = tf.keras.Model(inputs=input_layer,
                                      outputs=qkeras_biRealNet)
    qkeras_biRealNet.load_weights(PATH_BIREALNET)
    return qkeras_biRealNet

  def build_larq_birealnet(self):
    """Builds the larq version of the birealnet

    Returns:
      Larq model of the birealnet
    """
    input_layer = tf.keras.Input(shape=(224, 224, 3))
    larq_biRealNet = tf.keras.layers.Conv2D(64, (7, 7), strides=2,
                                            padding="same",
                                            use_bias=False)(input_layer)
    larq_biRealNet = tf.keras.layers.BatchNormalization(momentum=0.8)(
      larq_biRealNet)
    larq_biRealNet = tf.keras.layers.MaxPool2D((3, 3), strides=2,
                                               padding="same") \
      (larq_biRealNet)

    for _ in range(0, 4):
      larq_biRealNet = self.add_larq_residual_block(larq_biRealNet, 64)
    larq_biRealNet = self.add_larq_connection_block(larq_biRealNet, 128)
    for _ in range(0, 3):
      larq_biRealNet = self.add_larq_residual_block(larq_biRealNet, 128)
    larq_biRealNet = self.add_larq_connection_block(larq_biRealNet, 256)
    for _ in range(0, 3):
      larq_biRealNet = self.add_larq_residual_block(larq_biRealNet, 256)
    larq_biRealNet = self.add_larq_connection_block(larq_biRealNet, 512)
    for _ in range(0, 3):
      larq_biRealNet = self.add_larq_residual_block(larq_biRealNet, 512)

    larq_biRealNet = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(
      larq_biRealNet)
    larq_biRealNet = tf.keras.layers.Flatten()(larq_biRealNet)
    larq_biRealNet = tf.keras.layers.Dense(1000)(larq_biRealNet)
    larq_biRealNet = tf.keras.layers.Activation("softmax", dtype="float32")(
      larq_biRealNet)
    larq_biRealNet = tf.keras.Model(inputs=input_layer, outputs=larq_biRealNet)
    larq_biRealNet.load_weights(PATH_BIREALNET)
    return larq_biRealNet


if __name__ == "__main__":
  # Create a random dataset with 100 samples
  random_data = create_random_dataset(100)

  network = BirealNet()
  qkeras_network, larq_network = network.build()
  # Compare mean MSE and Absolute error of the the networks
  compare_network(qkeras_network=qkeras_network, larq_network=larq_network,
                  dataset=random_data, network_name=BIREALNET_NAME)
  dump_network_to_json(qkeras_network=qkeras_network,
                       larq_network=larq_network,
                       network_name=BIREALNET_NAME)
