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
  """
  Class to create and load weights of: biRealNet
  Attributes:
        network_name: Name of the network
  """

  def __init__(self):
    self.__weights_path = PATH_BIREALNET
    self.network_name = BIREALNET_NAME

  @staticmethod
  def add_qkeras_residual_block(model, filters_num, strides=1):
    """
    Add a sequence of: Activation quantization, Quantized Conv2D
    :param model: model where to add the sequence
    :param filters_num: number of filters for Cov2D
    :param strides: strides for Conv2D
    """
    model.add(q.QActivation("binary(alpha=1)"))
    model.add(q.QConv2D(filters_num, (3, 3), strides=strides, padding="same",
                        kernel_quantizer="binary(alpha=1)", use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())

  @staticmethod
  def add_qkeras_connection_block(model, filters_num):
    """
    Add a sequence of: Activation quantization, Quantized Conv2D, reshape,
    Average Pooling, Conv2D, 2x BatchNormalization
    :param model: model where to add the sequence
    :param filters_num: number of filters for Cov2D
    """
    model.add(q.QActivation("binary"))
    model.add(q.QConv2D(filters_num, (3, 3), strides=(2, 2), use_bias=False,
                        padding="same", kernel_quantizer="binary(alpha=1)"))
    # Prepare shapes for reshape layers
    shape_in = model.output_shape[1] * model.output_shape[2] * \
               model.output_shape[3]
    shape_out = (model.output_shape[1], model.output_shape[2],
                 model.output_shape[3] // 2)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Reshape(target_shape=(shape_in, 1)))
    model.add(tf.keras.layers.AvgPool1D(1, strides=2, padding="same"))
    model.add(tf.keras.layers.Reshape(target_shape=shape_out))
    model.add(tf.keras.layers.Conv2D(filters_num, (1, 1), padding="same",
                                     use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.BatchNormalization())

  @staticmethod
  def add_larq_residual_block(model, features, strides=1):
    """
    Same method of add_qkeras_residual_block but for a larq network
    """
    model.add(lq.layers.QuantConv2D(features, (3, 3), strides=strides,
                                    padding="same", use_bias=False,
                                    input_quantizer="approx_sign",
                                    kernel_quantizer=
                                    "magnitude_aware_sign",
                                    kernel_constraint="weight_clip", ))
    model.add(tf.keras.layers.BatchNormalization())

  @staticmethod
  def add_larq_connection_block(model, filters_num):
    """
    Same method of add_qkeras_connection_block but for a larq network
    """
    model.add(
      lq.layers.QuantConv2D(filters_num, (3, 3), strides=(2, 2), use_bias=False,
                            padding="same",
                            input_quantizer="approx_sign",
                            kernel_quantizer="magnitude_aware_sign",
                            kernel_constraint="weight_clip"))

    shape_in = model.output_shape[1] * model.output_shape[2] * \
               model.output_shape[3]
    shape_out = (model.output_shape[1], model.output_shape[2],
                 model.output_shape[3] // 2)
    # Prepare shapes for reshape layers
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Reshape(target_shape=(shape_in, 1)))
    model.add(tf.keras.layers.AvgPool1D(1, strides=2, padding="same"))
    model.add(tf.keras.layers.Reshape(target_shape=shape_out))
    model.add(tf.keras.layers.Conv2D(filters_num, (1, 1), padding="same",
                                     use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.BatchNormalization())

  def build(self):
    """
    Build the model
    :return: qkeras and larq models
    """
    qkeras_network = self.build_qkeras_birealnet()
    print("\nQKeras network successfully created")
    larq_network = self.build_larq_birealnet()
    print("Larq network successfully created")
    return qkeras_network, larq_network

  def build_qkeras_birealnet(self):
    """
    Build the qkeras version of the birealnet
    :return: qkeras model of the birealnet
    """
    qkeras_biRealNet = tf.keras.models.Sequential()
    qkeras_biRealNet.add(tf.keras.layers.InputLayer(input_shape=(224, 224, 3)))
    qkeras_biRealNet.add(
      tf.keras.layers.Conv2D(64, (7, 7), strides=2, padding="same",
                             use_bias=False))
    qkeras_biRealNet.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    qkeras_biRealNet.add(
      tf.keras.layers.MaxPool2D((3, 3), strides=2, padding="same"))
    for _ in range(0, 4):
      self.add_qkeras_residual_block(qkeras_biRealNet, 64)
    self.add_qkeras_connection_block(qkeras_biRealNet, 128)

    for _ in range(0, 3):
      self.add_qkeras_residual_block(qkeras_biRealNet, 128)
    self.add_qkeras_connection_block(qkeras_biRealNet, 256)

    for _ in range(0, 3):
      self.add_qkeras_residual_block(qkeras_biRealNet, 256)
    self.add_qkeras_connection_block(qkeras_biRealNet, 512)

    for _ in range(0, 3):
      self.add_qkeras_residual_block(qkeras_biRealNet, 512)
    qkeras_biRealNet.add(tf.keras.layers.AveragePooling2D(pool_size=(7, 7)))
    qkeras_biRealNet.add(tf.keras.layers.Flatten())
    qkeras_biRealNet.add(tf.keras.layers.Dense(1000))
    qkeras_biRealNet.add(tf.keras.layers.Activation("softmax", dtype="float32"))
    qkeras_biRealNet.load_weights(PATH_BIREALNET)
    return qkeras_biRealNet

  def build_larq_birealnet(self):
    """
    Build the larq version of the birealnet
    :return: larq model of the birealnet
    """
    larq_biRealNet = tf.keras.models.Sequential()
    larq_biRealNet.add(tf.keras.layers.InputLayer(input_shape=(224, 224, 3)))
    larq_biRealNet.add(
      tf.keras.layers.Conv2D(64, (7, 7), strides=2, padding="same",
                             use_bias=False))
    larq_biRealNet.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    larq_biRealNet.add(
      tf.keras.layers.MaxPool2D((3, 3), strides=2, padding="same"))
    for _ in range(0, 4):
      self.add_larq_residual_block(larq_biRealNet, 64)
    self.add_larq_connection_block(larq_biRealNet, 128)

    for _ in range(0, 3):
      self.add_larq_residual_block(larq_biRealNet, 128)
    self.add_larq_connection_block(larq_biRealNet, 256)

    for _ in range(0, 3):
      self.add_larq_residual_block(larq_biRealNet, 256)
    self.add_larq_connection_block(larq_biRealNet, 512)

    for _ in range(0, 3):
      self.add_larq_residual_block(larq_biRealNet, 512)
    larq_biRealNet.add(tf.keras.layers.AveragePooling2D(pool_size=(7, 7)))
    larq_biRealNet.add(tf.keras.layers.Flatten())
    larq_biRealNet.add(tf.keras.layers.Dense(1000))
    larq_biRealNet.add(tf.keras.layers.Activation("softmax", dtype="float32"))
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
