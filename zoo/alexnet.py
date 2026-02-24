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
# Alexnet -> https://drive.google.com/file/d/1-65sB1xnJuOoPhL00TYY0s3Fov0zxBHJ/view?usp=sharing

import qkeras as q
import tensorflow as tf
import larq as lq
from utils import compare_network, create_random_dataset, dump_network_to_json

# Define path to the pre-trained weights
PATH_ALEXNET = "weights/binary_alexnet_weights.h5"
ALEXNET_NAME = "alexNet"


class AlexNet:
  """Class to create and load weights of: alexnet

  Attributes:
        network_name: Name of the network
  """

  def __init__(self):
    self.__weights_path = PATH_ALEXNET
    self.network_name = ALEXNET_NAME

  @staticmethod
  def add_qkeras_conv_block(given_model, filters_num, kernel_size, pool,
                            qnt, strides=1):
    """Adds a sequence of layers to the given model

    Add a sequence of: Activation quantization, Quantized Conv2D, MaxPooling
    and BatchNormalization to the given model

    Args:
      given_model: model where to add the sequence
      filters_num: number of filters for Conv2D
      kernel_size: kernel size for Conv2D
      pool: boolean to decide if MaxPool is performed or not
      qnt: boolean to decide if Activation quantization is performed
           or not
      strides: strides for Conv2D

    Returns:
      Given Model plus the sequence
    """
    if qnt:
      given_model.add(q.QActivation("binary(alpha=1)"))
    given_model.add(
      q.QConv2D(filters_num, kernel_size, strides=strides, padding="same",
                use_bias=False, kernel_quantizer="binary(alpha=1)"))
    if pool:
      given_model.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
    given_model.add(tf.keras.layers.BatchNormalization(scale=False,
                                                       momentum=0.9))
    return given_model

  @staticmethod
  def add_qkeras_dense_block(given_model, units):
    """Adds a sequence of layers to the given model

    Add a sequence of: Activation quantization, Quantized Dense and
    Batch Normalization to the given model

    Args:
      given_model: model where to add the sequence
      units: neurons of the Dense

    Returns:
      Given Model plus the sequence
    """
    given_model.add(q.QActivation("binary(alpha=1)"))
    given_model.add(
      q.QDense(units, kernel_quantizer="binary(alpha=1)", use_bias=False))
    given_model.add(tf.keras.layers.BatchNormalization(scale=False,
                                                       momentum=0.9))
    return given_model

  @staticmethod
  def add_larq_conv_block(given_model, filters_num, kernel_size, pool, qnt,
                          strides=1):
    """Same method of add_qkeras_conv_block but for a larq network
    """
    given_model.add(
      lq.layers.QuantConv2D(filters_num, kernel_size, strides=strides,
                            padding="same", use_bias=False,
                            input_quantizer=None if not qnt else "ste_sign",
                            kernel_quantizer="ste_sign",
                            kernel_constraint="weight_clip"))
    if pool:
      given_model.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
    given_model.add(tf.keras.layers.BatchNormalization(scale=False,
                                                       momentum=0.9))
    return given_model

  @staticmethod
  def add_larq_dense_block(given_model, units):
    """Same method of add_qkeras_dense_block but for a larq network
    """
    given_model.add(lq.layers.QuantDense(units, use_bias=False,
                                         input_quantizer="ste_sign",
                                         kernel_quantizer="ste_sign",
                                         kernel_constraint="weight_clip"))
    given_model.add(tf.keras.layers.BatchNormalization(scale=False,
                                                       momentum=0.9))
    return given_model

  def build(self):
    """Build the model

    Returns:
      Qkeras and larq models
    """
    qkeras_network = self.build_qkeras_alexnet()
    print("\nQKeras network successfully created")
    larq_network = self.build_larq_alexnet()
    print("Larq network successfully created")
    return qkeras_network, larq_network

  def build_qkeras_alexnet(self):
    """Build the qkeras version of the alexnet

    Return:
      Qkeras model of the alexnet
    """
    qkeras_alexNet = tf.keras.models.Sequential()
    qkeras_alexNet.add(tf.keras.layers.InputLayer(input_shape=(224, 224, 3)))
    self.add_qkeras_conv_block(qkeras_alexNet, filters_num=64, kernel_size=11,
                               strides=4, pool=True, qnt=False)
    self.add_qkeras_conv_block(qkeras_alexNet, filters_num=192, kernel_size=5,
                               pool=True, qnt=True)
    self.add_qkeras_conv_block(qkeras_alexNet, filters_num=384, kernel_size=3,
                               pool=False, qnt=True)
    self.add_qkeras_conv_block(qkeras_alexNet, filters_num=384, kernel_size=3,
                               pool=False, qnt=True)
    self.add_qkeras_conv_block(qkeras_alexNet, filters_num=256, kernel_size=3,
                               pool=True, qnt=True)
    qkeras_alexNet.add(tf.keras.layers.Flatten())
    self.add_qkeras_dense_block(qkeras_alexNet, units=4096)
    self.add_qkeras_dense_block(qkeras_alexNet, units=4096)
    self.add_qkeras_dense_block(qkeras_alexNet, units=1000)
    qkeras_alexNet.add(tf.keras.layers.Activation("softmax", dtype="float32"))
    qkeras_alexNet.load_weights(self.__weights_path)
    return qkeras_alexNet

  def build_larq_alexnet(self):
    """Build the larq version of the alexnet

    Return:
      Larq model of the alexnet
    """
    larq_alexnet = tf.keras.models.Sequential()
    larq_alexnet.add(tf.keras.layers.InputLayer(input_shape=(224, 224, 3)))
    self.add_larq_conv_block(larq_alexnet, filters_num=64, kernel_size=11,
                             strides=4, pool=True, qnt=False)
    self.add_larq_conv_block(larq_alexnet, filters_num=192, kernel_size=5,
                             pool=True, qnt=True)
    self.add_larq_conv_block(larq_alexnet, filters_num=384, kernel_size=3,
                             pool=False, qnt=True)
    self.add_larq_conv_block(larq_alexnet, filters_num=384, kernel_size=3,
                             pool=False, qnt=True)
    self.add_larq_conv_block(larq_alexnet, filters_num=256, kernel_size=3,
                             pool=True, qnt=True)
    larq_alexnet.add(tf.keras.layers.Flatten())
    self.add_larq_dense_block(larq_alexnet, units=4096)
    self.add_larq_dense_block(larq_alexnet, units=4096)
    self.add_larq_dense_block(larq_alexnet, units=1000)
    larq_alexnet.add(tf.keras.layers.Activation("softmax", dtype="float32"))
    larq_alexnet.load_weights(self.__weights_path)
    return larq_alexnet


if __name__ == "__main__":
  # Create a random dataset with 100 samples
  random_data = create_random_dataset(100)

  network = AlexNet()
  qkeras_network, larq_network = network.build()
  # Compare mean MSE and Absolute error of the the networks
  compare_network(qkeras_network=qkeras_network, larq_network=larq_network,
                  dataset=random_data, network_name=ALEXNET_NAME)
  dump_network_to_json(qkeras_network=qkeras_network,
                       larq_network=larq_network,
                       network_name=ALEXNET_NAME)
