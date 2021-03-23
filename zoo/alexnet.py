###############################################################################
# .. attention::
#
# Copyright(c) 2021 Francesco Loro, Master Degree Student Università degli studi di Padova.
# All rights reserved.
#
# This software component is licensed by Apache License Version 2.0
# http://www.apache.org/licenses/
# Same as QKeras
# You may not use this file except in compliance with# the License.
#
###############################################################################

__author__ = "Francesco Loro"
__email__ = "francesco.official@gmail.com"
__supervisor__ = "Danilo Pau"
__email__ = "danilo.pau@st.com"

# Download pretrained weight from:
# Alexnet -> https://github.com/larq/zoo/releases/download/binary_alexnet-v0.2.0/binary_alexnet_weights.h5

import qkeras as q
import tensorflow as tf
import larq as lq
from utils import compare_network, create_random_dataset, dump_network_to_json

# Define path to the pre-trained weights
path_alexnet = "./weigths/binary_alexnet_weights.h5"
alexnet_name = "alexNet"


class alexnet():
  """
  Class to create and load weights of: alexnet
  """

  def __init__(self):
    self.__weights_path = path_alexnet
    self.network_name = alexnet_name

  @staticmethod
  def add_qkeras_conv_block(model, filters_num, kernel_size, pool,
                            qnt, strides=1):
    """
    Add a sequence of: Activation quantization, Quantized Conv2D, MaxPooling
    and BatchNormalization to the given model
    :param model: model where to add the sequence
    :param filters_num: number of filters for Cov2D
    :param kernel_size: kernel size for Cov2D
    :param pool: boolean to decide if MaxPool is performed or not
    :param qnt: boolean to decide if Activation quantization is performed
                or not
    :param strides: strides for Conv2D
    :return: model plus the sequence
    """
    if qnt:
      model.add(q.QActivation("binary(alpha=1)"))
    model.add(
      q.QConv2D(filters_num, kernel_size, strides=strides, padding="same",
                use_bias=False, kernel_quantizer="binary(alpha=1)"))
    if pool:
      model.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
    model.add(tf.keras.layers.BatchNormalization(scale=False, momentum=0.9))
    return model

  @staticmethod
  def add_qkeras_dense_block(model, units):
    """
    Add a sequence of: Activation quantization, Quantized Dense to the given
    model
    :param model: model where to add the sequence
    :param units: neurons of the Dense
    :return: model plus the sequence
    """
    model.add(q.QActivation("binary(alpha=1)"))
    model.add(
      q.QDense(units, kernel_quantizer="binary(alpha=1)", use_bias=False))
    model.add(tf.keras.layers.BatchNormalization(scale=False, momentum=0.9))
    return model

  @staticmethod
  def add_larq_conv_block(model, filters_num, kernel_size, pool, qnt, strides=1):
    """
    Same method of add_qkeras_conv_block but for a larq network
    """
    model.add(
      lq.layers.QuantConv2D(filters_num, kernel_size, strides=strides,
                            padding="same", use_bias=False,
                            input_quantizer=None if not qnt else "ste_sign",
                            kernel_quantizer="ste_sign",
                            kernel_constraint="weight_clip"))
    if pool:
      model.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
    model.add(tf.keras.layers.BatchNormalization(scale=False, momentum=0.9))
    return model

  @staticmethod
  def add_larq_dense_block(model, units):
    """
    Same method of add_qkeras_dense_block but for a larq network
    """
    model.add(lq.layers.QuantDense(units, use_bias=False,
                                   input_quantizer="ste_sign",
                                   kernel_quantizer="ste_sign",
                                   kernel_constraint="weight_clip"))
    model.add(tf.keras.layers.BatchNormalization(scale=False, momentum=0.9))
    return model

  def build(self):
    """
    Build the model
    :return: qkeras and larq models
    """
    qkeras_network = self.build_qkeras_alexnet()
    print("\nQKeras network successfully created")
    larq_network = self.build_larq_alexnet()
    print("Larq network successfully created")
    return qkeras_network, larq_network

  def build_qkeras_alexnet(self):
    """
    Build the qkeras version of the alexnet
    :return: qkeras model of the alexnet
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
    """
    Build the larq version of the alexnet
    :return: larq model of the alexnet
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

  network = alexnet()
  qkeras_network, larq_network = network.build()
  # Compare mean MSE and Absolute error of the the networks
  compare_network(qkeras_network=qkeras_network, larq_network=larq_network,
                  dataset=random_data, network_name=alexnet_name)
  dump_network_to_json(qkeras_network=qkeras_network,
                       larq_network=larq_network,
                       network_name=alexnet_name)
