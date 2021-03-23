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
# QuickNet -> https://github.com/larq/zoo/releases/download/quicknet-v1.0/quicknet_weights.h5
# QuickNetSmall -> https://github.com/larq/zoo/releases/download/quicknet-v1.0/quicknet_small_weights.h5
# QuickNetLarge -> https://github.com/larq/zoo/releases/download/quicknet-v1.0/quicknet_large_weights.h5

import qkeras as q
import tensorflow as tf
import larq as lq
from utils import compare_network, create_random_dataset, dump_network_to_json


# Define path to the pre-trained weights
path_quicknet = "./weigths/quicknet_weights.h5"
path_quicknet_small = "./weigths/quicknet_small_weights.h5"
path_quicknet_large = "./weigths/quicknet_large_weights.h5"
quicknet_large_name = ["quickNet_large"]
quicknet_small_name = ["quickNet_small"]
quicknet_name = ["quickNet"]


class quicknet():
  """
  Class to create and load weights of: quicknet, quicknet small and quicknet
  large networks. Select the size of the network from size param. If None size
  is provided creates the quicknet version.
  :param size: size of the network
  """

  def __init__(self, size=None):
    if str(size).lower() == "large":
      self.__id = 0
      self.__filters = ((64, 128, 256, 512))
      self.__weights_path = path_quicknet_large
      self.network_name = quicknet_large_name
    elif str(size).lower() == "small":
      self.__id = 1
      self.__filters = ((32, 64, 256, 512))
      self.__weights_path = path_quicknet_small
      self.network_name = quicknet_small_name
    elif str(size) == "":
      self.__id = 2
      self.__filters = ((64, 128, 256, 512))
      self.__weights_path = path_quicknet
      self.network_name = quicknet_name
    else:
      raise NameError("name:", str, "not recognized")

  @staticmethod
  def add_qkeras_residual(model, filters_num):
    """
    Add a sequence of: Activation quantization, Quantized Conv2D,
    BatchNormalization to the given model
    :param model: model where to add the sequence
    :param filters_num: number of filters for QConv2D
    :return: model plus the sequence
    """
    model.add(q.QActivation("binary"))
    model.add(q.QConv2D(filters_num, (3, 3), activation="relu",
                        kernel_quantizer="binary(alpha=1)",
                        kernel_initializer="glorot_normal",
                        padding="same", use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    return model

  @staticmethod
  def add_qkeras_transistion(model, strides, filters_num):
    """
    Add a sequence of: Activation quantization, Quantized Conv2D,
    BatchNormalization to the given model
    :param model: model where to add the sequence
    :param strides: strides param for MaxPool2d and QConv2D
    :param filters_num: number of filters for QConv2D
    :return: model plus the sequence
    """
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=strides, strides=1))
    model.add(tf.keras.layers.DepthwiseConv2D((3, 3), padding="same",
                                              strides=strides, trainable=False,
                                              use_bias=False))
    model.add(q.QConv2D(filters_num, (1, 1), kernel_initializer="glorot_normal",
                        use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    return model

  @staticmethod
  def add_larq_residual(model, filters_num):
    """
    Same method of add_qkeras_residual but for a larq network
    """
    model.add(lq.layers.QuantConv2D(filters_num, (3, 3), activation="relu",
                                    input_quantizer="ste_sign",
                                    kernel_quantizer=
                                    lq.quantizers.SteSign(clip_value=1.25),
                                    kernel_constraint=
                                    lq.constraints.WeightClip(clip_value=1.25),
                                    kernel_initializer="glorot_normal",
                                    padding="same", use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    return model

  @staticmethod
  def add_larq_transistion(model, strides, filters_num):
    """
    Same method of add_qkeras_transistion but for a larq network
    """
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=strides, strides=1))
    model.add(tf.keras.layers.DepthwiseConv2D((3, 3), padding="same",
                                              strides=strides,
                                              trainable=False,
                                              use_bias=False))
    model.add(lq.layers.QuantConv2D(filters_num, (1, 1),
                                    kernel_initializer="glorot_normal",
                                    use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    return model

  def add_qkeras_first_block(self, model):
    """
    Add a sequence of: Input, QConv2D, BatchNormalization, Activation,
    QdepthWiseConv2D, BatchNormalization, QConv2d, BatchNormalization
    :param model: model where to add the sequence
    :return: model plus the sequence
    """
    model.add(tf.keras.layers.InputLayer(input_shape=(224, 224, 3)))
    model.add(q.QConv2D(self.__filters[0] // 4, (3, 3),
                        kernel_initializer="he_normal",
                        padding="same",
                        strides=2, use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))
    model.add(q.QDepthwiseConv2D((3, 3), padding="same", strides=2,
                                 use_bias=False))
    model.add(tf.keras.layers.BatchNormalization(scale=False,
                                                 center=False))
    model.add(q.QConv2D(self.__filters[0], 1,
                        kernel_initializer="he_normal",
                        use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    return model

  def add_qkeras_last_block(self, model):
    """
    Add a sequence of: Activation, AveragePooling2D, Flatten, Dense
    :param model: model where to add the sequence
    :return: model plus the sequence
    """
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(7, 7)))
    model.add(tf.keras.layers.Flatten())
    model.add(q.QDense(1000, kernel_initializer="glorot_normal"))
    model.add(tf.keras.layers.Activation("softmax", dtype="float32"))
    model.load_weights(self.__weights_path)
    return model

  def add_larq_first_block(self, model):
    """
    Same method of add_qkeras_first_block but for a larq network
    """
    model.add(tf.keras.layers.InputLayer(input_shape=(224, 224, 3)))
    model.add(lq.layers.QuantConv2D(self.__filters[0] // 4, (3, 3),
                                    kernel_initializer="he_normal",
                                    padding="same", strides=2,
                                    use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))
    model.add(lq.layers.QuantDepthwiseConv2D((3, 3), padding="same",
                                             strides=2, use_bias=False))
    model.add(tf.keras.layers.BatchNormalization(scale=False,
                                                 center=False))
    model.add(lq.layers.QuantConv2D(self.__filters[0], 1,
                                    kernel_initializer="he_normal",
                                    use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())

  def add_larq_last_block(self, model):
    """
    Same method of add_larq_first_block but for a larq network
    """
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(7, 7)))
    model.add(tf.keras.layers.Flatten())
    model.add(lq.layers.QuantDense(1000, kernel_initializer="glorot_normal"))
    model.add(tf.keras.layers.Activation("softmax", dtype="float32"))
    model.load_weights(self.__weights_path)

  def build(self):
    """
    Build the model based on its ID
    :return: qkeras and larq models
    """
    if self.__id == 0:
      qkeras_network = self.build_larq_quicknet_large()
      print("\nQKeras network successfully created")
      larq_network = self.build_larq_quicknet_large()
      print("Larq network successfully created")
      return qkeras_network, larq_network

    else:
      qkeras_network = self.build_qkeras_quicknet()
      print("\nQKeras network successfully created")
      larq_network = self.build_larq_quicknet()
      print("Larq network successfully created")
      return qkeras_network, larq_network

  def build_qkeras_quicknet_large(self):
    """
    Build the qkeras version of the quicknet large
    :return: qkeras model of the quicknet large
    """
    # Input layer
    qkeras_quicknet = tf.keras.models.Sequential()
    self.add_qkeras_first_block(qkeras_quicknet)
    for _ in range(0, 6):
      self.add_qkeras_residual(qkeras_quicknet, filters_num=self.__filters[0])
    self.add_qkeras_transistion(qkeras_quicknet, strides=2,
                                filters_num=self.__filters[1])

    for _ in range(0, 8):
      self.add_qkeras_residual(qkeras_quicknet, filters_num=self.__filters[1])
    self.add_qkeras_transistion(qkeras_quicknet, strides=2,
                                filters_num=self.__filters[2])

    for _ in range(0, 12):
      self.add_qkeras_residual(qkeras_quicknet, filters_num=self.__filters[2])
    self.add_qkeras_transistion(qkeras_quicknet, strides=2,
                                filters_num=self.__filters[3])

    for _ in range(0, 6):
      self.add_qkeras_residual(qkeras_quicknet, filters_num=self.__filters[3])
    self.add_qkeras_last_block(qkeras_quicknet)
    return qkeras_quicknet

  def build_larq_quicknet_large(self):
    """
    Build the larq version of the quicknet large
    :return: larq model of the quicknet large
    """
    # Input layer
    larq_quicknet = tf.keras.models.Sequential()
    self.add_larq_first_block(larq_quicknet)
    for _ in range(0, 6):
      self.add_qkeras_residual(larq_quicknet, filters_num=self.__filters[0])
    self.add_qkeras_transistion(larq_quicknet, strides=2,
                                filters_num=self.__filters[1])

    for _ in range(0, 8):
      self.add_qkeras_residual(larq_quicknet, filters_num=self.__filters[1])
    self.add_qkeras_transistion(larq_quicknet, strides=2,
                                filters_num=self.__filters[2])

    for _ in range(0, 12):
      self.add_qkeras_residual(larq_quicknet, filters_num=self.__filters[2])
    self.add_qkeras_transistion(larq_quicknet, strides=2,
                                filters_num=self.__filters[3])

    for _ in range(0, 6):
      self.add_qkeras_residual(larq_quicknet, filters_num=self.__filters[3])
    self.add_larq_last_block(larq_quicknet)
    return larq_quicknet

  def build_qkeras_quicknet(self):
    """
    Build the qkeras version of the quicknet
    :return: qkeras model of the quicknet
    """
    # Input layer
    qkeras_quicknet = tf.keras.models.Sequential()
    self.add_qkeras_first_block(qkeras_quicknet)
    for filters_index in range(0, 3):
      # Residual block
      for _ in range(0, 4):
        filters_num = self.__filters[filters_index]
        self.add_qkeras_residual(qkeras_quicknet, filters_num=filters_num)
      # Transition block
      filters_num = self.__filters[filters_index + 1]
      self.add_qkeras_transistion(qkeras_quicknet, strides=2,
                                  filters_num=filters_num)
    # Residual block
    for _ in range(0, 4):
      filters_num = self.__filters[3]
      self.add_qkeras_residual(qkeras_quicknet, filters_num=filters_num)
    self.add_qkeras_last_block(qkeras_quicknet)
    return qkeras_quicknet

  def build_larq_quicknet(self):
    """
    Build the larq version of the quicknet
    :return: larq model of the quicknet
    """
    # Input layer
    larq_quicknet = tf.keras.models.Sequential()
    self.add_larq_first_block(larq_quicknet)
    for filters_index in range(0, 3):
      # Residual block
      for _ in range(0, 4):
        filters_num = self.__filters[filters_index]
        self.add_larq_residual(larq_quicknet, filters_num=filters_num)
      # Transition block
      filters_num = self.__filters[filters_index + 1]
      self.add_larq_transistion(larq_quicknet, strides=2,
                                filters_num=filters_num)
    # Residual block
    for _ in range(0, 4):
      filters_num = self.__filters[3]
      self.add_larq_residual(larq_quicknet, filters_num=filters_num)
    self.add_larq_last_block(larq_quicknet)
    return larq_quicknet


if __name__ == "__main__":
  # Create a random dataset with 100 samples
  random_data = create_random_dataset(100)

  network_names = ["quickNet", "quickNet_large", "quickNet_small"]
  sizes = ["", "large", "small"]

  for size, name in zip(sizes, network_names):
    network = quicknet(size)
    qkeras_network, larq_network = network.build()
    # Compare mean MSE and Absolute error of the the networks
    compare_network(qkeras_network=qkeras_network, larq_network=larq_network,
                  dataset=random_data, network_name=name)
    dump_network_to_json(qkeras_network=qkeras_network,
                         larq_network=larq_network, network_name=name+"_"+size)