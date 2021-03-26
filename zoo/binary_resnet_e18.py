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
# resnet_e18 -> https://drive.google.com/file/d/1-eRhwVTzIKm3D0WoOls3eyeXmJE7Q6Cn/view?usp=sharing

import qkeras as q
import tensorflow as tf
import larq as lq
from utils import compare_network, create_random_dataset, dump_network_to_json

# Define path to the pre-trained weights
PATH_RESNET_E18 = "weights/resnet_e_18_weights.h5"
RESNET_E18_NAME = "binary_resnet_e18"


class ResNetE18:
  """
  Class to create and load weights of: resnet_e18
  Attributes:
        network_name: Name of the network
        filters: Number of filters for Conv2D
  """

  def __init__(self):
    self.__weights_path = PATH_RESNET_E18
    self.network_name = RESNET_E18_NAME
    self.filters = (64,128,256,512)

  @staticmethod
  def add_qkeras_quant_block(model, filters_num, strides=1):
    """
    Add a sequence of: Activation quantization, Quantized Conv2D,
    Batch Normalization
    :param model: model where to add the sequence
    :param filters_num: number of filters for Cov2D
    :param strides: strides for Conv2D
    """
    model.add(q.QActivation("binary(alpha=1)"))
    model.add(
      q.QConv2D(filters_num, kernel_size=3, strides=strides,
                            padding="same",
                            kernel_quantizer="binary(alpha=1)",
                            kernel_initializer="glorot_normal",
                            use_bias=False))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5))

  @staticmethod
  def add_qkeras_connection_block(model, filters_num):
    """
    Add a sequence of: Activation quantization, Quantized Conv2D, reshape,
    Average Pooling, Conv2D, 2x BatchNormalization
    :param model: model where to add the sequence
    :param filters_num: number of filters for Cov2D
    """
    model.add(q.QActivation("binary(alpha=1)"))
    model.add(
      q.QConv2D(filters_num, (3, 3), strides=(2, 2), use_bias=False,
                            padding="same",
                            kernel_quantizer="binary(alpha=1)",
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

  @staticmethod
  def add_larq_quant_block(model, filters_num, strides=1):
    """
   Same method of add_qkeras_quant_block but for a larq network
   """
    model.add(
      lq.layers.QuantConv2D(filters_num, kernel_size=3, strides=strides,
                            padding="same",
                            input_quantizer=lq.quantizers.SteSign(
                              clip_value=1.25),
                            kernel_quantizer=lq.quantizers.SteSign(
                              clip_value=1.25),
                            kernel_constraint=lq.constraints.WeightClip(
                              clip_value=1.25),
                            kernel_initializer="glorot_normal",
                            use_bias=False))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5))

  @staticmethod
  def add_larq_connection_block(model, filters_num):
    """
    Same method of add_qkeras_connection_block but for a larq network
    """
    model.add(
      lq.layers.QuantConv2D(filters_num, (3, 3), strides=(2, 2), use_bias=False,
                            padding="same",
                            input_quantizer=lq.quantizers.SteSign(
                              clip_value=1.25),
                            kernel_quantizer=lq.quantizers.SteSign(
                              clip_value=1.25),
                            kernel_constraint=lq.constraints.WeightClip(
                              clip_value=1.25),))

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
    qkeras_network = self.build_qkeras_resnet_e18()
    print("\nQKeras network successfully created")
    larq_network = self.build_larq_resnet_e18()
    print("Larq network successfully created")
    return qkeras_network, larq_network

  def build_qkeras_resnet_e18(self):
    """
    Build the qkeras version of the resnet_e18
    :return: qkeras model of the resnet_e18
    """
    qkeras_resnet = tf.keras.models.Sequential()
    qkeras_resnet.add(tf.keras.layers.InputLayer(input_shape=(224, 224, 3)))
    qkeras_resnet.add(
      tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding="same",
                             kernel_initializer="he_normal", use_bias=False))
    qkeras_resnet.add(
      tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5))
    qkeras_resnet.add(tf.keras.layers.Activation("relu"))
    qkeras_resnet.add(tf.keras.layers.MaxPool2D(3, strides=2, padding="same"))
    qkeras_resnet.add(
      tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5))

    for _ in range(0, 4):
      self.add_qkeras_quant_block(qkeras_resnet, 64)
    self.add_qkeras_connection_block(qkeras_resnet, 128)

    for i in range(1, 3):
      for _ in range(0, 3):
        self.add_qkeras_quant_block(qkeras_resnet, filters_num=self.filters[i])
      self.add_qkeras_connection_block(qkeras_resnet,
                                     filters_num=self.filters[i + 1])
    for _ in range(0, 3):
      self.add_qkeras_quant_block(qkeras_resnet, filters_num=self.filters[-1])
    qkeras_resnet.add(tf.keras.layers.Activation("relu"))
    qkeras_resnet.add(tf.keras.layers.MaxPool2D(pool_size=7))
    qkeras_resnet.add(tf.keras.layers.Flatten())
    qkeras_resnet.add(
      tf.keras.layers.Dense(1000, kernel_initializer="glorot_normal"))
    qkeras_resnet.add(tf.keras.layers.Activation("softmax", dtype="float32"))
    qkeras_resnet.load_weights(self.__weights_path)
    return qkeras_resnet


  def build_larq_resnet_e18(self):
    """
    Build the larq version of the resnet_e18
    :return: larq model of the resnet_e18
    """
    larq_resnet = tf.keras.models.Sequential()
    larq_resnet.add(tf.keras.layers.InputLayer(input_shape=(224, 224, 3)))
    larq_resnet.add(
      tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding="same",
                             kernel_initializer="he_normal", use_bias=False))
    larq_resnet.add(
      tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5))
    larq_resnet.add(tf.keras.layers.Activation("relu"))
    larq_resnet.add(tf.keras.layers.MaxPool2D(3, strides=2, padding="same"))
    larq_resnet.add(
      tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5))

    for _ in range(0, 4):
      self.add_larq_quant_block(larq_resnet, 64)
    self.add_larq_connection_block(larq_resnet, 128)

    for i in range(1, 3):
      for _ in range(0, 3):
        self.add_larq_quant_block(larq_resnet, filters_num=self.filters[i])
      self.add_larq_connection_block(larq_resnet, filters_num=self.filters[i+1])
    for _ in range(0, 3):
      self.add_larq_quant_block(larq_resnet, filters_num=self.filters[-1])
    larq_resnet.add(tf.keras.layers.Activation("relu"))
    larq_resnet.add(tf.keras.layers.MaxPool2D(pool_size=7))
    larq_resnet.add(tf.keras.layers.Flatten())
    larq_resnet.add(
      tf.keras.layers.Dense(1000, kernel_initializer="glorot_normal"))
    larq_resnet.add(tf.keras.layers.Activation("softmax", dtype="float32"))
    larq_resnet.load_weights(self.__weights_path)
    return larq_resnet


if __name__ == "__main__":
  # Create a random dataset with 100 samples
  random_data = create_random_dataset(100)

  network = ResNetE18()
  qkeras_network, larq_network = network.build()
  # Compare mean MSE and Absolute error of the the networks
  compare_network(qkeras_network=qkeras_network, larq_network=larq_network,
                  dataset=random_data, network_name=RESNET_E18_NAME)
  dump_network_to_json(qkeras_network=qkeras_network,
                       larq_network=larq_network,
                       network_name=RESNET_E18_NAME)
