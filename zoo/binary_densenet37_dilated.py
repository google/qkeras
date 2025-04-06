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
# densenet_e37_dilated -> https://drive.google.com/file/d/1JZoiPcQlMJ8KAe3vdli4ixM_hGWHkPXm/view?usp=sharing

import qkeras as q
import tensorflow as tf
import larq as lq
from utils import compare_network, create_random_dataset, dump_network_to_json

# Define path to the pre-trained weights
PATH_DENSENET_E37_dilated = "./weights/binary_densenet_37_dilated_weights.h5"
DENSENET_E37_DILATED_NAME = "binary_densenet_e37_dilated"


class DenseNetE37Dilated:
  """Class to create and load weights of: densenet_e37_dilated

  Attributes:
    network_name: Name of the network
    filters: Number of filters for Conv2D
  """

  def __init__(self):
    self.__weights_path = PATH_DENSENET_E37_dilated
    self.network_name = DENSENET_E37_DILATED_NAME
    self.filters = (128, 192, 256)
    self.__filter_repetition = (6, 8, 12, 6)

  @staticmethod
  def add_qkeras_quant_block(given_model, filters_num):
    """Adds a sequence of layers to the given model

    Add a sequence of: Batch Normalization, Quantization Activations, Conv2D

    Args:
      given_model: model where to add the sequence
      filters_num: number of filters for Conv2D

    Returns:
      Given Model plus the sequence
    """
    y = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(given_model)
    y = q.QActivation("binary(alpha=1)")(y)
    y = q.QConv2D(filters=filters_num, kernel_size=3,
                  kernel_quantizer="binary(alpha=1)",
                  kernel_initializer="glorot_normal",
                  padding="same",
                  use_bias=False,
                  )(y)
    return tf.keras.layers.concatenate([given_model, y])

  @staticmethod
  def add_larq_quant_block(given_model, filters_num):
    """Same method of add_qkeras_quant_block but for a larq network
    """
    y = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(given_model)
    y = lq.layers.QuantConv2D(
      filters=filters_num,
      kernel_size=3,
      input_quantizer=lq.quantizers.SteSign(clip_value=1.3),
      kernel_quantizer=lq.quantizers.SteSign(clip_value=1.3),
      kernel_initializer="glorot_normal",
      kernel_constraint=lq.constraints.WeightClip(clip_value=1.3),
      padding="same",
      use_bias=False,
    )(y)
    return tf.keras.layers.concatenate([given_model, y])

  @staticmethod
  def add_connection_block(given_model, filters_num, max_pool=0):
    """Adds a sequence of layers to the given model

    Add a sequence of: Batch Normalization, MaxPooling2D, Activation, Conv2D

    Args:
      given_model: model where to add the sequence
      filters_num: number of filters for Conv2D

    Returns:
      Given Model plus the sequence
    """
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(given_model)
    if max_pool:
      x = tf.keras.layers.MaxPooling2D(2, strides=2)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(filters=filters_num, kernel_size=1,
                               kernel_initializer="he_normal",
                               use_bias=False)(x)
    return x

  def build(self):
    """Build the model

    Returns:
      Qkeras and larq models
    """
    qkeras_network = self.build_qkeras_densenet_e37_dilated()
    print("\nQKeras network successfully created")
    larq_network = self.build_larq_densenet_e37_dilated()
    print("Larq network successfully created")
    return qkeras_network, larq_network

  def build_qkeras_densenet_e37_dilated(self):
    """Build the qkeras version of the densenet_e37_dilated

    Returns:
      Qkeras model of the densenet_e37_dilated
    """
    input_layer = tf.keras.Input(shape=(224, 224, 3))
    qkeras_densenet = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2,
                                             padding="same",
                                             kernel_initializer="he_normal",
                                             use_bias=False)(input_layer)
    qkeras_densenet = tf.keras.layers.BatchNormalization(momentum=0.9,
                                                         epsilon=1e-5)(
      qkeras_densenet)
    qkeras_densenet = tf.keras.layers.Activation("relu")(qkeras_densenet)
    qkeras_densenet = tf.keras.layers.MaxPool2D(3, strides=2, padding="same")(
      qkeras_densenet)

    for filter_index, filter_num in enumerate(self.filters):
      for _ in range(0, self.__filter_repetition[filter_index]):
        qkeras_densenet = self.add_larq_quant_block(qkeras_densenet, 64)
      qkeras_densenet = \
        self.add_connection_block(qkeras_densenet, filter_num,
                                  max_pool=1 if filter_index == 1 else 0)

    for _ in range(0, self.__filter_repetition[-1]):
      qkeras_densenet = self.add_larq_quant_block(qkeras_densenet, 64)

    qkeras_densenet = tf.keras.layers.BatchNormalization(momentum=0.9,
                                                         epsilon=1e-5)(
      qkeras_densenet)
    qkeras_densenet = tf.keras.layers.Activation("relu")(qkeras_densenet)
    qkeras_densenet = tf.keras.layers.MaxPool2D(pool_size=28)(qkeras_densenet)
    qkeras_densenet = tf.keras.layers.Flatten()(qkeras_densenet)
    qkeras_densenet = tf.keras.layers.Dense(1000,
                                            kernel_initializer="glorot_normal")(
      qkeras_densenet)
    qkeras_densenet = tf.keras.layers.Activation("softmax", dtype="float32")(
      qkeras_densenet)
    qkeras_densenet = tf.keras.Model(inputs=input_layer,
                                     outputs=qkeras_densenet)
    qkeras_densenet.load_weights(self.__weights_path)
    return qkeras_densenet

  def build_larq_densenet_e37_dilated(self):
    """Build the larq version of the densenet_e37_dilated

    Returns:
      Larq model of the densenet_e37_dilated
    """
    input_layer = tf.keras.Input(shape=(224, 224, 3))
    larq_densenet = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2,
                                           padding="same",
                                           kernel_initializer="he_normal",
                                           use_bias=False)(input_layer)
    larq_densenet = tf.keras.layers.BatchNormalization(momentum=0.9,
                                                       epsilon=1e-5)(
      larq_densenet)
    larq_densenet = tf.keras.layers.Activation("relu")(larq_densenet)
    larq_densenet = tf.keras.layers.MaxPool2D(3, strides=2, padding="same")(
      larq_densenet)

    for filter_index, filter_num in enumerate(self.filters):
      for _ in range(0, self.__filter_repetition[filter_index]):
        larq_densenet = self.add_larq_quant_block(larq_densenet, 64)
      larq_densenet = \
        self.add_connection_block(larq_densenet, filter_num,
                                  max_pool=1 if filter_index == 1 else 0)

    for _ in range(0, self.__filter_repetition[-1]):
      larq_densenet = self.add_larq_quant_block(larq_densenet, 64)

    larq_densenet = tf.keras.layers.BatchNormalization(momentum=0.9,
                                                       epsilon=1e-5)(
      larq_densenet)
    larq_densenet = tf.keras.layers.Activation("relu")(larq_densenet)
    larq_densenet = tf.keras.layers.MaxPool2D(pool_size=28)(larq_densenet)
    larq_densenet = tf.keras.layers.Flatten()(larq_densenet)
    larq_densenet = tf.keras.layers.Dense(1000,
                                          kernel_initializer="glorot_normal")(
      larq_densenet)
    larq_densenet = tf.keras.layers.Activation("softmax", dtype="float32")(
      larq_densenet)
    larq_densenet = tf.keras.Model(inputs=input_layer, outputs=larq_densenet)
    larq_densenet.load_weights(self.__weights_path)
    return larq_densenet


if __name__ == "__main__":
  # Create a random dataset with 100 samples
  random_data = create_random_dataset(100)

  network = DenseNetE37Dilated()
  qkeras_network, larq_network = network.build()
  # Compare mean MSE and Absolute error of the the networks
  compare_network(qkeras_network=qkeras_network, larq_network=larq_network,
                  dataset=random_data, network_name=DENSENET_E37_DILATED_NAME)
  dump_network_to_json(qkeras_network=qkeras_network,
                       larq_network=larq_network,
                       network_name=DENSENET_E37_DILATED_NAME)
