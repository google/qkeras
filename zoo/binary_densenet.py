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
# densenet_e28 -> https://drive.google.com/file/d/1-Plw5XKKCeTP17nUpnzRM3JnUdlwv_MJ/view?usp=sharing
# densenet_e37 -> https://drive.google.com/file/d/1PldbeERqq9-Xz8HQtaRAHtbtqItRznLp/view?usp=sharing
# densenet e45 -> https://drive.google.com/file/d/1Lpc-rRAleNJSF-Y4SlDab9bW2L8cUkn0/view?usp=sharing

import qkeras as q
import tensorflow as tf
import larq as lq
from utils import compare_network, create_random_dataset, dump_network_to_json

# Define path to the pre-trained weights
PATH_DENSENET_E28 = "./weights/binary_densenet_28_weights.h5"
DENSENET_E28_NAME = "binary_densenet_e28"
PATH_DENSENET_E37 = "./weights/binary_densenet_37_weights.h5"
DENSENET_E37_NAME = "binary_densenet_e37"
PATH_DENSENET_E45 = "./weights/binary_densenet_45_weights.h5"
DENSENET_E45_NAME = "binary_densenet_e45"


class DenseNet():
  """Class to create and load weights of: densenet

  Attributes:
    size : int to specify the size of the network
    network_name: Name of the network

  Raises:
      NameError: if the size is not one of these: 28, 37, 45
  """

  def __init__(self, size=28):
    if size == 28:
      self.__weights_path = PATH_DENSENET_E28
      self.network_name = DENSENET_E28_NAME
      self.__filters = (160, 192, 256)
      self.__ID = 0

    elif size == 37:
      self.__weights_path = PATH_DENSENET_E37
      self.network_name = DENSENET_E37_NAME
      self.__filters = (128, 192, 256)
      self.__filter_repetition = (6, 8, 12, 6)
      self.__ID = 1

    elif size == 45:
      self.__weights_path = PATH_DENSENET_E45
      self.network_name = DENSENET_E45_NAME
      self.__filters = (160, 288, 288)
      self.__filter_repetition = (6, 12, 14, 6, 8)
      self.__ID = 2

    else:
      raise NameError("size:", str(size), "not recognized")

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
    y = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5) \
      (given_model)
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
    y = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(
      given_model)
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
  def add_connection_block(given_model, filters_num):
    """Adds a sequence of layers to the given model

    Add a sequence of: Batch Normalization, MaxPooling2D, Activation, Conv2D

    Args:
      given_model: model where to add the sequence
      filters_num: number of filters for Conv2D

    Returns:
      Given Model plus the sequence
    """
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(
      given_model)
    x = tf.keras.layers.MaxPooling2D(2, strides=2)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(filters=filters_num, kernel_size=1,
                               kernel_initializer="he_normal",
                               use_bias=False)(x)
    return x

  def build(self):
    """Builds the model by its ID

    Returns:
      Qkeras and larq models
    """
    if self.__ID == 0:
      qkeras_network = self.build_qkeras_densenet_e28()
      print("\nQKeras network successfully created")
      larq_network = self.build_larq_densenet_e28()
      print("Larq network successfully created")
      return qkeras_network, larq_network

    if self.__ID == 1:
      qkeras_network = self.build_qkeras_densenet_e37()
      print("\nQKeras network successfully created")
      larq_network = self.build_larq_densenet_e37()
      print("Larq network successfully created")
      return qkeras_network, larq_network

    if self.__ID == 2:
      qkeras_network = self.build_qkeras_densenet_e45()
      print("\nQKeras network successfully created")
      larq_network = self.build_larq_densenet_e45()
      print("Larq network successfully created")
      return qkeras_network, larq_network

  def build_qkeras_densenet_e28(self):
    """Build the qkeras version of the densenet_e28

    Returns:
      Qkeras model of the densenet_e28
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

    for filter_num in self.__filters:
      for _ in range(0, 6):
        qkeras_densenet = self.add_larq_quant_block(qkeras_densenet, 64)
      qkeras_densenet = self.add_connection_block(qkeras_densenet, filter_num)

    for _ in range(0, 5):
      qkeras_densenet = self.add_larq_quant_block(qkeras_densenet, 64)

    qkeras_densenet = tf.keras.layers.BatchNormalization(momentum=0.9,
                                                         epsilon=1e-5)(
      qkeras_densenet)
    qkeras_densenet = tf.keras.layers.Activation("relu")(qkeras_densenet)
    qkeras_densenet = tf.keras.layers.MaxPool2D(pool_size=7)(qkeras_densenet)
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

  def build_larq_densenet_e28(self):
    """Build the larq version of the densenet_e28

       Returns:
         Larq model of the densenet_e28
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

    for filter_num in self.__filters:
      for _ in range(0, 6):
        larq_densenet = self.add_larq_quant_block(larq_densenet, 64)
      larq_densenet = self.add_connection_block(larq_densenet, filter_num)

    for _ in range(0, 5):
      larq_densenet = self.add_larq_quant_block(larq_densenet, 64)

    larq_densenet = tf.keras.layers.BatchNormalization(momentum=0.9,
                                                       epsilon=1e-5)(
      larq_densenet)
    larq_densenet = tf.keras.layers.Activation("relu")(larq_densenet)
    larq_densenet = tf.keras.layers.MaxPool2D(pool_size=7)(larq_densenet)
    larq_densenet = tf.keras.layers.Flatten()(larq_densenet)
    larq_densenet = tf.keras.layers.Dense(1000,
                                          kernel_initializer="glorot_normal")(
      larq_densenet)
    larq_densenet = tf.keras.layers.Activation("softmax", dtype="float32")(
      larq_densenet)
    larq_densenet = tf.keras.Model(inputs=input_layer, outputs=larq_densenet)
    larq_densenet.load_weights(self.__weights_path)
    return larq_densenet

  def build_qkeras_densenet_e37(self):
    """Build the qkeras version of the densenet_e37

    Returns:
      Qkeras model of the densenet_e37
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

    for filter_index, filter_num in enumerate(self.__filters):
      for _ in range(0, self.__filter_repetition[filter_index]):
        qkeras_densenet = self.add_larq_quant_block(qkeras_densenet, 64)
      qkeras_densenet = self.add_connection_block(qkeras_densenet, filter_num)

    for _ in range(0, self.__filter_repetition[-1]):
      qkeras_densenet = self.add_larq_quant_block(qkeras_densenet, 64)

    qkeras_densenet = tf.keras.layers.BatchNormalization(momentum=0.9,
                                                         epsilon=1e-5)(
      qkeras_densenet)
    qkeras_densenet = tf.keras.layers.Activation("relu")(qkeras_densenet)
    qkeras_densenet = tf.keras.layers.MaxPool2D(pool_size=7)(qkeras_densenet)
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

  def build_larq_densenet_e37(self):
    """Build the larq version of the densenet_e28

    Returns:
      Larq model of the densenet_e28
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

    for filter_index, filter_num in enumerate(self.__filters):
      for _ in range(0, self.__filter_repetition[filter_index]):
        larq_densenet = self.add_larq_quant_block(larq_densenet, 64)
      larq_densenet = self.add_connection_block(larq_densenet, filter_num)

    for _ in range(0, self.__filter_repetition[-1]):
      larq_densenet = self.add_larq_quant_block(larq_densenet, 64)

    larq_densenet = tf.keras.layers.BatchNormalization(momentum=0.9,
                                                       epsilon=1e-5)(
      larq_densenet)
    larq_densenet = tf.keras.layers.Activation("relu")(larq_densenet)
    larq_densenet = tf.keras.layers.MaxPool2D(pool_size=7)(larq_densenet)
    larq_densenet = tf.keras.layers.Flatten()(larq_densenet)
    larq_densenet = tf.keras.layers.Dense(1000,
                                          kernel_initializer="glorot_normal")(
      larq_densenet)
    larq_densenet = tf.keras.layers.Activation("softmax", dtype="float32")(
      larq_densenet)
    larq_densenet = tf.keras.Model(inputs=input_layer, outputs=larq_densenet)
    larq_densenet.load_weights(self.__weights_path)
    return larq_densenet

  def build_qkeras_densenet_e45(self):
    """Build the qkeras version of the densenet_e45

    Returns:
      Qkeras model of the densenet_e45
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

    for filter_index, filter_num in enumerate(self.__filters):
      for _ in range(0, self.__filter_repetition[filter_index]):
        qkeras_densenet = self.add_larq_quant_block(qkeras_densenet, 64)
      qkeras_densenet = self.add_connection_block(qkeras_densenet, filter_num)

    for _ in range(0, self.__filter_repetition[-1]):
      qkeras_densenet = self.add_larq_quant_block(qkeras_densenet, 64)

    qkeras_densenet = tf.keras.layers.BatchNormalization(momentum=0.9,
                                                         epsilon=1e-5)(
      qkeras_densenet)
    qkeras_densenet = tf.keras.layers.Activation("relu")(qkeras_densenet)
    qkeras_densenet = tf.keras.layers.MaxPool2D(pool_size=7)(qkeras_densenet)
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

  def build_larq_densenet_e45(self):
    """Build the larq version of the densenet_e45

    Returns:
      Larq model of the densenet_e45
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

    for filter_index, filter_num in enumerate(self.__filters):
      for _ in range(0, self.__filter_repetition[filter_index]):
        larq_densenet = self.add_larq_quant_block(larq_densenet, 64)
      larq_densenet = self.add_connection_block(larq_densenet, filter_num)

    for _ in range(0, self.__filter_repetition[-1]):
      larq_densenet = self.add_larq_quant_block(larq_densenet, 64)

    larq_densenet = tf.keras.layers.BatchNormalization(momentum=0.9,
                                                       epsilon=1e-5)(
      larq_densenet)
    larq_densenet = tf.keras.layers.Activation("relu")(larq_densenet)
    larq_densenet = tf.keras.layers.MaxPool2D(pool_size=7)(larq_densenet)
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
  network_names = [DENSENET_E28_NAME, DENSENET_E37_NAME, DENSENET_E45_NAME]
  sizes = [28, 37, 45]

  # Create a random dataset with 100 samples
  random_data = create_random_dataset(100)
  for size, name in zip(sizes, network_names):
    network = DenseNet(size)
    qkeras_network, larq_network = network.build()
    # Compare mean MSE and Absolute error of the the networks
    compare_network(qkeras_network=qkeras_network, larq_network=larq_network,
                    dataset=random_data, network_name=name)
    dump_network_to_json(qkeras_network=qkeras_network,
                         larq_network=larq_network,
                         network_name=name)
