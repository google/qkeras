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
import numpy as np
from tensorflow.keras import constraints

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from .qlayers import QActivation
from .quantizers import get_quantizer


class QAveragePooling2D(AveragePooling2D):
  """Computes the quantized version of AveragePooling2D."""

  def __init__(self, pool_size=(2, 2),
               strides=None,
               padding="valid",
               data_format=None,
               average_quantizer=None,
               activation=None,
               **kwargs):

    self.average_quantizer = average_quantizer
    self.average_quantizer_internal = get_quantizer(self.average_quantizer)
    self.quantizers = [self.average_quantizer_internal]

    if activation is not None:
      self.activation = get_quantizer(activation)
    else:
      self.activation = activation

    super(QAveragePooling2D, self).__init__(
        pool_size=pool_size, strides=strides, padding=padding,
        data_format=data_format, **kwargs)

  def call(self, inputs):
    """Performs quantized AveragePooling followed by QActivation.

    Since there is no specific parameter for averaging op, we couldn't apply
    averaging quantizer to the averaging op. We have two options:
    1. we perform our own average as sum first then multiply with the
       inversion
       of the division factor: sum(x) * quantize(1/pool_area)
    2. first, we call keras version of averaging first: y1 = keras_average(x)
       then multiply it with pool_size^2: y2 = y1 * pool_area
       Last, y3 = y2 * quantize(1/ pool_area)
    Our numerical anaysis suggests negligible error between 1 and 2. Therefore
    we use option #2 here for the simplicity of implementation.

    """

    x = super(QAveragePooling2D, self).call(inputs)

    if self.average_quantizer:
      if isinstance(self.pool_size, int):
        pool_area = self.pool_size * self.pool_size
      else:
        pool_area = np.prod(self.pool_size)

      # Revertes the division results.
      x = x * pool_area

      # Quantizes the multiplication factor.
      mult_factor = 1.0 / pool_area

      q_mult_factor = self.average_quantizer_internal(mult_factor)
      q_mult_factor = K.cast_to_floatx(q_mult_factor)
      x = x * q_mult_factor

    if self.activation is not None:
      return self.activation(x)
    return x

  def get_config(self):
    config = {
        "average_quantizer":
            constraints.serialize(self.average_quantizer_internal),
        "activation": constraints.serialize(self.activation),
    }
    base_config = super(QAveragePooling2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_quantization_config(self):
    return {
        "average_quantizer":
            str(self.average_quantizer_internal),
        "activation":
            str(self.activation)
    }

  def get_quantizers(self):
    return self.quantizers


class QGlobalAveragePooling2D(GlobalAveragePooling2D):
  """Computes the quantized version of GlobalAveragePooling2D."""

  def __init__(self, data_format=None,
               average_quantizer=None,
               activation=None,
               **kwargs):

    self.average_quantizer = average_quantizer
    self.average_quantizer_internal = get_quantizer(self.average_quantizer)
    self.quantizers = [self.average_quantizer_internal]

    if activation is not None:
      self.activation = get_quantizer(activation)
    else:
      self.activation = activation

    super(QGlobalAveragePooling2D, self).__init__(
        data_format=data_format, **kwargs)

  def compute_pooling_area(self, input_shape):
    input_shape = input_shape.as_list()
    if self.data_format == "channels_last":
      return input_shape[1] * input_shape[2]
    else:
      return input_shape[2] * input_shape[3]

  def call(self, inputs):
    """Performs quantized GlobalAveragePooling followed by QActivation.

    Since there is no specific parameter for averaging op, we couldn't apply
    averaging quantizer to the averaging op. We have two options:
    1. we perform our own average as sum first then multiply with the
       inversion
       of the division factor: sum(x) * quantize(1/pool_area)
    2. first, we call keras version of averaging first:
       y1 = keras_global_average(x)
       then multiply it with the denominator(pool_area) used by averaging:
       y2 = y1 * pool_area
       Last, y3 = y2 * quantize(1/ pool_area)
    Our numerical anaysis suggests negligible error between 1 and 2. Therefore
    we use option #2 here for the simplicity of implementation.

    """

    x = super(QGlobalAveragePooling2D, self).call(inputs)

    if self.average_quantizer:
      pool_area = self.compute_pooling_area(input_shape=inputs.shape)
      # Reverts the division results
      x = x * pool_area
      # Quantizes the multiplication factor
      mult_factor = 1.0 / pool_area
      q_mult_factor = self.average_quantizer_internal(mult_factor)
      x = x * q_mult_factor

    if self.activation is not None:
      return self.activation(x)
    return x

  def get_config(self):
    config = {
        "average_quantizer":
            constraints.serialize(self.average_quantizer_internal),
        "activation": constraints.serialize(self.activation),
    }
    base_config = super(QGlobalAveragePooling2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_quantization_config(self):
    return {
        "average_quantizer":
            str(self.average_quantizer_internal),
        "activation":
            str(self.activation)
    }

  def get_quantizers(self):
    return self.quantizers
