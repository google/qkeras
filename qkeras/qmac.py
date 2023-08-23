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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.keras import constraints
from .quantizers import get_quantizer

from tensorflow_model_optimization.python.core.sparsity.keras.prunable_layer import PrunableLayer
from .qlayers import get_auto_range_constraint_initializer


# QKeras needs to support more layers for matrix multiplication and shift
# operations such as in Tranformer. Such layers should be all placed here.


class QScaleShift(tf.keras.layers.Layer, PrunableLayer):
  """Quantized scale and shift layer.

  output = scale * x + bias where scale and bias are each of shape (1,).

  QScaleShift is similar to the special case in QDepthwiseConv2D
    where kernel_size=(1,1). However there are several differences:
  1) There is no concept of padding and striding in QScaleShift since
    it's not a conv layer;
  2) QDepthwiseConv2D expected min_ndim=4 for input shape; while QScaleShift
    input could be any shape;
  3) In QDepthwiseConv2D each output channel has its own weight value;
    while QScaleShift share the same weight across the entire input tensor.
  4) Since it's not a Conv operation, hardware implementation for
    QScaleShift and QDWConv2D is fundamentally different. Therefore it
    makes sense to separate them as two different types of layers.
  """

  def __init__(self,
               weight_quantizer=None,
               bias_quantizer=None,
               use_bias=True,
               activation=None,
               weight_initializer="he_normal",
               weight_regularizer=None,
               bias_initializer="zeros",
               bias_regularizer=None,
               **kwargs):

    super().__init__()
    self.use_bias = use_bias
    self.weight_regularizer = weight_regularizer
    self.bias_regularizer = bias_regularizer

    self.weight_quantizer = weight_quantizer
    self.bias_quantizer = bias_quantizer

    self.weight_quantizer_internal = get_quantizer(self.weight_quantizer)
    self.bias_quantizer_internal = get_quantizer(self.bias_quantizer)

    _, self.weight_initializer = (
        get_auto_range_constraint_initializer(
            self.weight_quantizer_internal, None,
            weight_initializer))

    _, self.bias_initializer = (
        get_auto_range_constraint_initializer(
            self.bias_quantizer_internal, None, bias_initializer))

    # optimize parameter set to "auto" scaling mode if possible
    if hasattr(self.weight_quantizer_internal, "_set_trainable_parameter"):
      self.weight_quantizer_internal._set_trainable_parameter()
    if hasattr(self.bias_quantizer_internal, "_set_trainable_parameter"):
      self.bias_quantizer_internal._set_trainable_parameter()

    self.quantizers = [self.weight_quantizer_internal,
                       self.bias_quantizer_internal]

    self.activation = get_quantizer(activation)

    super().__init__(**kwargs)

  def build(self, input_shape):
    self.weight = self.add_weight(
        name="weight", shape=(1, 1), dtype="float32",
        initializer=self.weight_initializer,
        regularizer=self.weight_regularizer, trainable=True)

    if self.use_bias:
      self.bias = self.add_weight(
          name="bias", shape=(1, 1), dtype="float32",
          initializer=self.bias_initializer, regularizer=self.bias_regularizer,
          trainable=True)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs):

    quantized_weight = (
        self.weight_quantizer_internal(self.weight) if
        self.weight_quantizer_internal is not None else self.weight)

    outputs = tf.math.multiply(inputs, quantized_weight)

    if self.use_bias:
      quantized_bias = (
          self.bias_quantizer_internal(self.bias) if
          self.bias_quantizer_internal is not None else self.bias)

      outputs = quantized_bias + outputs

    return self.activation(outputs) if self.activation is not None else outputs

  def get_config(self):
    config = {
        "weight_quantizer": constraints.serialize(
            self.weight_quantizer_internal
        ),
        "bias_quantizer": constraints.serialize(
            self.bias_quantizer_internal
        ),
        "weight_initializer": constraints.serialize(
            self.weight_initializer),
        "bias_initializer": constraints.serialize(
            self.bias_initializer),
        "activation": constraints.serialize(
            self.activation),
        "use_bias": self.use_bias,
        "weight_regularizer": constraints.serialize(
            self.weight_regularizer),
        "bias_regularizer": constraints.serialize(
            self.bias_regularizer),
    }
    base_config = super().get_config()
    base_config.update(config)
    return base_config

  def get_quantization_config(self):
    return {
        "weight_quantizer":
            str(self.weight_quantizer_internal),
        "bias_quantizer":
            str(self.bias_quantizer_internal),
        "activation":
            str(self.activation)
    }

  def get_quantizers(self):
    return self.quantizers

  def get_prunable_weights(self):
    return [self.weight, self.bias]
