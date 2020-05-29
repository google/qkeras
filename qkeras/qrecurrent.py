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
""" Quantized Recurrent layers. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import SimpleRNNCell
from tensorflow.keras.layers import LSTMCell
from tensorflow.keras.layers import GRUCell
from tensorflow.keras.layers import RNN
from tensorflow.python.util import nest
from tensorflow.python.keras.engine.input_spec import InputSpec

import tensorflow.keras.backend as K
from .qlayers import get_auto_range_constraint_initializer
from .qlayers import QActivation
from .quantizers import get_quantized_initializer
from .quantizers import get_quantizer


class QSimpleRNNCell(SimpleRNNCell):
  """
  
  """
  def __init__(self,
               units,
               activation='quantized_tanh',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               kernel_quantizer=None,
               recurrent_quantizer=None,
               bias_quantizer=None,
               dropout=0.,
               recurrent_dropout=0.,
               **kwargs):

    self.kernel_quantizer = kernel_quantizer
    self.recurrent_quantizer = recurrent_quantizer # not needed
    self.bias_quantizer = bias_quantizer

    self.kernel_quantizer_internal = get_quantizer(self.kernel_quantizer)
    self.recurrent_quantizer_internal = get_quantizer(self.recurrent_quantizer)
    self.bias_quantizer_internal = get_quantizer(self.bias_quantizer)

    self.quantizers = [
      self.kernel_quantizer_internal,
      self.recurrent_quantizer_internal, 
      self.bias_quantizer_internal
    ]

    if activation is not None:
      activation = get_quantizer(activation)

    super(QSimpleRNNCell, self).__init__(
      units=units,
      activation=activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      recurrent_initializer=recurrent_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      recurrent_regularizer=recurrent_regularizer,
      bias_regularizer=bias_regularizer,
      kernel_constraint=kernel_constraint,
      recurrent_constraint=recurrent_constraint,
      bias_constraint=bias_constraint,
      dropout=dropout,
      recurrent_dropout=recurrent_dropout,
      **kwargs
    )


  def call(self, inputs, states, training=None):
    prev_output = states[0] if nest.is_sequence(states) else states
    dp_mask = self.get_dropout_mask_for_cell(inputs, training)
    rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
        prev_output, training)

    if self.kernel_quantizer:
      self.kernel_quantizer_internal.set_istraining_var(training)
      quantized_kernel = self.kernel_quantizer_internal(self.kernel)
    else:
      quantized_kernel = self.kernel

    if dp_mask is not None:
      h = K.dot(inputs * dp_mask, quantized_kernel)
    else:
      h = K.dot(inputs, quantized_kernel)
    
    if self.bias is not None:
      if self.bias_quantizer:
        self.bias_quantizer_internal.set_istraining_var(training)
        quantized_bias = self.bias_quantizer_internal(self.bias)
      else:
        quantized_bias = self.bias

      h = K.bias_add(h, quantized_bias)

    if rec_dp_mask is not None:
      prev_output = prev_output * rec_dp_mask

    if self.recurrent_quantizer:
      self.recurrent_quantizer_internal.set_istraining_var(training)
      quantized_recurrent = self.recurrent_quantizer_internal(self.recurrent_kernel)
    else:
      quantized_recurrent = self.recurrent_kernel

    output = h + K.dot(prev_output, quantized_recurrent)

    if self.activation is not None:
      output = self.activation(output)
    return output, [output]
  
  
  def get_config(self):
    config = {
        "kernel_quantizer":
            constraints.serialize(self.kernel_quantizer_internal),
        "recurrent_quantizer":
            constraints.serialize(self.recurrent_quantizer_internal),
        "bias_quantizer":
            constraints.serialize(self.bias_quantizer_internal)
    }
    base_config = super(QSimpleRNNCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))



class QSimpleRNN(RNN):
  """
  Quantized simple recurrent layer  
  """

  def __init__(self,
               units,
               activation='quantized_tanh',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               kernel_quantizer=None,
               recurrent_quantizer=None,
               bias_quantizer=None,
               dropout=0.,
               recurrent_dropout=0.,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               **kwargs):

    if 'enable_caching_device' in kwargs:
      cell_kwargs = {'enable_caching_device':
                     kwargs.pop('enable_caching_device')}
    else:
      cell_kwargs = {}

    cell = QSimpleRNNCell(
        units,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        kernel_quantizer=kernel_quantizer,
        recurrent_quantizer=recurrent_quantizer,
        bias_quantizer=bias_quantizer,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        dtype=kwargs.get('dtype'),
        trainable=kwargs.get('trainable', True),
        **cell_kwargs)


    super(QSimpleRNN, self).__init__(
        cell,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll,
        **kwargs)
    self.activity_regularizer = regularizers.get(activity_regularizer)
    self.input_spec = [InputSpec(ndim=3)]


  def call(self, inputs, mask=None, training=None, initial_state=None):
    self._maybe_reset_cell_dropout_mask(self.cell)
    return super(QSimpleRNN, self).call(
        inputs, mask=mask, training=training, initial_state=initial_state)


  def get_quantizers(self):
    return self.cell.quantizers

  @property
  def units(self):
    return self.cell.units

  @property
  def activation(self):
    return self.cell.activation

  @property
  def use_bias(self):
    return self.cell.use_bias

  @property
  def kernel_initializer(self):
    return self.cell.kernel_initializer

  @property
  def recurrent_initializer(self):
    return self.cell.recurrent_initializer

  @property
  def bias_initializer(self):
    return self.cell.bias_initializer

  @property
  def kernel_regularizer(self):
    return self.cell.kernel_regularizer

  @property
  def recurrent_regularizer(self):
    return self.cell.recurrent_regularizer

  @property
  def bias_regularizer(self):
    return self.cell.bias_regularizer

  @property
  def kernel_constraint(self):
    return self.cell.kernel_constraint

  @property
  def recurrent_constraint(self):
    return self.cell.recurrent_constraint

  @property
  def bias_constraint(self):
    return self.cell.bias_constraint

  @property
  def dropout(self):
    return self.cell.dropout

  @property
  def recurrent_dropout(self):
    return self.cell.recurrent_dropout

  @property
  def kernel_quantizer(self):
    return self.cell.kernel_quantizer

  @property
  def recurrent_quantizer(self):
    return self.cell.recurrent_quantizer

  @property
  def bias_quantizer(self):
    return self.cell.bias_quantizer

  def get_config(self):
    config = {
        "kernel_quantizer":
            constraints.serialize(self.kernel_quantizer),
        "recurrent_quantizer":
            constraints.serialize(self.recurrent_quantizer),
        "bias_quantizer":
            constraints.serialize(self.bias_quantizer)
    }
    base_config = super(QSimpleRNN, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
