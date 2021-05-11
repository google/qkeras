# Copyright 2020 Google LLC
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
from tensorflow.keras.layers import Bidirectional
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow_model_optimization.python.core.sparsity.keras.prunable_layer import PrunableLayer

import tensorflow.keras.backend as K
from .qlayers import get_auto_range_constraint_initializer
from .qlayers import QActivation
from .quantizers import get_quantized_initializer
from .quantizers import get_quantizer


class QSimpleRNNCell(SimpleRNNCell):
  """
  Cell class for the QSimpleRNNCell layer.

  Most of these parameters follow the implementation of SimpleRNNCell in
  Keras, with the exception of kernel_quantizer, recurrent_quantizer,
  bias_quantizer, and state_quantizer.

  kernel_quantizer: quantizer function/class for kernel
  recurrent_quantizer: quantizer function/class for recurrent kernel
  bias_quantizer: quantizer function/class for bias
  state_quantizer: quantizer function/class for states

  We refer the reader to the documentation of SimpleRNNCell in Keras for the
  other parameters.

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
               state_quantizer=None,
               dropout=0.,
               recurrent_dropout=0.,
               **kwargs):

    self.kernel_quantizer = kernel_quantizer
    self.recurrent_quantizer = recurrent_quantizer
    self.bias_quantizer = bias_quantizer
    self.state_quantizer = state_quantizer

    self.kernel_quantizer_internal = get_quantizer(self.kernel_quantizer)
    self.recurrent_quantizer_internal = get_quantizer(self.recurrent_quantizer)
    self.bias_quantizer_internal = get_quantizer(self.bias_quantizer)
    self.state_quantizer_internal = get_quantizer(self.state_quantizer)

    self.quantizers = [
        self.kernel_quantizer_internal, self.recurrent_quantizer_internal,
        self.bias_quantizer_internal, self.state_quantizer_internal
    ]

    if hasattr(self.kernel_quantizer_internal, "_set_trainable_parameter"):
      self.kernel_quantizer_internal._set_trainable_parameter()

    if hasattr(self.recurrent_quantizer_internal, "_set_trainable_parameter"):
      self.recurrent_quantizer_internal._set_trainable_parameter()

    kernel_constraint, kernel_initializer = (
        get_auto_range_constraint_initializer(self.kernel_quantizer_internal,
                                              kernel_constraint,
                                              kernel_initializer))

    recurrent_constraint, recurrent_initializer = (
        get_auto_range_constraint_initializer(self.recurrent_quantizer_internal,
                                              recurrent_constraint,
                                              recurrent_initializer))

    if use_bias:
      bias_constraint, bias_initializer = (
          get_auto_range_constraint_initializer(self.bias_quantizer_internal,
                                                bias_constraint,
                                                bias_initializer))

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

    if self.state_quantizer:
      quantized_prev_output = self.state_quantizer_internal(prev_output)
    else:
      quantized_prev_output = prev_output

    if self.kernel_quantizer:
      quantized_kernel = self.kernel_quantizer_internal(self.kernel)
    else:
      quantized_kernel = self.kernel

    if dp_mask is not None:
      h = K.dot(inputs * dp_mask, quantized_kernel)
    else:
      h = K.dot(inputs, quantized_kernel)

    if self.bias is not None:
      if self.bias_quantizer:
        quantized_bias = self.bias_quantizer_internal(self.bias)
      else:
        quantized_bias = self.bias

      h = K.bias_add(h, quantized_bias)

    if rec_dp_mask is not None:
      quantized_prev_output = quantized_prev_output * rec_dp_mask

    if self.recurrent_quantizer:
      quantized_recurrent = self.recurrent_quantizer_internal(self.recurrent_kernel)
    else:
      quantized_recurrent = self.recurrent_kernel

    output = h + K.dot(quantized_prev_output, quantized_recurrent)

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
            constraints.serialize(self.bias_quantizer_internal),
        "state_quantizer":
            constraints.serialize(self.state_quantizer_internal)
    }
    base_config = super(QSimpleRNNCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class QSimpleRNN(RNN, PrunableLayer):
  """
  Class for the QSimpleRNN layer.

  Most of these parameters follow the implementation of SimpleRNN in
  Keras, with the exception of kernel_quantizer, recurrent_quantizer,
  bias_quantizer and state_quantizer.


  kernel_quantizer: quantizer function/class for kernel
  recurrent_quantizer: quantizer function/class for recurrent kernel
  bias_quantizer: quantizer function/class for bias
  state_quantizer: quantizer function/class for states


  We refer the reader to the documentation of SimpleRNN in Keras for the
  other parameters.

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
               state_quantizer=None,
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
        state_quantizer=state_quantizer,
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
    self.input_spec = [tf.keras.layers.InputSpec(ndim=3)]

  def call(self, inputs, mask=None, training=None, initial_state=None):
    self._maybe_reset_cell_dropout_mask(self.cell)
    return super(QSimpleRNN, self).call(
        inputs, mask=mask, training=training, initial_state=initial_state)

  def get_quantizers(self):
    return self.cell.quantizers

  def get_prunable_weights(self):
    return [self.cell.kernel, self.cell.recurrent_kernel]

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
  def kernel_quantizer_internal(self):
    return self.cell.kernel_quantizer_internal

  @property
  def recurrent_quantizer_internal(self):
    return self.cell.recurrent_quantizer_internal

  @property
  def bias_quantizer_internal(self):
    return self.cell.bias_quantizer_internal

  @property
  def state_quantizer_internal(self):
    return self.cell.state_quantizer_internal

  @property
  def kernel_quantizer(self):
    return self.cell.kernel_quantizer

  @property
  def recurrent_quantizer(self):
    return self.cell.recurrent_quantizer

  @property
  def bias_quantizer(self):
    return self.cell.bias_quantizer

  @property
  def state_quantizer(self):
    return self.cell.state_quantizer

  @property
  def dropout(self):
    return self.cell.dropout

  @property
  def recurrent_dropout(self):
    return self.cell.recurrent_dropout

  def get_config(self):
    config = {
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        "kernel_quantizer":
            constraints.serialize(self.kernel_quantizer_internal),
        "recurrent_quantizer":
            constraints.serialize(self.recurrent_quantizer_internal),
        "bias_quantizer":
            constraints.serialize(self.bias_quantizer_internal),
        "state_quantizer":
            constraints.serialize(self.state_quantizer_internal),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout
    }
    base_config = super(QSimpleRNN, self).get_config()
    del base_config['cell']
    return dict(list(base_config.items()) + list(config.items()))

  def get_quantization_config(self):
    return {
        "kernel_quantizer":
            str(self.kernel_quantizer_internal),
        "recurrent_quantizer":
            str(self.recurrent_quantizer_internal),
        "bias_quantizer":
            str(self.bias_quantizer_internal),
        "state_quantizer":
            str(self.state_quantizer_internal),
        "activation":
            str(self.activation)
    }

  @classmethod
  def from_config(cls, config):
    if 'implementation' in config:
      config.pop('implementation')
    return cls(**config)


class QLSTMCell(LSTMCell):
  """
  Cell class for the QLSTMCell layer.

  Most of these parameters follow the implementation of LSTMCell in
  Keras, with the exception of kernel_quantizer, recurrent_quantizer,
  bias_quantizer, state_quantizer.


  kernel_quantizer: quantizer function/class for kernel
  recurrent_quantizer: quantizer function/class for recurrent kernel
  bias_quantizer: quantizer function/class for bias
  state_quantizer: quantizer function/class for states

  We refer the reader to the documentation of LSTMCell in Keras for the
  other parameters.

  """

  def __init__(self,
               units,
               activation='quantized_tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               kernel_quantizer=None,
               recurrent_quantizer=None,
               bias_quantizer=None,
               state_quantizer=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=1,
               **kwargs):
    self.kernel_quantizer = kernel_quantizer
    self.recurrent_quantizer = recurrent_quantizer
    self.bias_quantizer = bias_quantizer
    self.state_quantizer = state_quantizer

    self.kernel_quantizer_internal = get_quantizer(self.kernel_quantizer)
    self.recurrent_quantizer_internal = get_quantizer(self.recurrent_quantizer)
    self.bias_quantizer_internal = get_quantizer(self.bias_quantizer)
    self.state_quantizer_internal = get_quantizer(self.state_quantizer)

    self.quantizers = [
      self.kernel_quantizer_internal,
      self.recurrent_quantizer_internal,
      self.bias_quantizer_internal,
      self.state_quantizer_internal,
    ]

    if hasattr(self.kernel_quantizer_internal, "_set_trainable_parameter"):
      self.kernel_quantizer_internal._set_trainable_parameter()

    if hasattr(self.recurrent_quantizer_internal, "_set_trainable_parameter"):
      self.recurrent_quantizer_internal._set_trainable_parameter()

    kernel_constraint, kernel_initializer = (
        get_auto_range_constraint_initializer(self.kernel_quantizer_internal,
                                              kernel_constraint,
                                              kernel_initializer))

    recurrent_constraint, recurrent_initializer = (
        get_auto_range_constraint_initializer(self.recurrent_quantizer_internal,
                                              recurrent_constraint,
                                              recurrent_initializer))

    if use_bias:
      bias_constraint, bias_initializer = (
          get_auto_range_constraint_initializer(self.bias_quantizer_internal,
                                                bias_constraint,
                                                bias_initializer))

    if activation is not None:
      activation = get_quantizer(activation)

    if recurrent_activation is not None:
      recurrent_activation = get_quantizer(recurrent_activation)

    super(QLSTMCell, self).__init__(
      units=units,
      activation=activation,
      use_bias=use_bias,
      recurrent_activation=recurrent_activation,
      kernel_initializer=kernel_initializer,
      recurrent_initializer=recurrent_initializer,
      bias_initializer=bias_initializer,
      unit_forget_bias=True,
      kernel_regularizer=kernel_regularizer,
      recurrent_regularizer=recurrent_regularizer,
      bias_regularizer=bias_regularizer,
      kernel_constraint=kernel_constraint,
      recurrent_constraint=recurrent_constraint,
      bias_constraint=bias_constraint,
      dropout=dropout,
      recurrent_dropout=recurrent_dropout,
      implementation=implementation,
      **kwargs
    )

  def _compute_carry_and_output(self, x, h_tm1, c_tm1, quantized_recurrent):
    """Computes carry and output using split kernels."""
    x_i, x_f, x_c, x_o = x
    h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
    i = self.recurrent_activation(
        x_i + K.dot(h_tm1_i, quantized_recurrent[:, :self.units]))
    f = self.recurrent_activation(x_f + K.dot(
        h_tm1_f, quantized_recurrent[:, self.units:self.units * 2]))
    c = f * c_tm1 + i * self.activation(x_c + K.dot(
        h_tm1_c, quantized_recurrent[:, self.units * 2:self.units * 3]))
    o = self.recurrent_activation(
        x_o + K.dot(h_tm1_o, quantized_recurrent[:, self.units * 3:]))
    return c, o

  def _compute_carry_and_output_fused(self, z, c_tm1):
    """Computes carry and output using fused kernels."""
    z0, z1, z2, z3 = z
    i = self.recurrent_activation(z0)
    f = self.recurrent_activation(z1)
    c = f * c_tm1 + i * self.activation(z2)
    o = self.recurrent_activation(z3)
    return c, o

  def call(self, inputs, states, training=None):
    h_tm1_tmp = states[0]  # previous memory state
    c_tm1_tmp = states[1]  # previous carry state

    if self.state_quantizer:
      c_tm1 = self.state_quantizer_internal(c_tm1_tmp)
      h_tm1 = self.state_quantizer_internal(h_tm1_tmp)
    else:
      c_tm1 = c_tm1_tmp
      h_tm1 = h_tm1_tmp

    dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
    rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
        h_tm1, training, count=4)

    if self.kernel_quantizer:
      quantized_kernel = self.kernel_quantizer_internal(self.kernel)
    else:
      quantized_kernel = self.kernel
    if self.recurrent_quantizer:
      quantized_recurrent = self.recurrent_quantizer_internal(self.recurrent_kernel)
    else:
      quantized_recurrent = self.recurrent_kernel
    if self.bias_quantizer:
      quantized_bias = self.bias_quantizer_internal(self.bias)
    else:
      quantized_bias = self.bias

    if self.implementation == 1:
      if 0 < self.dropout < 1.:
        inputs_i = inputs * dp_mask[0]
        inputs_f = inputs * dp_mask[1]
        inputs_c = inputs * dp_mask[2]
        inputs_o = inputs * dp_mask[3]
      else:
        inputs_i = inputs
        inputs_f = inputs
        inputs_c = inputs
        inputs_o = inputs
      k_i, k_f, k_c, k_o = array_ops.split(
          quantized_kernel, num_or_size_splits=4, axis=1)
      x_i = K.dot(inputs_i, k_i)
      x_f = K.dot(inputs_f, k_f)
      x_c = K.dot(inputs_c, k_c)
      x_o = K.dot(inputs_o, k_o)
      if self.use_bias:
        b_i, b_f, b_c, b_o = array_ops.split(
            quantized_bias, num_or_size_splits=4, axis=0)
        x_i = K.bias_add(x_i, b_i)
        x_f = K.bias_add(x_f, b_f)
        x_c = K.bias_add(x_c, b_c)
        x_o = K.bias_add(x_o, b_o)

      if 0 < self.recurrent_dropout < 1.:
        h_tm1_i = h_tm1 * rec_dp_mask[0]
        h_tm1_f = h_tm1 * rec_dp_mask[1]
        h_tm1_c = h_tm1 * rec_dp_mask[2]
        h_tm1_o = h_tm1 * rec_dp_mask[3]
      else:
        h_tm1_i = h_tm1
        h_tm1_f = h_tm1
        h_tm1_c = h_tm1
        h_tm1_o = h_tm1
      x = (x_i, x_f, x_c, x_o)
      h_tm1 = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)
      c, o = self._compute_carry_and_output(x, h_tm1, c_tm1, quantized_recurrent)
    else:
      if 0. < self.dropout < 1.:
        inputs = inputs * dp_mask[0]
      z = K.dot(inputs, quantized_kernel)
      z += K.dot(h_tm1, quantized_recurrent)
      if self.use_bias:
        z = K.bias_add(z, quantized_bias)

      z = array_ops.split(z, num_or_size_splits=4, axis=1)
      c, o = self._compute_carry_and_output_fused(z, c_tm1)

    h = o * self.activation(c)
    return h, [h, c]

  def get_config(self):
    config = {
        "kernel_quantizer":
            constraints.serialize(self.kernel_quantizer_internal),
        "recurrent_quantizer":
            constraints.serialize(self.recurrent_quantizer_internal),
        "bias_quantizer":
            constraints.serialize(self.bias_quantizer_internal),
        "state_quantizer":
            constraints.serialize(self.state_quantizer_internal)
    }
    base_config = super(QLSTMCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class QLSTM(RNN, PrunableLayer):
  """
  Class for the QLSTM layer.

  Most of these parameters follow the implementation of LSTM in
  Keras, with the exception of kernel_quantizer, recurrent_quantizer,
  bias_quantizer, state_quantizer.


  kernel_quantizer: quantizer function/class for kernel
  recurrent_quantizer: quantizer function/class for recurrent kernel
  bias_quantizer: quantizer function/class for bias
  state_quantizer: quantizer function/class for states

  We refer the reader to the documentation of LSTM in Keras for the
  other parameters.

  """

  def __init__(self,
               units,
               activation='quantized_tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
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
               state_quantizer=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=1,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               **kwargs):
    if implementation == 0:
      print('`implementation=0` has been deprecated, '
              'and now defaults to `implementation=1`.'
              'Please update your layer call.')

    if 'enable_caching_device' in kwargs:
      cell_kwargs = {'enable_caching_device':
                     kwargs.pop('enable_caching_device')}
    else:
      cell_kwargs = {}

    cell = QLSTMCell(
        units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        unit_forget_bias=unit_forget_bias,
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
        state_quantizer=state_quantizer,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=implementation,
        dtype=kwargs.get('dtype'),
        trainable=kwargs.get('trainable', True),
        **cell_kwargs)

    super(QLSTM, self).__init__(
        cell,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll,
        **kwargs)
    self.activity_regularizer = regularizers.get(activity_regularizer)
    self.input_spec = [tf.keras.layers.InputSpec(ndim=3)]

  def call(self, inputs, mask=None, training=None, initial_state=None):
    self._maybe_reset_cell_dropout_mask(self.cell)
    return super(QLSTM, self).call(
        inputs, mask=mask, training=training, initial_state=initial_state)

  def get_quantizers(self):
    return self.cell.quantizers

  def get_prunable_weights(self):
    return [self.cell.kernel, self.cell.recurrent_kernel]

  @property
  def units(self):
    return self.cell.units

  @property
  def activation(self):
    return self.cell.activation

  @property
  def recurrent_activation(self):
    return self.cell.recurrent_activation

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
  def unit_forget_bias(self):
    return self.cell.unit_forget_bias

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
  def kernel_quantizer_internal(self):
    return self.cell.kernel_quantizer_internal

  @property
  def recurrent_quantizer_internal(self):
    return self.cell.recurrent_quantizer_internal

  @property
  def bias_quantizer_internal(self):
    return self.cell.bias_quantizer_internal

  @property
  def state_quantizer_internal(self):
    return self.cell.state_quantizer_internal

  @property
  def kernel_quantizer(self):
    return self.cell.kernel_quantizer

  @property
  def recurrent_quantizer(self):
    return self.cell.recurrent_quantizer

  @property
  def bias_quantizer(self):
    return self.cell.bias_quantizer

  @property
  def state_quantizer(self):
    return self.cell.state_quantizer

  @property
  def dropout(self):
    return self.cell.dropout

  @property
  def recurrent_dropout(self):
    return self.cell.recurrent_dropout

  @property
  def implementation(self):
    return self.cell.implementation

  def get_config(self):
    config = {
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'recurrent_activation':
            activations.serialize(self.recurrent_activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'unit_forget_bias':
            self.unit_forget_bias,
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        "kernel_quantizer":
            constraints.serialize(self.kernel_quantizer_internal),
        "recurrent_quantizer":
            constraints.serialize(self.recurrent_quantizer_internal),
        "bias_quantizer":
            constraints.serialize(self.bias_quantizer_internal),
        "state_quantizer":
            constraints.serialize(self.state_quantizer_internal),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout,
        'implementation':
            self.implementation
    }
    base_config = super(QLSTM, self).get_config()
    del base_config['cell']
    return dict(list(base_config.items()) + list(config.items()))

  def get_quantization_config(self):
    return {
        "kernel_quantizer":
            str(self.kernel_quantizer_internal),
        "recurrent_quantizer":
            str(self.recurrent_quantizer_internal),
        "bias_quantizer":
            str(self.bias_quantizer_internal),
        "state_quantizer":
            str(self.state_quantizer_internal),
        "activation":
            str(self.activation),
        "recurrent_activation":
            str(self.recurrent_activation),
    }

  @classmethod
  def from_config(cls, config):
    if 'implementation' in config and config['implementation'] == 0:
      config['implementation'] = 1
    return cls(**config)


class QGRUCell(GRUCell):
  """
  Cell class for the QGRUCell layer.

  Most of these parameters follow the implementation of GRUCell in
  Keras, with the exception of kernel_quantizer, recurrent_quantizer,
  bias_quantizer and state_quantizer.


  kernel_quantizer: quantizer function/class for kernel
  recurrent_quantizer: quantizer function/class for recurrent kernel
  bias_quantizer: quantizer function/class for bias
  state_quantizer: quantizer function/class for states


  We refer the reader to the documentation of GRUCell in Keras for the
  other parameters.

  """
  def __init__(self,
               units,
               activation='quantized_tanh',
               recurrent_activation='hard_sigmoid',
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
               state_quantizer=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=1,
               reset_after=False,
               **kwargs):

    self.kernel_quantizer = kernel_quantizer
    self.recurrent_quantizer = recurrent_quantizer
    self.bias_quantizer = bias_quantizer
    self.state_quantizer = state_quantizer

    self.kernel_quantizer_internal = get_quantizer(self.kernel_quantizer)
    self.recurrent_quantizer_internal = get_quantizer(self.recurrent_quantizer)
    self.bias_quantizer_internal = get_quantizer(self.bias_quantizer)
    self.state_quantizer_internal = get_quantizer(self.state_quantizer)

    self.quantizers = [
      self.kernel_quantizer_internal,
      self.recurrent_quantizer_internal,
      self.bias_quantizer_internal,
      self.state_quantizer_internal
    ]

    if hasattr(self.kernel_quantizer_internal, "_set_trainable_parameter"):
      self.kernel_quantizer_internal._set_trainable_parameter()

    if hasattr(self.recurrent_quantizer_internal, "_set_trainable_parameter"):
      self.recurrent_quantizer_internal._set_trainable_parameter()

    kernel_constraint, kernel_initializer = (
        get_auto_range_constraint_initializer(self.kernel_quantizer_internal,
                                              kernel_constraint,
                                              kernel_initializer))

    recurrent_constraint, recurrent_initializer = (
        get_auto_range_constraint_initializer(self.recurrent_quantizer_internal,
                                              recurrent_constraint,
                                              recurrent_initializer))

    if use_bias:
      bias_constraint, bias_initializer = (
          get_auto_range_constraint_initializer(self.bias_quantizer_internal,
                                                bias_constraint,
                                                bias_initializer))

    if activation is not None:
      activation = get_quantizer(activation)

    if recurrent_activation is not None:
      recurrent_activation = get_quantizer(recurrent_activation)

    super(QGRUCell, self).__init__(
      units=units,
      activation=activation,
      recurrent_activation=recurrent_activation,
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
      implementation=implementation,
      reset_after=reset_after,
      **kwargs)

  def call(self, inputs, states, training=None):
    # previous memory
    h_tm1_tmp = states[0] if nest.is_sequence(states) else states

    dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=3)
    rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
        h_tm1_tmp, training, count=3)

    if self.state_quantizer:
      h_tm1 = self.state_quantizer_internal(h_tm1_tmp)
    else:
      h_tm1 = h_tm1_tmp

    if self.kernel_quantizer:
      quantized_kernel = self.kernel_quantizer_internal(self.kernel)
    else:
      quantized_kernel = self.kernel
    if self.recurrent_quantizer:
      quantized_recurrent = self.recurrent_quantizer_internal(self.recurrent_kernel)
    else:
      quantized_recurrent = self.kernel

    if self.use_bias:
      if self.bias_quantizer:
        quantized_bias = self.bias_quantizer_internal(self.bias)
      else:
        quantized_bias = self.bias

      if not self.reset_after:
        input_bias, recurrent_bias = quantized_bias, None
      else:
        input_bias, recurrent_bias = array_ops.unstack(quantized_bias)

    if self.implementation == 1:
      if 0. < self.dropout < 1.:
        inputs_z = inputs * dp_mask[0]
        inputs_r = inputs * dp_mask[1]
        inputs_h = inputs * dp_mask[2]
      else:
        inputs_z = inputs
        inputs_r = inputs
        inputs_h = inputs

      x_z = K.dot(inputs_z, quantized_kernel[:, :self.units])
      x_r = K.dot(inputs_r, quantized_kernel[:, self.units:self.units * 2])
      x_h = K.dot(inputs_h, quantized_kernel[:, self.units * 2:])

      if self.use_bias:
        x_z = K.bias_add(x_z, input_bias[:self.units])
        x_r = K.bias_add(x_r, input_bias[self.units: self.units * 2])
        x_h = K.bias_add(x_h, input_bias[self.units * 2:])

      if 0. < self.recurrent_dropout < 1.:
        h_tm1_z = h_tm1 * rec_dp_mask[0]
        h_tm1_r = h_tm1 * rec_dp_mask[1]
        h_tm1_h = h_tm1 * rec_dp_mask[2]
      else:
        h_tm1_z = h_tm1
        h_tm1_r = h_tm1
        h_tm1_h = h_tm1

      recurrent_z = K.dot(h_tm1_z, quantized_recurrent[:, :self.units])
      recurrent_r = K.dot(h_tm1_r,
                          quantized_recurrent[:, self.units:self.units * 2])
      if self.reset_after and self.use_bias:
        recurrent_z = K.bias_add(recurrent_z, recurrent_bias[:self.units])
        recurrent_r = K.bias_add(recurrent_r,
                                 recurrent_bias[self.units:self.units * 2])

      z = self.recurrent_activation(x_z + recurrent_z)
      r = self.recurrent_activation(x_r + recurrent_r)

      # reset gate applied after/before matrix multiplication
      if self.reset_after:
        recurrent_h = K.dot(h_tm1_h, quantized_recurrent[:, self.units * 2:])
        if self.use_bias:
          recurrent_h = K.bias_add(recurrent_h, recurrent_bias[self.units * 2:])
        recurrent_h = r * recurrent_h
      else:
        recurrent_h = K.dot(r * h_tm1_h,
                            quantized_recurrent[:, self.units * 2:])

      hh = self.activation(x_h + recurrent_h)
    else:
      if 0. < self.dropout < 1.:
        inputs = inputs * dp_mask[0]

      # inputs projected by all gate matrices at once
      matrix_x = K.dot(inputs, quantized_kernel)
      if self.use_bias:
        # biases: bias_z_i, bias_r_i, bias_h_i
        matrix_x = K.bias_add(matrix_x, input_bias)

      x_z, x_r, x_h = array_ops.split(matrix_x, 3, axis=-1)

      if self.reset_after:
        # hidden state projected by all gate matrices at once
        matrix_inner = K.dot(h_tm1, quantized_recurrent)
        if self.use_bias:
          matrix_inner = K.bias_add(matrix_inner, recurrent_bias)
      else:
        # hidden state projected separately for update/reset and new
        matrix_inner = K.dot(h_tm1, quantized_recurrent[:, :2 * self.units])

      recurrent_z, recurrent_r, recurrent_h = array_ops.split(
          matrix_inner, [self.units, self.units, -1], axis=-1)

      z = self.recurrent_activation(x_z + recurrent_z)
      r = self.recurrent_activation(x_r + recurrent_r)

      if self.reset_after:
        recurrent_h = r * recurrent_h
      else:
        recurrent_h = K.dot(r * h_tm1,
                            quantized_recurrent[:, 2 * self.units:])

      hh = self.activation(x_h + recurrent_h)
    # previous and candidate state mixed by update gate
    h = z * h_tm1 + (1 - z) * hh
    return h, [h]

  def get_config(self):
    config = {
        "kernel_quantizer":
            constraints.serialize(self.kernel_quantizer_internal),
        "recurrent_quantizer":
            constraints.serialize(self.recurrent_quantizer_internal),
        "bias_quantizer":
            constraints.serialize(self.bias_quantizer_internal),
        "state_quantizer":
            constraints.serialize(self.state_quantizer_internal)
    }
    base_config = super(QGRUCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class QGRU(RNN, PrunableLayer):
  """
  Class for the QGRU layer.

  Most of these parameters follow the implementation of GRU in
  Keras, with the exception of kernel_quantizer, recurrent_quantizer,
  bias_quantizer and state_quantizer.


  kernel_quantizer: quantizer function/class for kernel
  recurrent_quantizer: quantizer function/class for recurrent kernel
  bias_quantizer: quantizer function/class for bias
  state_quantizer: quantizer function/class for states


  We refer the reader to the documentation of GRU in Keras for the
  other parameters.

  """

  def __init__(self,
               units,
               activation='quantized_tanh',
               recurrent_activation='hard_sigmoid',
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
               state_quantizer=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=1,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               reset_after=False,
               **kwargs):
    if implementation == 0:
      print('`implementation=0` has been deprecated, '
              'and now defaults to `implementation=1`.'
              'Please update your layer call.')

    if 'enable_caching_device' in kwargs:
      cell_kwargs = {'enable_caching_device':
                     kwargs.pop('enable_caching_device')}
    else:
      cell_kwargs = {}

    cell = QGRUCell(
        units,
        activation=activation,
        recurrent_activation=recurrent_activation,
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
        state_quantizer=state_quantizer,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=implementation,
        reset_after=reset_after,
        dtype=kwargs.get('dtype'),
        trainable=kwargs.get('trainable', True),
        **cell_kwargs)

    super(QGRU, self).__init__(
        cell,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll,
        **kwargs)
    self.activity_regularizer = regularizers.get(activity_regularizer)
    self.input_spec = [tf.keras.layers.InputSpec(ndim=3)]

  def call(self, inputs, mask=None, training=None, initial_state=None):
    self._maybe_reset_cell_dropout_mask(self.cell)
    return super(QGRU, self).call(
        inputs, mask=mask, training=training, initial_state=initial_state)

  def get_quantizers(self):
    return self.cell.quantizers

  def get_prunable_weights(self):
    return [self.cell.kernel, self.cell.recurrent_kernel]

  @property
  def units(self):
    return self.cell.units

  @property
  def activation(self):
    return self.cell.activation

  @property
  def recurrent_activation(self):
    return self.cell.recurrent_activation

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
  def kernel_quantizer_internal(self):
    return self.cell.kernel_quantizer_internal

  @property
  def recurrent_quantizer_internal(self):
    return self.cell.recurrent_quantizer_internal

  @property
  def bias_quantizer_internal(self):
    return self.cell.bias_quantizer_internal

  @property
  def state_quantizer_internal(self):
    return self.cell.state_quantizer_internal

  @property
  def kernel_quantizer(self):
    return self.cell.kernel_quantizer

  @property
  def recurrent_quantizer(self):
    return self.cell.recurrent_quantizer

  @property
  def bias_quantizer(self):
    return self.cell.bias_quantizer

  @property
  def state_quantizer(self):
    return self.cell.state_quantizer

  @property
  def dropout(self):
    return self.cell.dropout

  @property
  def recurrent_dropout(self):
    return self.cell.recurrent_dropout

  @property
  def implementation(self):
    return self.cell.implementation

  @property
  def reset_after(self):
    return self.cell.reset_after

  def get_config(self):
    config = {
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'recurrent_activation':
            activations.serialize(self.recurrent_activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        "kernel_quantizer":
            constraints.serialize(self.kernel_quantizer_internal),
        "recurrent_quantizer":
            constraints.serialize(self.recurrent_quantizer_internal),
        "bias_quantizer":
            constraints.serialize(self.bias_quantizer_internal),
        "state_quantizer":
            constraints.serialize(self.state_quantizer_internal),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout,
        'implementation':
            self.implementation,
        'reset_after':
            self.reset_after
    }
    base_config = super(QGRU, self).get_config()
    del base_config['cell']
    return dict(list(base_config.items()) + list(config.items()))

  def get_quantization_config(self):
    return {
        "kernel_quantizer":
            str(self.kernel_quantizer_internal),
        "recurrent_quantizer":
            str(self.recurrent_quantizer_internal),
        "bias_quantizer":
            str(self.bias_quantizer_internal),
        "state_quantizer":
            str(self.state_quantizer_internal),
        "activation":
            str(self.activation),
        "recurrent_activation":
            str(self.recurrent_activation),
    }

  @classmethod
  def from_config(cls, config):
    if 'implementation' in config and config['implementation'] == 0:
      config['implementation'] = 1
    return cls(**config)


class QBidirectional(Bidirectional):
  """
  Class for the QBidirecitonal wrapper.

  Most of these parameters follow the implementation of Bidirectional in
  Keras.

  We refer the reader to the documentation of Bidirectional in Keras for the
  other parameters.

  """
  def get_quantizers(self):
    """
    Returns quantizers in the order they were created.
    """
    return self.forward_layer.get_quantizers() + self.backward_layer.get_quantizers()

  @property
  def activation(self):
    return self.layer.activation

  def get_quantization_config(self):
    return {
      "layer" : self.layer.get_quantization_config(),
      "backward_layer" : self.backward_layer.get_quantization_config()
    }
