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
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import RNN
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU

import tensorflow.keras.backend as K
from .qlayers import get_auto_range_constraint_initializer
from .qlayers import QActivation
from .quantizers import get_quantized_initializer
from .quantizers import get_quantizer


class QRNN(RNN):
    """
    Quantized recurrent layer

    Most of these parameters follow the implementation of RNN in Keras,
    with the exception of kernel_quantizer and bias_quantizer, and
    kernel_initializer.
    
    kernel_quantizer: quantizer function/class for kernel
    bias_quantizer: quantizer function/class for bias
    kernel_range/bias_ranger: for quantizer functions whose values
        can go over [-1,+1], these values are used to set the clipping
        value of kernels and biases, respectively, instead of using the
        constraints specified by the user.
    
    we refer the reader to the documentation of Conv2D in Keras for the
    other parameters.

    See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
    for details about the usage of RNN API.
    
    """

    def __init__(self,
                 cell,
                 return_sequences=False,
                 return_state=False,
                 kernel_quantizer=None,
                 bias_quantizer=None,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 time_major=False,
                 **kwargs):
    if isinstance(cell, (list, tuple)):
      cell = StackedRNNCells(cell)
    if not 'call' in dir(cell):
      raise ValueError('`cell` should have a `call` method. '
                       'The RNN was passed:', cell)
    if not 'state_size' in dir(cell):
      raise ValueError('The RNN cell should have '
                       'an attribute `state_size` '
                       '(tuple of integers, '
                       'one integer per RNN state).')
    # If True, the output for masked timestep will be zeros, whereas in the
    # False case, output from previous timestep is returned for masked timestep.
    self.zero_output_for_mask = kwargs.pop('zero_output_for_mask', False)

    if 'input_shape' not in kwargs and (
        'input_dim' in kwargs or 'input_length' in kwargs):
      input_shape = (kwargs.pop('input_length', None),
                     kwargs.pop('input_dim', None))
      kwargs['input_shape'] = input_shape
    
    super(RNN, self).__init__(**kwargs)



    def call(self,
            inputs,
            mask=None,
            training=None,
            initial_state=None,
            constants=None):
        pass
