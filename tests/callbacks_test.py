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
"""Tests for callbacks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest
from numpy.testing import assert_equal
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow.compat.v2 as tf

from qkeras import *
from qkeras.utils import get_model_sparsity
from qkeras.utils import model_quantize
from qkeras.callbacks import QNoiseScheduler


def qconv_model():
  x = x_in = tf.keras.layers.Input((4, 4, 1), name="input")
  x = QConv2D(
      1,
      2,
      1,
      kernel_quantizer=quantized_bits(6, 2, 1, alpha=1.0),
      bias_quantizer=quantized_bits(4, 0, 1),
      name="qconv2d_1")(
          x)
  x = QActivation("quantized_relu(4)", name="QA_1")(x)
  model = keras.Model(inputs=[x_in], outputs=[x])
  return model


def test_QNoiseScheduler():
  model = qconv_model()
  model.compile(optimizer="sgd", loss=tf.keras.losses.MeanSquaredError())
  num_data = 5
  x_train = np.random.rand(num_data, 4, 4, 1)
  y_train = np.random.rand(num_data, 1)

  #########################
  # Test "step" freq_type #
  #########################

  # The number of batch passes the finish of 4.
  gradual_qnoise_callback_0 = QNoiseScheduler(
      start=2, finish=4, freq_type="step", exponent=3.0)

  model.fit(
      x_train,
      y_train,
      batch_size=1,
      epochs=1,
      verbose=0,
      callbacks=[
          gradual_qnoise_callback_0,
      ],
  )

  # QConv2D has a kernel_quantizer and a bias_quantizer, and QActivation has a
  # quantizer.
  num_quantizers_with_qnoise_factor = 0
  for quantizer in gradual_qnoise_callback_0.quantizers:
    if hasattr(quantizer, "qnoise_factor"):
      num_quantizers_with_qnoise_factor += 1
  assert_equal(num_quantizers_with_qnoise_factor, 3)  # Test "step"

  qnoise_factor = [
      np.array(d.qnoise_factor) for d in gradual_qnoise_callback_0.quantizers
  ]
  assert_equal(qnoise_factor, np.ones_like(qnoise_factor))


  # The number of batch does not pass the finish of 10. Exponent 3.0
  gradual_qnoise_callback_1 = QNoiseScheduler(
      start=2, finish=10, freq_type="step", exponent=3.0)

  model.fit(
      x_train,
      y_train,
      batch_size=1,
      epochs=1,
      verbose=0,
      callbacks=[
          gradual_qnoise_callback_1,
      ],
  )
  qnoise_factor = [
      np.array(d.qnoise_factor) for d in gradual_qnoise_callback_1.quantizers
  ]
  val = 1 - np.power((10.0 - 4.0) / (10.0 - 2.0), 3)
  assert_equal(qnoise_factor, np.full_like(qnoise_factor, val))

  # The number of batch does not pass the finish of 10. Exponent 2.0
  gradual_qnoise_callback_2 = QNoiseScheduler(
      start=2, finish=10, freq_type="step", exponent=2.0)

  model.fit(
      x_train,
      y_train,
      batch_size=1,
      epochs=1,
      verbose=0,
      callbacks=[
          gradual_qnoise_callback_2,
      ],
  )
  qnoise_factor = [
      np.array(d.qnoise_factor) for d in gradual_qnoise_callback_2.quantizers
  ]
  val = 1 - np.power((10.0 - 4.0) / (10.0 - 2.0), 2)
  assert_equal(qnoise_factor, np.full_like(qnoise_factor, val))

  # The number of batch does not pass the start of 6.
  gradual_qnoise_callback_3 = QNoiseScheduler(
      start=6, finish=10, freq_type="step", exponent=3.0)

  model.fit(
      x_train,
      y_train,
      batch_size=1,
      epochs=1,
      verbose=0,
      callbacks=[
          gradual_qnoise_callback_3,
      ],
  )
  qnoise_factor = [
      np.array(d.qnoise_factor) for d in gradual_qnoise_callback_3.quantizers
  ]
  assert_equal(qnoise_factor, np.zeros_like(qnoise_factor))


  # The number of training iterations passes the number of batches of an epoch.
  gradual_qnoise_callback_4 = QNoiseScheduler(
      start=6, finish=20, freq_type="step", exponent=3.0)
  epochs = 2
  model.fit(
      x_train,
      y_train,
      batch_size=1,
      epochs=epochs,
      verbose=0,
      callbacks=[
          gradual_qnoise_callback_4,
      ],
  )
  qnoise_factor = [
      np.array(d.qnoise_factor) for d in gradual_qnoise_callback_4.quantizers
  ]
  val = 1 - np.power((20.0 - (epochs*num_data - 1)) / (20.0 - 6.0), 3)
  assert_equal(qnoise_factor, np.full_like(qnoise_factor, val))

  # The number of training iterations passes the number of batches of an epoch
  # with update_freq = 2.
  gradual_qnoise_callback_5 = QNoiseScheduler(
      start=0,
      finish=20,
      freq_type="step",
      update_freq=2,
      exponent=3.0)
  epochs = 2
  model.fit(
      x_train,
      y_train,
      batch_size=1,
      epochs=epochs,
      verbose=0,
      callbacks=[
          gradual_qnoise_callback_5,
      ],
  )
  qnoise_factor = [
      np.array(d.qnoise_factor) for d in gradual_qnoise_callback_5.quantizers
  ]
  # It updates when the number of training iterations modulo update_freq is 0.
  val = 1 - np.power(
      (20.0 - epochs * ((epochs * num_data - 1) // epochs)) / (20.0 - 0.0), 3)
  assert_equal(qnoise_factor, np.full_like(qnoise_factor, val))


  ##########################
  # Test "epoch" freq_type #
  ##########################
  # The number of epoch does not pass the finish of 5.
  gradual_qnoise_callback_6 = QNoiseScheduler(
      start=1, finish=5, freq_type="epoch", exponent=3.0)

  model.fit(
      x_train,
      y_train,
      batch_size=1,
      epochs=3,
      verbose=0,
      callbacks=[
          gradual_qnoise_callback_6,
      ],
  )
  qnoise_factor = [
      np.array(d.qnoise_factor) for d in gradual_qnoise_callback_6.quantizers
  ]
  val = 1 - np.power((5.0 - 2.0) / (5.0 - 1.0), 3)
  assert_equal(qnoise_factor, np.full_like(qnoise_factor, val))
  assert_equal(len(gradual_qnoise_callback_6.quantizers), 3)  # Test "epoch"


if __name__ == "__main__":
  pytest.main([__file__])
