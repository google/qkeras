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
"""Test the QAdaptiveActivation layer from qlayers.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
import pytest
import tensorflow.compat.v2 as tf

from qkeras.qlayers import QAdaptiveActivation
from qkeras.quantizers import _get_integer_bits


def run_qadaptiveactivation_test(input_val, kwargs):
  """Helper function to test QAdaptiveActivation inputs and outputs."""
  err = 'Failed test with {} on input {}'.format(kwargs, input_val)

  # Only test inputs of shape (batch_size, width, height, channels)
  assert len(input_val.shape) == 4, err

  # Only test short term layer usage with ema_decay == 0
  assert kwargs['ema_decay'] == 0, err
  assert kwargs['ema_freeze_delay'] is None, err

  # Prepare layer in a static TF graph
  model = tf.keras.Sequential([QAdaptiveActivation(**kwargs)])
  model.compile()

  # Test input on untrained EMAs
  qout = model(input_val, training=False).numpy()
  assert np.isclose(model.layers[0].quantizer(input_val), qout).all(), err
  assert np.isclose(model.layers[0].ema_min.numpy().flatten(), 0).all(), err
  assert np.isclose(model.layers[0].ema_max.numpy().flatten(), 0).all(), err

  # Run an unquantized input and train the EMA
  unquantized_out = model(input_val, training=True).numpy()
  assert kwargs['current_step'].numpy() == 0, err
  if kwargs['activation'] == 'quantized_relu':
    assert np.isclose(unquantized_out, np.maximum(input_val, 0)).all(), err
  elif kwargs['activation'] == 'quantized_bits':
    assert np.isclose(unquantized_out, input_val).all(), err
  else:
    raise ValueError('Invalid quantizer type ', kwargs['activation'])

  # Check EMAs
  if kwargs['per_channel']:
    assert np.isclose(model.layers[0].ema_min.numpy(),
                      np.min(input_val, axis=(0, 1, 2))).all(), err
    assert np.isclose(model.layers[0].ema_max.numpy(),
                      np.max(input_val, axis=(0, 1, 2))).all(), err
  else:
    assert np.isclose(model.layers[0].ema_min.numpy(),
                      np.min(input_val, axis=(0, 1, 2, 3))).all(), err
    assert np.isclose(model.layers[0].ema_max.numpy(),
                      np.max(input_val, axis=(0, 1, 2, 3))).all(), err

  # Check quantizer
  quant = model.layers[0].quantizer
  assert quant.__class__.__name__ == kwargs['activation'], err
  assert quant.bits == kwargs['total_bits'], err
  assert quant.symmetric == kwargs['symmetric'], err
  keep_negative = None
  if kwargs['activation'] == 'quantized_relu':
    assert not quant.is_quantized_clip, err
    assert quant.negative_slope == kwargs['relu_neg_slope'], err
    assert quant.relu_upper_bound is None, err
    keep_negative = kwargs['relu_neg_slope'] != 0
  elif kwargs['activation'] == 'quantized_bits':
    assert quant.keep_negative, err
    assert quant.alpha == 1.0, err
    keep_negative = True
  expected_integer_bits = _get_integer_bits(model.layers[0].ema_min.numpy(),
                                            model.layers[0].ema_max.numpy(),
                                            kwargs['total_bits'],
                                            kwargs['symmetric'],
                                            keep_negative,
                                            kwargs['po2_rounding']).numpy()
  assert np.isclose(expected_integer_bits, quant.integer.numpy()).all(), err

  # Skip to a step where the quantization is used
  kwargs['current_step'].assign(tf.constant(kwargs['quantization_delay'],
                                            tf.int64))

  # Check quantized output
  # To set qnoise_factor to 1.0 explicitly.
  qnoise_factor = np.array(quant.qnoise_factor)
  quant.update_qnoise_factor(1.0)
  expected_qout = np.copy(quant(input_val))
  # Revert qnoise_factor to its original value.
  quant.update_qnoise_factor(qnoise_factor)
  qout = model(input_val, training=True).numpy()
  assert np.isclose(expected_qout, qout).all(), err

  # Check testing mode
  qout = model(input_val, training=False).numpy()
  assert np.isclose(quant(input_val), qout).all(), err


@pytest.mark.parametrize(
    'momentum, ema_freeze_delay, total_steps, estimate_step_count',
    [(0.9, 50, 100, False), (0.5, 1000, 1500, False), (0.1, 2, 100, False),
     (0.999, 98, 100, False), (0.9, 50, 100, True), (0.5, 1000, 1500, True),
     (0.1, 2, 100, True), (0.999, 98, 100, True)])
def test_qadaptiveact_ema(momentum, ema_freeze_delay, total_steps,
                          estimate_step_count):
  """Test the exponential moving averages over time for QAdaptiveActivation."""

  # Initialize a QAdaptiveActivation layer just for testing the EMA
  if estimate_step_count:
    step = None
  else:
    step = tf.Variable(0, dtype=tf.int64)
  q_act = QAdaptiveActivation(activation='quantized_bits',
                              total_bits=8,
                              current_step=step,
                              quantization_delay=total_steps*2,
                              ema_freeze_delay=ema_freeze_delay,
                              ema_decay=momentum,
                              per_channel=True,
                              po2_rounding=False)
  model = tf.keras.Sequential([q_act])
  model.compile()

  # Simulate a number of training steps and check the EMA values
  exp_ema_max = 0.0
  exp_ema_min = 0.0
  for i in range(0, total_steps):
    vals = np.random.random((1, 2, 1)) * i  # generate random values for update
    model(vals, training=True)  # Simulate training

    # Check the steps match
    if estimate_step_count:
      assert np.equal(q_act.step.numpy(), i)

    # Calculate expected values
    if i <= ema_freeze_delay:
      exp_ema_max = (exp_ema_max * momentum) + (vals.max() * (1.0 - momentum))
      exp_ema_min = (exp_ema_min * momentum) + (vals.min() * (1.0 - momentum))
    exp_int_bits = _get_integer_bits(exp_ema_min, exp_ema_max,
                                     q_act.quantizer.bits,
                                     q_act.quantizer.symmetric,
                                     q_act.quantizer.symmetric, False)

    # Check results
    assert np.abs(exp_ema_max - q_act.ema_max.numpy()[0]) < 0.0001

    assert np.isclose(exp_int_bits.numpy(), q_act.quantizer.integer.numpy())
    if not estimate_step_count:
      step.assign_add(1)


def test_qadaptiveactivation():
  """Test a wide variety of inputs to the QAdaptiveActivation layer."""
  test_options = {
      'activation': ['quantized_bits', 'quantized_relu'],
      'total_bits': [1, 2, 4, 8, 16],
      'symmetric': [True, False],
      'quantization_delay': [1],  # We will only run for one step
      'per_channel': [True, False],
      'po2_rounding': [True, False],
      'relu_neg_slope': [0.0, -0.5]
  }

  for args in itertools.product(*test_options.values()):
    args = {list(test_options.keys())[i]: args[i] for i in range(len(args))}
    args['ema_freeze_delay'] = None  # This test does not test the EMA freeze
    args['ema_decay'] = 0 # This test not test the EMA delay
    for img_shape in [(1, 28, 28, 3), (1, 3, 4, 5)]:
      for input_scale in [255, 1]:
        args['current_step'] = tf.Variable(0, dtype=tf.int64)
        img = np.random.random(img_shape) * input_scale
        run_qadaptiveactivation_test(img, args)

if __name__ == '__main__':
  pytest.main([__file__])
