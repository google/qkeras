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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import os
import tensorflow as tf


class QNoiseScheduler(tf.keras.callbacks.Callback):
  """Schedules the gradual quantization noise training for each step (or epoch).

     It updates the qnoise_factor in the quantizers to gradually introduce the
     quantization noise during training.

     The idea was adopted from "https://arxiv.org/pdf/1903.01061.pdf"
  """

  def __init__(self,
               start,
               finish,
               freq_type="epoch",
               update_freq=1,
               initial_step_or_epoch=0,
               exponent=3.0,
               use_ste=True,
               log_dir=None):
    """Initializes this QNoiseScheduler.

    Args:
      start: Int. The step (epoch) to start the gradual training.
      finish: Int. The step (epoch) to finish the gradual training. When the
        start and the finish are equal, the qnoise_factor will be 1.0 in the
        beginning of the training.
      freq_type: Str. "step" or "epoch". It sets the qnoise_factor update
        frequency type.
      update_freq: Int. Updating frequency of the qnoise_factor.
      initial_step_or_epoch:  Int. Step or epoch at which to start training.
      exponent: Float. It is the exponent in the qnoise_factor calculation. It
        controls the rate of the gradual qnoise_factor change.
      use_ste: Bool. Whether to use "straight-through estimator" (STE) method or
        not.
      log_dir: Str. log directory to save qnoise_factor every epoch end.
    """
    super(QNoiseScheduler, self).__init__()

    self.start = start
    self.finish = finish
    if start > finish:
      raise ValueError(
          ("start {} must be greater than finish {}").format(start, finish))
    supported_freq_type = ["step", "epoch"]
    if freq_type not in supported_freq_type:
      raise ValueError(("Invalid frequency type {}. only {} are "
                        "supported.").format(freq_type, supported_freq_type))
    self.freq_type = freq_type
    self.update_freq = update_freq
    self.initial_step_or_epoch = initial_step_or_epoch
    self.exponent = exponent
    self.qnoise_factor = None
    self.use_ste = use_ste
    self.quantizers = None
    self.summary_writer = None
    if log_dir:
      self.summary_writer = tf.summary.create_file_writer(log_dir)
    self.num_iters = np.array(0, dtype="int64")

  def calculate_qnoise_factor(self, freq):
    """Returns calculated qnoise_factor based on the current step (epoch) and
    the schedule parameters.

    Args:
      freq: The current step (or epoch) to calculate the qnoise_factor.

    Returns:
      qnoise_factor : calculated qnoise_factor.
    """
    if freq < self.start:
      qnoise_factor = 0.0
    elif freq <= self.finish and self.start != self.finish:
      val = float(self.finish - freq) / float(self.finish - self.start)
      qnoise_factor = 1.0 - np.power(val, self.exponent)
    else:
      qnoise_factor = 1.0

    return qnoise_factor

  def set_qnoise_factor(self, quantizer, qnoise_factor):
    """Set self.qnoise_factor and update the qnoise_factor of the quantizer."""

    # Updating the qnoise_factor of the quantizer.
    quantizer.update_qnoise_factor(qnoise_factor)
    # Updating the qnoise_factor of the callback.
    self.qnoise_factor = qnoise_factor

  def set_quantizers(self):
    """Set quantizers to update the qnoise_factor.

    This must be called before building the quantizers.
    """
    for quantizer in self.quantizers:
      if hasattr(quantizer, "use_ste"):
        quantizer.use_ste = self.use_ste
      if hasattr(quantizer, "use_variables"):
        quantizer.use_variables = True
      if hasattr(quantizer, "built"):
        # If the quantizer has been built but not using tf.Variable then it
        # builds again to create tf.Variables.
        if quantizer.built and not isinstance(quantizer.qnoise_factor,
                                                 tf.Variable):
          quantizer.build(use_variables=True)

      # Set the qnoise_factor to 0.0 to pretrain without quantization.
      self.set_qnoise_factor(quantizer, qnoise_factor=0.0)

  def get_quantizers(self, model):
    """Returns a list of quantizers with qnoise_factor in the model.

    Args:
      model: model to get a list of quantizers with qnoise_factor.

    Returns:
      A list of quantizers with the qnoise_factor variable.
    """
    all_quantizers = []
    for layer in model.layers:
      # A list of attributes holding the quantizer(s).
      for attr in ["quantizers", "quantizer"]:
        if hasattr(layer, attr):
          quantizers = getattr(layer, attr)
          quantizers = quantizers if attr == "quantizers" else [quantizers]
          for quantizer in quantizers:
            if hasattr(quantizer, "qnoise_factor"):
              all_quantizers.append(quantizer)

    return all_quantizers

  def update_qnoise_factor(self, freq):
    """Update the qnoise_factor of the model.

    Args:
      freq: The current step (epoch) to calculate the qnoise_factor.
    """
    # Update the qnoise_factor at the frequency of self.update_freq.
    if freq % self.update_freq != 0:
      self.num_iters += 1
      return

    new_qnoise_factor = self.calculate_qnoise_factor(freq)
    for quantizer in self.quantizers:
      # Updates the qnoise factors of the quantizers in the model.
      self.set_qnoise_factor(quantizer, new_qnoise_factor)
    self.num_iters += 1

  def on_train_begin(self, logs=None):
    if not self.quantizers:
      # Build a list of quantizers which is used for updating qnoise_factor.
      self.quantizers = self.get_quantizers(self.model)
      self.set_quantizers()

  def on_epoch_begin(self, epoch, logs=None):
    if self.freq_type == "epoch":
      self.update_qnoise_factor(self.initial_step_or_epoch + self.num_iters)

  def on_epoch_end(self, epoch, logs=None):
    if self.summary_writer:
      with self.summary_writer.as_default():
        tf.summary.scalar("qnoise_factor", data=self.qnoise_factor, step=epoch)

  def on_train_batch_begin(self, batch, logs=None):
    if self.freq_type == "step":
      self.update_qnoise_factor(self.initial_step_or_epoch + self.num_iters)
