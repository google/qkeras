# Copyright 2025 Google LLC
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
import tensorflow.compat.v2 as tf
import tensorflow.keras.backend as K


def _create_variable_name(attr_name, var_name=None):
  """Creates variable name.

  Arguments:
    attr_name: string. attribute name
    var_name: string. variable name

  Returns:
    string. variable name
  """

  if var_name:
    return var_name + "/" + attr_name

  # This naming scheme is to solve a problem of a layer having more than
  # one quantizer can have multiple qnoise_factor variables with the same
  # name of "qnoise_factor".
  return attr_name + "_" + str(K.get_uid(attr_name))


class BaseQuantizer(tf.Module):
  """Base quantizer.

  Defines behavior all quantizers should follow.
  """

  def __init__(self):
    self.built = False

  def build(self, var_name=None, use_variables=False):
    if use_variables:
      if hasattr(self, "qnoise_factor"):
        self.qnoise_factor = tf.Variable(
            lambda: tf.constant(self.qnoise_factor, dtype=tf.float32),
            name=_create_variable_name("qnoise_factor", var_name=var_name),
            dtype=tf.float32,
            trainable=False,
        )
    self.built = True

  def _set_trainable_parameter(self):
    pass

  def update_qnoise_factor(self, qnoise_factor):
    """Update qnoise_factor."""
    if isinstance(self.qnoise_factor, tf.Variable):
      # self.qnoise_factor is a tf.Variable.
      # This is to update self.qnoise_factor during training.
      self.qnoise_factor.assign(qnoise_factor)
    else:
      if isinstance(qnoise_factor, tf.Variable):
        # self.qnoise_factor is a numpy variable, and qnoise_factor is a
        # tf.Variable.
        self.qnoise_factor = qnoise_factor.eval()
      else:
        # self.qnoise_factor and qnoise_factor are numpy variables.
        # This is to set self.qnoise_factor before building
        # (creating tf.Variable) it.
        self.qnoise_factor = qnoise_factor

  # Override not to expose the quantizer variables.
  @property
  def variables(self):
    return ()

  # Override not to expose the quantizer variables.
  @property
  def trainable_variables(self):
    return ()

  # Override not to expose the quantizer variables.
  @property
  def non_trainable_variables(self):
    return ()
