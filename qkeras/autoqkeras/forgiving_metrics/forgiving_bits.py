# Lint as: python3
# ==============================================================================
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
"""Implements forgiving factor metrics bit model size in bits."""

import numpy as np
import six
from qkeras.autoqkeras.forgiving_metrics.forgiving_factor import ForgivingFactor   # pylint: disable=line-too-long
from qkeras import get_quantizer


class ForgivingFactorBits(ForgivingFactor):
  """Implements forgiving factor with target as number of bits."""

  def __init__(
      self, delta_p, delta_n, rate, stress=1.0,
      input_bits=8, output_bits=8, ref_bits=8, config=None):
    self.stress = stress
    self.input_bits = input_bits
    self.output_bits = output_bits
    self.ref_bits = ref_bits
    self.ref_size = {}
    self.config = config if config else {}

    super(ForgivingFactorBits, self).__init__(delta_p, delta_n, rate)

  def _param_size(self, layer):
    """Computes size of parameters of a layer in bits."""
    t_size = self.ref_bits
    parameter_size = 0
    # we only compute parameter sizes for these layers, and BatchNormalization
    # is a special case because it exports mean and beta that is absorbed by
    # previous or next layer. As mean and beta will be compressed into a single
    # value, we actually only need to take care of the shape.
    if layer.__class__.__name__ in [
        "Dense", "Conv2D", "Conv1D", "DepthwiseConv2D"]:
      for w in layer.get_weights():
        parameter_size += t_size * np.prod(w.shape)
    elif layer.__class__.__name__ in [
        "QDense", "QConv2D", "QConv1D", "QDepthwiseConv2D"]:
      for i, w in enumerate(layer.get_weights()):
        if layer.get_quantizers()[i]:
          bits = layer.get_quantizers()[i].bits
        else:
          bits = t_size
        parameter_size += bits * np.prod(w.shape)
    elif layer.__class__.__name__ in ["BatchNormalization"]:
      # scale
      index = -1
      parameter_size += t_size * np.prod(layer.get_weights()[index].shape)
      # center (bias)
      if layer.center:
        index = int(bool(layer.scale))
        parameter_size += t_size * np.prod(layer.get_weights()[index].shape)
    elif layer.__class__.__name__ in ["QBatchNormalization"]:
      # scale
      index = -1
      bits = 6
      parameter_size += bits * np.prod(layer.get_weights()[index].shape)
      # center (bias)
      if layer.center:
        bits = 5
        index = int(bool(layer.scale))
        parameter_size += bits * np.prod(layer.get_weights()[index].shape)
    return parameter_size

  def _act_size(self, layer):
    """Computes size of activations of a layer in bits."""
    i_size = self.input_bits
    o_size = self.output_bits
    t_size = self.ref_bits
    output_size = np.prod(layer.output.shape[1:])
    # we compute activation sizes for inputs and outputs
    if layer.__class__.__name__ in ["InputLayer"]:
      return i_size * output_size
    elif layer.__class__.__name__ in [
        "Dense", "Conv2D", "Conv1D", "DepthwiseConv2D"]:
      if layer.activation is not None and layer.activation.__name__ != "linear":
        return t_size * output_size
      else:
        return 0
    elif layer.__class__.__name__ in [
        "QDense", "QConv2D", "QConv1D", "QDepthwiseConv2D"]:
      if layer.activation is None:
        is_softmax = False
        is_linear = False
      else:
        if isinstance(layer.activation, six.string_types):
          is_softmax = layer.activation == "softmax"
          is_linear = layer.activation == "linear"
        elif hasattr(layer.activation, "__name__"):
          is_softmax = layer.activation.__name__ == "softmax"
          is_linear = layer.activation.__name__ == "linear"
        else:
          is_softmax = False
          is_linear = False

        if is_softmax:
          bits = o_size
        elif is_linear:
          bits = 0
        else:
          assert not isinstance(layer.activation, six.string_types)
          if hasattr(layer.activation, "bits"):
            bits = layer.activation.bits
          else:
            bits = t_size

        return bits * np.prod(layer.output.shape.as_list()[1:])
    elif layer.__class__.__name__ in ["QActivation", "Activation"]:
      if isinstance(layer.activation, six.string_types):
        is_linear = layer.activation == "linear"
        is_softmax = layer.activation == "softmax"
        is_sigmoid = layer.activation == "sigmoid"
      else:
        is_linear = layer.activation.__name__ == "linear"
        is_softmax = layer.activation.__name__ == "softmax"
        is_sigmoid = layer.activation.__name__ == "sigmoid"

      if is_linear:
        bits = 0
      elif is_softmax or is_sigmoid:
        bits = o_size
      else:
        if isinstance(layer.activation, six.string_types):
          activation = get_quantizer(layer.activation)
        else:
          activation = layer.activation
        if hasattr(activation, "bits"):
          bits = activation.bits
        else:
          bits = t_size
      return bits * output_size
    return 0

  def compute_model_size(self, model):
    """Computes size of model."""

    a_size = 0
    p_size = 0
    total_size = 0
    model_size_dict = {}
    for layer in model.layers:
      layer_name = layer.__class__.__name__
      layer_config = self.config.get(
          layer_name, self.config.get("default", None))
      if layer_config:
        parameters = self._param_size(layer)
        activations = self._act_size(layer)
        p_weight = ("parameters" in layer_config)
        a_weight = ("activations" in layer_config)
        total = p_weight * parameters + a_weight * activations
        model_size_dict[layer.name] = {
            "parameters": parameters,
            "activations": activations,
            "total": total
        }
        a_size += a_weight * activations
        p_size += p_weight * parameters
        total_size += total

    return (total_size, p_size, a_size, model_size_dict)

  def get_reference(self, model):
    if not hasattr(self, "reference_size"):
      cached_result = self.compute_model_size(model)
      self.reference_size = cached_result[0] * self.stress
      self.ref_p = cached_result[1]
      self.ref_a = cached_result[2]
      self.reference_size_dict = cached_result[3]

    return self.reference_size

  def get_reference_stats(self):
    return self.reference_size_dict

  def get_trial(self, model):
    """Computes size of quantization trial."""

    result = self.compute_model_size(model)
    self.trial_size = result[0]
    self.total_p_bits = result[1]
    self.total_a_bits = result[2]
    self.trial_size_dict = result[3]

    return self.trial_size

  def get_total_factor(self):
    """we adjust the learning rate by size reduction."""
    ref_total = self.ref_a + self.ref_p
    trial_total = self.total_a_bits + self.total_p_bits
    return (trial_total - ref_total) / ref_total

  def print_stats(self):
    """Prints statistics of current model."""
    str_format = (
        "stats: delta_p={} delta_n={} rate={} trial_size={} reference_size={}\n"
        "       delta={:.2f}%"
    )

    print(
        str_format.format(
            self.delta_p, self.delta_n, self.rate, self.trial_size,
            int(self.reference_size), 100*self.delta())
    )

    a_percentage = np.round(
        100.0 * (self.total_a_bits - self.ref_a) / self.ref_a, 2)
    p_percentage = np.round(
        100.0 * (self.total_p_bits - self.ref_p) / self.ref_p, 2)
    ref_total = self.ref_a + self.ref_p
    trial_total = self.total_a_bits + self.total_p_bits
    total_percentage = np.round(
        100.0 * (trial_total - ref_total) / ref_total, 2)

    print(
        (
            "       a_bits={}/{} ({:.2f}%) p_bits={}/{} ({:.2f}%)\n"
            "       total={}/{} ({:.2f}%)"
        ).format(
            int(self.total_a_bits), int(self.ref_a), a_percentage,
            int(self.total_p_bits), int(self.ref_p), p_percentage,
            int(trial_total), int(ref_total), total_percentage
        ))
