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
"""Implements support for auto-quantization."""

import collections
import json
import os
import re
import copy
from absl import logging
import keras_tuner as kt
from keras_tuner import HyperModel
from keras_tuner import BayesianOptimization
from keras_tuner import Hyperband
from keras_tuner import RandomSearch
import numpy as np
import six
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.metrics import sparse_categorical_accuracy
from qkeras.autoqkeras.forgiving_metrics import forgiving_factor  # pylint: disable=line-too-long
from qkeras.autoqkeras.forgiving_metrics import ForgivingFactor  # pylint: disable=line-too-long
from qkeras.autoqkeras.quantization_config import default_quantization_config  # pylint: disable=line-too-long
from qkeras.autoqkeras.utils import print_qmodel_summary
from qkeras.utils import clone_model
from qkeras.utils import model_quantize


# AutoQKHyperModel is implemented on top of keras_tuner
# It basically creates a quantized model based on some rules
# and it computes a acc_delta that boosts the accuracy when
# choosing smaller models.

# Boosting function behaves like this.
# We use the following formula to compute the decrease factor:
#   reference_size: number of parameters + activations of the model,
#     assuming an 8-bit implementation.
#   trial_size: number of parameters + activations of trial.
#
#   1) First, we compute how many times we decresed/increased the model
#     i = log(reference_size / trial_size) / log(rate)
#
#   2) Then, we use delta_p / delta_n if model is smaller/bigger
#      than reference model.
#
#      delta = i * (
#          (i < 0) * delta_n + (i >= 0) * delta_p
#      )
#
#   3) the accuracy of the model (score) is adjusted by acc * delta
#
#   The delta "boosts" the accuracy to allow worse model to be
#   chosen by hypermodel tuner.
#

REGISTERED_LAYERS = ["Dense", "Conv1D", "Conv2D", "DepthwiseConv2D",
                     "SimpleRNN", "LSTM", "GRU", "Bidirectional",
                     "Conv2DTranspose", "SeparableConv1D", "SeparableConv2D"]

Q_LAYERS = list(map(lambda x : 'Q' + x, REGISTERED_LAYERS))

SEQUENCE_LAYERS = ["SimpleRNN", "LSTM", "GRU", "Bidirectional"]

class AutoQKHyperModel(HyperModel):
  """Creates an hypermodel to attempt to quantize a reference model.

     Arguments:
       model: Model to be quantized.
       metrics: List of metrics to be used.
       custom_objects: Custom objects used by Keras during quantization.
       target: Secondary metric to chase during search ("bits" or "energy").
       transfer_weights: if true, transfer weights from unquantized model.
       frozen_layers: if true, these layers will not be quantized but
         weights transferred from original model.
       activation_bits: parameter to be used by 'model_quantize'.
       limit: limit the number of bits in quantizers, specified as dictionary.
       tune_filters: one of "block", "layer", "none" for tuning entire
         network, each layer separately, or no tuning.
       tune_filters_exceptions: name of layers that will not be tuned.
       layer_indexes: we only quantize layers whose ids are in layer_indexes.
       learning_rate_optimizer: if true, we optimize learning rate along with
         other parameters.
       head_name: specify which head to calcuate score/trial-size from in
         autoqkeras
       quantization_config: dictionary containing configuration of
         quantizers for kernel, bias and activation.
       extend_model_metrics: If to append the trial size and score metrics to
         model metrics, which are used for AutoQKeras to determine the quality
         of a model.

     Returns:
       quantized model in trial and boosted accuracy function compiled
       into quantized model.
  """

  def __init__(
      self, model, metrics, custom_objects=None, target=None,
      transfer_weights=False, frozen_layers=None, activation_bits=4, limit=None,
      tune_filters="none", tune_filters_exceptions=None,
      layer_indexes=None, learning_rate_optimizer=False,
      head_name=None, quantization_config=None, extend_model_metrics=True,
  ):
    self.model = model
    self.metrics = metrics
    self.custom_objects = custom_objects if custom_objects else {}

    self.target = target

    self.reference_size = self.target.get_reference(model)

    self.transfer_weights = transfer_weights
    self.frozen_layers = frozen_layers if frozen_layers else []
    self.activation_bits = activation_bits
    self.head_name = head_name
    self.extend_model_metrics = extend_model_metrics
    # make sure we have at least 3 elements in list
    # first one for kernel, second one for bias and thid one for activations.
    #
    # limit is in the format, where default replaces missing values:
    # '{
    #      "Conv2D":[weight,bias,activation],
    #      "RNN":[weight,bias,recurrent,activation],
    #      "Dense":[weight,bias,activation],
    #      "Activation":[activation]
    #      "default": value
    #  }'

    if limit is None:
      self.limit = {}
    else:
      self.limit = limit

    self.groups = {}

    assert isinstance(self.limit, dict)

    if self.limit.get("default", None) is None:
      default = 8
    else:
      default = self.limit["default"]

    # make sure we have entries for every type of layer we process
    self._adjust_limit(default)

    print("Limit configuration:" + json.dumps(self.limit))

    assert tune_filters in ["block", "layer", "none"]

    self.tune_filters = tune_filters
    self.tune_filters_exceptions = re.compile(tune_filters_exceptions)

    self.layer_indexes = layer_indexes
    self.learning_rate_optimizer = learning_rate_optimizer

    # load quantizer types for each type of quantizer
    if quantization_config is None:
      self.quantization_config = default_quantization_config
    else:
      self.quantization_config = quantization_config

  def _adjust_limit(self, default):
    """Makes sure limit has all the fields required."""
    if isinstance(default, list):
      assert 3 <= len(default) <= 4 
    else:
      default = [default] * 3

    # we consider that if name is not there, we will ignore the layer
    for name in REGISTERED_LAYERS:
      if name in self.limit:
        length = len(self.limit[name])
        if length < 4 and name in SEQUENCE_LAYERS:
          assert len(default) == 4
          self.limit[name] = self.limit[name] + default[length:]
        elif length < 3:
          # No recurrent limit needed for non recurrent layers
          self.limit[name] = self.limit[name] + default[length:2] + default[-1:]

  def _n(self, name, s_list):
    """Creates a unique name for the tuner."""
    return name + "_".join([str(v) for v in s_list])

  def _get_quantizer(self, hp, head, layer_name, layer_class_name,
                     i_list=None, is_kernel=True, is_linear=False):
    """Gets a quantizer randomly for kernels/bias/activations."""

    # first pick up which group we belong to.

    if not i_list:
      i_list = []

    if is_linear:
      # linear quantizers
      field_name = "linear"
      kq = self.quantization_config["linear"]
      index = 0
      q_list = list(kq.keys())
      q_dict = kq
    elif "kernel" in head:
      # kernel quantizers
      field_name = "kernel"
      kq = self.quantization_config["kernel"]
      index = 0
      q_list = list(kq.keys())
      q_dict = kq
    elif "bias" in head:
      # bias quantizers
      field_name = "bias"
      bq = self.quantization_config["bias"]
      index = 1
      q_list = list(bq.keys())
      q_dict = bq
    elif "pointwise_kernel" in head: # limit is same as kernel
      # pointwise kernel quantizers
      field_name = "pointwise_kernel"
      kq = self.quantization_config["pointwise_kernel"]
      index = 2
      q_list = list(kq.keys())
      q_dict = kq
    elif "recurrent_kernel" in head: # limit is same as kernel
      # recurrent kernel quantizers
      field_name = "recurrent_kernel"
      kq = self.quantization_config["recurrent_kernel"]
      index = 2
      q_list = list(kq.keys())
      q_dict = kq
    elif "recurrent_activation" in head: # limit is same as kernel
      # recurrent activation quantizers
      field_name = "recurrent_activation"
      raq = self.quantization_config["recurrent_activation"]
      index = -1
      q_list = list(raq.keys())
      q_dict = raq
    else:
      # activation quantizers
      field_name = "activation"
      aq = self.quantization_config["activation"]
      index = -1
      q_list = list(aq.keys())
      q_dict = aq

    # we first we search for layer name. If it is not there, we switch to
    # layer class name.

    found_pattern = False
    name = layer_class_name
    count = -1
    for i, pattern in enumerate(self.limit):
      if re.match(pattern, layer_name):
        found_pattern = True
        name = pattern
        count = i
        break

    # for partially quantized networks we may not have
    # the layer class name in the set.

    if name == layer_class_name and name not in self.limit:
      return None, -1

    # groups is a dictionary that contains dictionary of the
    # patterns so that we can group everything together

    if found_pattern:
      if name in self.groups and index in self.groups[name]:
        return self.groups[name][index]

      # not there, let's use a different name for
      # the head and field
      head = "qk_group_" + str(count) + "_" + field_name
      head = name + "_" + field_name

    # limit group can be a list of quantizers or a
    # number that tells us maximum number of bits

    if isinstance(self.limit[name][index], list):
      # we assume this is a subset of the q_keys
      # entry in quantization_config will be like:
      #   "Conv2D": [ ["q1", "q2", "q3"], ... ]
      #
      # we always assume this list is a subset of
      # the original list or we will raise an
      # error.

      q_list = self.limit[name][index]
      q_dict = {
          key: q_dict[key] for key in q_list
      }
    else:
      q_dict = {
          key: value for (key, value) in q_dict.items()
          if value <= self.limit[name][index]
      }
      q_list = list(q_dict.keys())

    # didn't found a match in groups, create one.

    if len(q_list) == 1:
      q_name = hp.Fixed(self._n(head + "_quantizer", i_list), q_list[0])
    else:
      q_name = hp.Choice(self._n(head + "_quantizer", i_list), q_list)

    if found_pattern:
      if name not in self.groups:
        self.groups[name] = {index: (q_name, q_dict[q_name])}
      else:
        self.groups[name][index] = (q_name, q_dict[q_name])

    return (q_name, q_dict[q_name])

  def quantize_model(self, hp):
    """Quantize model by hyperparameter search and extracting size schema."""

    # configuration for quantization.
    q_dict = {}

    model = clone_model(self.model, self.custom_objects)

    fanin = []

    filter_range = [0.5, 0.75, 1.0, 1.5, 2.0]

    # network_filters=hp.Choice(...) should only be defined if we are sure
    # current blocks has any layer that need filter sweep.
    # Otherwise, when no layer needs filter sweep and a hp variable is defined,
    # there will be uneffective trials that loop around the network
    # filter range, even though none of the filter sweep was ever applied to
    # any layers. Therfore, we use filter_sweep_enabled to mark if any layer
    # in current block needs filter sweep.
    kernel_quantizer_dict = {}
    filter_sweep_enabled = False
    for layer in model.layers:
      if layer.__class__.__name__ in REGISTERED_LAYERS:
        kernel_quantizer, bits = self._get_quantizer(
            hp, layer.name + "_kernel", layer.name, layer.__class__.__name__,
            is_kernel=True)

        kernel_quantizer_dict[layer.name] = (kernel_quantizer, bits)

        # kernel_quantizer is not None ->  layer in the current block need
        # to be quantized
        if kernel_quantizer:
          if (
              not filter_sweep_enabled and self.tune_filters in
              ["layer", "block"]
              and not self.tune_filters_exceptions.search(layer.name) and
              layer.__class__.__name__ in
              ["Dense", "Conv1D", "Conv2D", "Conv2DTranspose"]
          ):
            filter_sweep_enabled = True

        if layer.__class__.__name__ in SEQUENCE_LAYERS:
          recurrent_quantizer, _ = self._get_quantizer(
            hp, layer.name + "_recurrent_kernel", layer.name, layer.__class__.__name__,
            is_kernel=True)

        if layer.__class__.__name__ in ["SeparableConv1D", "SeparableConv2D"]:
          pointwise_quantizer, _ = self._get_quantizer(
            hp, layer.name + "_pointwise_kernel", layer.name, layer.__class__.__name__,
            is_kernel=True)

    if self.tune_filters == "block" and filter_sweep_enabled:
      network_filters = hp.Choice(
          "network_filters",
          values=filter_range,
          default=1.0
      )
    else:
      network_filters = 1.0

    for layer_id, layer in enumerate(model.layers):

      # we can use these indexes to disable some layers, like the last
      # layer

      if self.layer_indexes is not None and layer_id not in self.layer_indexes:
        continue

      layer_d = {}

      if layer.__class__.__name__ in Q_LAYERS:
        weights = layer.get_weights()[0]
        if (
            layer.get_quantizers()[0] and
            hasattr(layer.get_quantizers()[0], "bits")
        ):
          bits = layer.get_quantizers()[0].bits
        else:
          bits = 8
        fanin.append(np.prod(weights.shape[:-1]) * (8. - bits) / 8.)
        
      if layer.__class__.__name__ in REGISTERED_LAYERS:
        # difference between depthwise and the rest is just the name
        # of the kernel.
        if layer.__class__.__name__ in [
            "DepthwiseConv2D", "SeparableConv1D", "SeparableConv2D"
        ]:
          kernel_name = "depthwise_quantizer"
        else:
          kernel_name = "kernel_quantizer"

        # sample kernel quantizer.
        (kernel_quantizer, bits) = kernel_quantizer_dict[layer.name]

        if not kernel_quantizer:
          continue

        # process fanin here

        if bits < 8:
          weights = layer.get_weights()[0]
          fanin.append(np.prod(weights.shape[:-1]) * (8. - bits) / 8.)

        # we only want to do that if we are going to quantize layer
        if (
            self.tune_filters in ["layer", "block"] and
            not self.tune_filters_exceptions.search(layer.name) and
            layer.__class__.__name__ in [
                "Dense", "Conv1D", "Conv2D", "Conv2DTranspose",
                "SeparableConv1D", "SeparableConv2D"
            ]
        ):
          if self.tune_filters == "layer":
            layer_filters = hp.Choice(
                "network_filters_" + layer.name,
                values=filter_range,
                default=1.0
            )
          else:
            layer_filters = network_filters

          if layer.__class__.__name__ == "Dense":
            layer.units = max(int(layer.units * layer_filters), 1)
          elif layer.__class__.__name__ in [
              "Conv1D", "Conv2D", "Conv2DTranspose",
              "SeparableConv1D", "SeparableConv2D"
          ]:
            layer.filters = max(int(layer.filters * layer_filters), 1)

        layer_d[kernel_name] = kernel_quantizer

        if layer.__class__.__name__ in SEQUENCE_LAYERS:
          layer_d['recurrent_quantizer'] = recurrent_quantizer

        if layer.__class__.__name__ in ["SeparableConv1D", "SeparableConv2D"]:
          layer_d['pointwise_quantizer'] = pointwise_quantizer

        if layer.__class__.__name__ in ["LSTM", "GRU", "Bidirectional"]:
          layer_d['recurrent_activation'], _  = self._get_quantizer(
              hp, layer.name + "_recurrent_activation", layer.name,
              layer.__class__.__name__, is_kernel=False)

        # if we use bias, sample quantizer.
        if layer.__class__.__name__ == "Bidirectional":
          layer_d["bias_quantizer"], bits = self._get_quantizer(
              hp, layer.name + "_bias", layer.name, layer.__class__.__name__,
              is_kernel=False)
          layer_d["activation"], bits = self._get_quantizer(
              hp, layer.name + "_activation", layer.name,
              layer.__class__.__name__, is_kernel=False)
          q_dict[layer.name] = layer_d 
        else:
          if layer.use_bias:
            layer_d["bias_quantizer"], bits = self._get_quantizer(
                hp, layer.name + "_bias", layer.name, layer.__class__.__name__,
                is_kernel=False)

          # if activation is not linear/softmax we need to process it.
          if layer.activation is None:
            is_softmax = False
            is_linear = False
          else:
            if isinstance(layer.activation, six.string_types):
              is_softmax = layer.activation == "softmax"
              is_linear = layer.activation == "linear"
            else:
              is_softmax = layer.activation.__name__ == "softmax"
              is_linear = layer.activation.__name__ == "linear"

          if not is_softmax and not is_linear:
            layer_d["activation"], bits = self._get_quantizer(
                hp, layer.name + "_activation", layer.name,
                layer.__class__.__name__, is_kernel=False)

          q_dict[layer.name] = layer_d

      elif layer.__class__.__name__ in ["Reshape"]:
        # we cannot handle fine tuning filters per layer right now.
        assert self.tune_filters in ["none", "block"]

        # we need to make sure this pattern exists, this should only occur for
        # "scheduler", so the name will be complete and not a pattern.

        if (
            self.tune_filters == "none" or
            layer.name not in self.limit or
            self.tune_filters_exceptions.search(layer.name)
        ):
          continue

        if K.image_data_format() == "channels_last":
          layer.target_shape = layer.target_shape[:-1] + (
              min(int(layer.target_shape[-1] * network_filters), 1),)
        else:
          layer.target_shape = (int(layer.target_shape[0] * network_filters),
                                ) + layer.target_shape[1:]

      elif layer.__class__.__name__ in ["Activation"]:
        if isinstance(layer.activation, six.string_types):
          is_linear = layer.activation == "linear"
          is_softmax = layer.activation == "softmax"
        else:
          is_linear = layer.activation.__name__ == "linear"
          is_softmax = layer.activation.__name__ == "softmax"

        # if it is a linear activation, we will notify the
        # quantizer we are searching for linear type of
        # quantizers

        if not is_softmax:
          activation, bits = self._get_quantizer(
              hp, layer.name + "_activation", layer.name,
              layer.__class__.__name__, is_kernel=False,
              is_linear=is_linear)

          if not activation:
            continue

          # look at documentation on model_quantize
          q_dict[layer.name] = activation
      elif layer.__class__.__name__ in self.limit:
        # mark it for conversion
        q_dict[layer.name] = {}
      else:
        for pattern in self.limit:
          if re.match(pattern, layer.name):
            q_dict[layer.name] = {}
            break

    q_model = model_quantize(
        model, q_dict, self.activation_bits,
        custom_objects=self.custom_objects,
        transfer_weights=self.transfer_weights)

    return q_model, fanin

  def build(self, hp):
    """Builds hyperparameterized quantized model."""

    self.groups = {}

    # we are not using the fanin right now.

    q_model, _ = self.quantize_model(hp)

    # transfer weights from previous run as we know we will not
    if self.learning_rate_optimizer:
      # if learning_rate_optimizer, we try to transfer weights from previous run
      print("... freezing layers {}.".format(", ".join(self.frozen_layers)))
      for layer_name in self.frozen_layers:
        o_weights = self.model.get_layer(layer_name).get_weights()
        layer = q_model.get_layer(layer_name)
        # don't know if setting trainable to False is good or not yet
        # try to do "soft-freeze" by transferring weights. More experiments
        # needed before we decide what to do.
        # layer.trainable = False
        weights = layer.get_weights()
        # because we can be changing number of layers, we do not know
        # if we can really use some of the weights or not.
        equal_layer = True
        for w in range(len(o_weights)):
          if o_weights[w].shape != weights[w].shape:
            equal_layer = False
            break
        if equal_layer:
          layer.set_weights(o_weights)

    self.trial_size = self.target.get_trial(q_model)

    # we will use a boosted accuracy computation

    delta = self.target.delta()

    # by default, we use the first metric specified by the
    # user to be the target metric.
    if not self.metrics:
      score_metric = None
    elif isinstance(self.metrics, dict):
      if not self.head_name:
      # if head_name not provided, find the first metric from the dict
        score_key = list(self.metrics.keys())[0]
      else:
        # find the metric assoicated with the head_name
        score_key = self.head_name
      score_metric = self.metrics[score_key]
      if isinstance(score_metric, list):
        score_metric = score_metric[0]
    elif isinstance(self.metrics, list):
      score_metric = self.metrics[0]

    self.score = AutoQKHyperModel.adjusted_score(
        self, delta, score_metric)

    # some papers suggest that we use learning_rate * sqrt(fanin) / layer
    # we cannot do that right now, but we can definitely do that
    # if we are quantizing one layer at a time
    #
    # https://arxiv.org/pdf/1511.00363.pdf

    # we use the magic number to smooth out the average
    total_factor = self.target.get_total_factor()
    delta_lr = 1.0 + (total_factor < 0) * total_factor

    # we assume model has been compiled at least.

    lr = float(self.model.optimizer.lr.numpy())

    # we assume that delta_lr can lower lr to accommodate
    # for more quantization
    #
    # if learning rate scheduler is used, we assume the callback to manage
    # learning rate. Just set it to constant.

    if self.learning_rate_optimizer:
      lr_range = list(lr * np.linspace(delta_lr, 1.1, 5))
      lr_choice = hp.Choice("learning_rate", lr_range)
      self.model.optimizer.learning_rate = lr_choice
    else:
      lr_choice = lr
      print("learning_rate: {}".format(lr))

    optimizer = self.model.optimizer

    q_model.summary()

    metrics = self.metrics

    # extend metrics by including score and trial_size metrics
    if self.extend_model_metrics:
      ext_metrics = copy.deepcopy(metrics)
      if isinstance(ext_metrics, dict):
        # for dict, add trial_size_metric and score metric to target output
        if not self.head_name:
          # if head_name not provided, find the first metric from the dict
          score_key = list(ext_metrics.keys())[0]
        else:
          # find the metric assoicated with the head_name
          score_key = self.head_name
        score_metric = ext_metrics[score_key]
        if isinstance(score_metric, list):
          score_metric += [self.trial_size_metric(self.trial_size), self.score]
        else:
          score_metric = [score_metric]
          score_metric += [self.trial_size_metric(self.trial_size), self.score]
        ext_metrics[score_key] = score_metric
      else:
        ext_metrics += [
            self.trial_size_metric(self.trial_size),
            self.score]
      metrics = ext_metrics

    q_model.compile(
        optimizer=optimizer,
        loss=self.model.loss,
        metrics=metrics
    )
    self.q_model = q_model

    # this just prints a summary of the quantization for debugging
    # purposes

    self.target.print_stats()
    print_qmodel_summary(q_model)

    return q_model

  @staticmethod
  def adjusted_score(hyper_model, delta, metric_function=None):
    def score(y_true, y_pred):
      y_t_rank = len(y_true.shape.as_list())
      y_p_rank = len(y_pred.shape.as_list())
      y_t_last_dim = y_true.shape.as_list()[-1]
      y_p_last_dim = y_pred.shape.as_list()[-1]

      is_binary = y_p_last_dim == 1
      is_sparse_categorical = (
          y_t_rank < y_p_rank or y_t_last_dim == 1 and y_p_last_dim > 1)

      if isinstance(metric_function, six.string_types):
        if metric_function in ["accuracy", "acc"]:
          if is_binary:
            metric = binary_accuracy(y_true, y_pred)
          elif is_sparse_categorical:
            metric = sparse_categorical_accuracy(y_true, y_pred)
          else:
            metric = categorical_accuracy(y_true, y_pred)
        else:
          metric = categorical_accuracy(y_true, y_pred)
      else:
        metric = metric_function(y_true, y_pred)

      return K.cast(metric * (1.0 + delta), K.floatx())

    if not metric_function:
      metric_function = "accuracy"

    return score

  @staticmethod
  def trial_size_metric(trial_size):
    def trial(y_true, y_pred):  # pylint: disable=unused-argument
      return K.cast(trial_size, K.floatx())
    return trial


class AutoQKeras:
  """Performs autoquantization in Keras model.

     Arguments:
       model: Model to be quantized.
       metrics: List of metrics to be used.
       custom_objects: Custom objects used by Keras during quantization.
       goal: Metric to compute secondary goal of search (bits or energy)
       output_dir: name of output directory to store results.
       mode: random, hyperband or bayesian used by keras_tuner.
       custom_tuner: The Keras Tuner class to use to search hyperparams
       transfer_weights: if true, transfer weights from unquantized model.
       frozen_layers: if true, these layers will not be quantized but
         weights transferred from original model.
       activation_bits: parameter to be used by 'model_quantize'.
       limit: limit the number of bits in quantizers specified as a dictionary.
       tune_filters: one of "block", "layer", "none" for tuning entire
         network, each layer separately, or no tuning.
       tune_filters_exceptions: name of layers that will not be tuned.
       layer_indexes: indexes of layers we will quantize.
       learning_rate_optimizer: if true, user will provide lr scheduler
         callback.
       quantization_config: file name of dictionary containing configuration of
         quantizers for kernel, bias and activation.
       head_name: specify which head to calcuate score/trial-size from in
         autoqkeras
       score_metric: Str. Optional metric name to use to evaluate the trials.
         Defaults to val_score
       tuner_kwargs: parameters for keras_tuner depending on whether
         mode is random, hyperband or baeysian. Please refer to the
         documentation of kerstuner Tuners.
  """

  def __init__(
      self, model, metrics=None, custom_objects=None, goal=None,
      output_dir="result", mode="random", custom_tuner=None,
      transfer_weights=False, frozen_layers=None, activation_bits=4,
      limit=None, tune_filters="none",
      tune_filters_exceptions=None, learning_rate_optimizer=False,
      layer_indexes=None, quantization_config=None, overwrite=True,
      head_name=None, score_metric=None, **tuner_kwargs):

    # Collect input arguments to AutoQKeras for usage by custom tuner
    autoqkeras_input_args = locals()

    if not metrics:
      metrics = []

    if not custom_objects:
      custom_objects = {}

    # goal: { "type": ["bits", "energy"], "params": {...} } or ForgivingFactor
    #   type
    # For type == "bits":
    #   delta_p: increment (in %) of the accuracy if trial is smaller.
    #   delta_n: decrement (in %) of the accuracy if trial is bigger.
    #   rate: rate of decrease/increase in model size in terms of bits.
    #   input_bits; size of input tensors.
    #   output_bits; size of output tensors.
    #   stress: parameter to reduce reference size to force tuner to
    #     choose smaller models.
    #   config: configuration on what to compute for each layer
    #     minimum configuration is { "default": ["parameters", "activations"] }

    # use simplest one - number of bits
    if not goal:
      goal = {
          "type": "bits",
          "params": {
              "delta_p": 8.0,
              "delta_n": 8.0,
              "rate": 2.0,
              "stress": 1.0,
              "input_bits": 8,
              "output_bits": 8,
              "ref_bits": 8,
              "config": {
                  "default": ["parameters", "activations"]
              }
          }
      }

    self.overwrite = overwrite

    # for multi-head model, we need to specify which head(/output) that
    # score and trial metric needs to calculate from
    self.head_name = head_name

    # if we have not created it already, create new one.
    if not isinstance(goal, ForgivingFactor):
      target = forgiving_factor[goal["type"]](**goal["params"])
    else:
      target = goal

    # if no metrics were specified, we want to make sure we monitor at least
    # accuracy.
    if not metrics:
      metrics = ["acc"]

    self.hypermodel = AutoQKHyperModel(
        model, metrics, custom_objects, target,
        transfer_weights=transfer_weights,
        frozen_layers=frozen_layers,
        activation_bits=activation_bits,
        limit=limit,
        tune_filters=tune_filters,
        tune_filters_exceptions=tune_filters_exceptions,
        layer_indexes=layer_indexes,
        learning_rate_optimizer=learning_rate_optimizer,
        head_name=head_name,
        quantization_config=quantization_config
    )

    # right now we create unique results directory
    idx = 0
    name = output_dir
    if self.overwrite:
      while os.path.exists(name):
        idx += 1
        name = output_dir + "_" + str(idx)
    output_dir = name
    self.output_dir = output_dir

    if score_metric is None:
      if self.head_name:
        score_metric = "val_" + self.head_name + "_score"
      else:
        score_metric = "val_score"
    assert mode in ["random", "bayesian", "hyperband"]
    if custom_tuner is not None:
      self.tuner = custom_tuner(
          self.hypermodel,
          autoqkeras_config=autoqkeras_input_args,
          objective=kt.Objective(score_metric, "max"),
          project_name=output_dir,
          **tuner_kwargs)
    elif mode == "random":
      self.tuner = RandomSearch(
          self.hypermodel,
          objective=kt.Objective(score_metric, "max"),
          project_name=output_dir,
          **tuner_kwargs)
    elif mode == "bayesian":
      self.tuner = BayesianOptimization(
          self.hypermodel,
          objective=kt.Objective(score_metric, "max"),
          project_name=output_dir,
          **tuner_kwargs)
    elif mode == "hyperband":
      self.tuner = Hyperband(
          self.hypermodel,
          objective=kt.Objective(score_metric, "max"),
          project_name=output_dir,
          **tuner_kwargs)
    else:
      pass

    self.tuner.search_space_summary()

  def _has_earlystopping(self, callbacks):
    """Check if EarlyStopping has been defined or not."""
    if callbacks is None:
      return False

    for callback in callbacks:
      if isinstance(callback, tf.keras.callbacks.EarlyStopping):
        return True
    return False

  def history(self, number_of_trials=-1):
    """Returns the history of the model search."""
    trials = self.tuner.oracle.get_best_trials(number_of_trials)
    state = [trial.get_state() for trial in trials]

    result = {}
    result["score"] = [
        state[i]["score"] for i in range(len(state))
        if trials[i].score is not None
    ]
    for i in range(len(state)):
      if trials[i].score is not None:
        keys = state[i]["metrics"]["metrics"].keys()

        for key in keys:
          if key != "score" and not key.startswith(
              "val_") and key != "loss" and key != "trial":

            cur_accuracy = state[i]["metrics"]["metrics"][key][
                "observations"][0]["value"][0]
            if "val_" + key in state[i]["metrics"]["metrics"].keys():
              cur_val_accuracy = state[i]["metrics"]["metrics"]["val_" + key][
                  "observations"][0]["value"][0]
            else:
              cur_val_accuracy = None

            # only update result if both key and val_key exist
            if cur_val_accuracy:
              if key not in result.keys():
                result[key] = [cur_accuracy]
                result["val_" + key] = [cur_val_accuracy]
              else:
                result[key].append(cur_accuracy)
                result["val_" + key].append(cur_val_accuracy)

    if self.head_name:
      trial_from_output = self.head_name + "_trial"
    else:
      trial_from_output = "trial"
    result["trial_size"] = [
        state[i]["metrics"]["metrics"][trial_from_output]["observations"][0]
        ["value"][0] for i in range(len(state)) if trials[i].score is not None
    ]

    return result

  def fit(self, *fit_args, **fit_kwargs):
    """Invokes tuner fit algorithm."""

    callbacks = fit_kwargs.get("callbacks", None)

    if callbacks is None:
      callbacks = []

    epochs = fit_kwargs.get("epochs", None)

    if epochs is None:
      epochs = 10

    if not self._has_earlystopping(callbacks):
      callbacks = callbacks + [
          tf.keras.callbacks.EarlyStopping(
              "val_loss", patience=min(20, epochs//5))
      ]
      fit_kwargs["callbacks"] = callbacks

    self.tuner.search(*fit_args, **fit_kwargs)

  @staticmethod
  def get_best_lr(qmodel):
    """Extracts best lr of model."""
    return qmodel.optimizer.lr.numpy()

  def get_best_model(self):
    params = self.tuner.get_best_hyperparameters()[0]

    q_model = self.tuner.hypermodel.build(params)

    self.learning_rate = q_model.optimizer.lr.numpy()

    return q_model

  def get_learning_rate(self):
    return self.learning_rate


class AutoQKerasScheduler:
  """Performs autoquantization one layer/group at a time.

     Arguments:
       model: Model to be quantized.
       metrics: List of metrics to be monitored.
       custom_objects: Custom objects used by Keras during quantization.
       goal: Metric to compute secondary goal of search (bits or energy)
       output_dir: name of output directory to store results.
       mode: random, hyperband or bayesian used by keras_tuner.
       transfer_weights: if true, transfer weights from unquantized model.
       activation_bits: parameter to be used by 'model_quantize'.
       limit: limit the number of bits in quantizers specified as a dictionary.
       tune_filters: one of "block", "layer", "none" for tuning entire
         network, each layer separately, or no tuning.
       tune_filters_exceptions: name of layers that will not be tuned.
       layer_indexes: indexes of layer to be quantized.
       learning_rate_optimizer: if true, user will provide lr scheduler
         callback.
       blocks: list of re patterns specifygin group configuration for layers.
       schedule_block: "sequential" or "cost". Schedule blocks using the
         order of the groups or decreasing cost (energy or bits).
       quantization_config: file name of dictionary containing configuration of
         quantizers for kernel, bias and activation.
       debug: if True, fit will just print the groups for debugging purposes.
       head_name: specify which head to calcuate score/trial-size from in
         autoqkeras
       tuner_kwargs: parameters for keras_tuner depending on whether
         mode is random, hyperband or baeysian. Please refer to the
         documentation of kerstuner Tuners.
  """

  def __init__(
      self, model, metrics=None, custom_objects=None, goal=None,
      output_dir="result", mode="random", transfer_weights=False,
      activation_bits=4, limit=None, tune_filters="none",
      tune_filters_exceptions=None, layer_indexes=None,
      learning_rate_optimizer=False, blocks=None, schedule_block="sequential",
      quantization_config=None, overwrite=True, debug=False, head_name=None,
      **tuner_kwargs):

    if not metrics:
      metrics = []

    if not custom_objects:
      custom_objects = {}

    # goal: { "type": ["bits", "energy"], "params": {...} }
    # For type == "bits":
    #   delta_p: increment (in %) of the accuracy if trial is smaller.
    #   delta_n: decrement (in %) of the accuracy if trial is bigger.
    #   rate: rate of decrease/increase in model size in terms of bits.
    #   input_bits; size of input tensors.
    #   output_bits; size of output tensors.
    #   stress: parameter to reduce reference size to force tuner to
    #     choose smaller models.
    #   config: configuration on what to compute for each layer
    #     minimum configuration is { "default": ["parameters", "activations"] }

    # use simplest one - number of bits
    if not goal:
      goal = {
          "type": "bits",
          "params": {
              "delta_p": 8.0,
              "delta_n": 8.0,
              "rate": 2.0,
              "stress": 1.0,
              "input_bits": 8,
              "output_bits": 8,
              "ref_bits": 8,
              "config": {
                  "default": ["parameters", "activations"]
              }
          }
      }

    self.target = forgiving_factor[goal["type"]](**goal["params"])

    self.model = model
    self.metrics = metrics
    self.custom_objects = custom_objects
    self.mode = mode
    self.transfer_weights = transfer_weights
    self.activation_bits = activation_bits
    self.limit = limit
    self.tune_filters = tune_filters
    self.tune_filters_exceptions = tune_filters_exceptions
    self.layer_indexes = layer_indexes
    self.learning_rate_optimizer = learning_rate_optimizer
    self.blocks = blocks
    self.schedule_block = schedule_block
    self.quantization_config = quantization_config
    self.tuner_kwargs = tuner_kwargs
    self.debug = debug
    self.head_name = head_name

    self.autoqk = None
    self.learning_rate = model.optimizer.lr.numpy()
    self.overwrite = overwrite

    assert self.schedule_block in ["sequential", "cost"]

    # right now we create unique results directory
    idx = 0
    name = output_dir
    if self.overwrite:
      while os.path.exists(name):
        idx += 1
        name = output_dir + "_" + str(idx)
    output_dir = name
    self.output_dir = output_dir
    self.next_block = self.get_next_block(overwrite)
    if self.next_block > 0:
      strategy = self.tuner_kwargs.get("distribution_strategy", None)
      if strategy:
        with strategy.scope():
          self.model = tf.keras.models.load_model(
              os.path.join(
                  self.output_dir, "model_block_" + str(self.next_block - 1)),
              custom_objects=self.custom_objects)
      else:
        self.model = tf.keras.models.load_model(
            os.path.join(
                self.output_dir, "model_block_" + str(self.next_block - 1)),
            custom_objects=self.custom_objects)
      print("Load model completed")

  def get_next_block(self, overwrite):
    """Get the next block id to be worked on."""
    if overwrite:
      return 0
    else:
      try:
        with tf.io.gfile.GFile(os.path.join(self.output_dir, "scheduler.json"),
                               "r") as f:
          scheduler_json = f.read()
        scheduler = json.loads(scheduler_json)
        return scheduler["next_block"]
      except:  # pylint: disable=bare-except
        return 0

  def get_limit(self, model, pattern):
    """Apply patterned group to limit to obtain new limit set."""
    limit = self.limit
    new_limit = {}
    new_pattern = collections.defaultdict(list)

    for layer_name in self.grouped_patterns[pattern]:
      layer = model.get_layer(layer_name)
      layer_class_name = layer.__class__.__name__

      target_quantizers = limit.get(layer_class_name, -1)
      for limit_pattern in limit:
        if re.match(limit_pattern, layer_name):
          target_quantizers = limit[limit_pattern]
          new_pattern[limit_pattern].append(layer_name)
          layer_name = limit_pattern
          break
      if target_quantizers != -1:
        new_limit[layer_name] = target_quantizers

    for key in new_pattern:
      # grouped pattern in regex need to be ^(word1|word2|...)$ instead of
      # ^word1|word2|...$; otherwise it cause non-exact match,
      # e.g., fc.*_0 and fc.*_0_relu were miss-matched
      new_key = "^" + "(" + "|".join(new_pattern[key]) + ")" + "$"
      new_limit[new_key] = new_limit[key]
      if new_key != key:
        del new_limit[key]

    return new_limit

  def fit(self, *fit_args, **fit_kwargs):
    """Invokes tuner fit algorithm."""

    self.history = []
    self.compute_block_costs(self.blocks, self.model)

    if self.tuner_kwargs.get("max_trials", None):
      max_trials = float(self.tuner_kwargs["max_trials"])

    lr = self.model.optimizer.lr.numpy()

    model = self.model

    frozen_layers = []

    for i, (pattern, cost) in enumerate(self.retrieve_max_block()):

      # now create new limit pattern
      if not self.overwrite:
        if i < self.next_block:
          print("Resume tuning. Skipping block ", i)
          continue

      print("... block cost: {:.0f} / {:.0f}".format(cost, self.reference_size))

      if self.tuner_kwargs.get("max_trials", None):
        self.tuner_kwargs["max_trials"] = int(
            max(10, max_trials * cost / self.reference_size))
        print("... adjusting max_trials for this block to {}".format(
            self.tuner_kwargs["max_trials"]))

      limit = self.get_limit(model, pattern)
      new_frozen_layers = self.grouped_patterns[pattern]

      # if dictionary is empty we did not match anything.
      # we have a bug in the patterns specified by the
      # user.

      assert limit

      print("Pattern {} is : {}".format(i, limit))

      if self.debug:
        frozen_layers = frozen_layers + new_frozen_layers
        continue

      self.autoqk = AutoQKeras(
          model, self.metrics,
          custom_objects=self.custom_objects,
          goal=self.target,
          output_dir=self.output_dir + "/" + str(i),
          mode=self.mode,
          transfer_weights=self.transfer_weights,
          frozen_layers=frozen_layers,
          activation_bits=self.activation_bits,
          limit=limit,
          tune_filters=self.tune_filters,
          tune_filters_exceptions=self.tune_filters_exceptions,
          layer_indexes=self.layer_indexes,
          learning_rate_optimizer=self.learning_rate_optimizer,
          quantization_config=self.quantization_config,
          overwrite=self.overwrite,
          head_name=self.head_name,
          **self.tuner_kwargs)

      self.autoqk.fit(*fit_args, **fit_kwargs)

      self.autoqk.tuner.results_summary()

      self.history.append(self.autoqk.history())

      model = self.autoqk.get_best_model()
      self.learning_rate = model.optimizer.lr.numpy()

      # restore learning rate
      # this is just a placeholder for the optimizer.

      model.compile(
          model.optimizer,
          loss=self.model.loss,
          metrics=self.model.metrics)

      frozen_layers = frozen_layers + new_frozen_layers

      filename = self.output_dir + "/model_block_" + str(i)
      model.save(filename)
      self.next_block = i + 1

      # update scheduler json
      with tf.io.gfile.GFile(os.path.join(self.output_dir, "scheduler.json"),
                             "w") as f:
        f.write(json.dumps({"next_block": self.next_block}))

    if self.debug:
      return

    self.best_model = model

    # make all layers trainable again
    for layer_name in frozen_layers:
      layer = model.get_layer(layer_name)
      layer.trainable = True

  def compute_block_costs(self, patterns, model):
    """Computes costs for each block."""

    # get block cost for original model
    self.reference_size = self.target.get_reference(model)
    self.model_size = self.target.get_reference_stats()

    # first group layers into the patterns

    groups = {pattern: [] for pattern in patterns}

    for layer_id, layer in enumerate(model.layers):
      if (
          self.layer_indexes is not None and
          layer_id not in self.layer_indexes
      ):
        continue

      for pattern in groups:
        if re.match(pattern, layer.name):
          groups[pattern].append(layer.name)

    self.grouped_patterns = groups

    # now compute cost for each group

    self.costs = []
    for pattern in patterns:  # self.grouped_patterns:
      total = 0
      for layer in self.grouped_patterns[pattern]:
        if layer in self.model_size:
          total += self.model_size[layer]["total"]
      self.costs.append((pattern, total))

    # the costs will be sorted by the total cost of the group
    if self.schedule_block == "cost":
      self.costs = sorted(self.costs, key=lambda cost_tuple: -cost_tuple[1])

  def retrieve_max_block(self):
    for cost in self.costs:
      yield cost

  def get_history(self):
    """Returns the history of the model search."""
    return self.history

  def get_best_model(self):
    """Returns the best model."""

    # check if we have run fit first.
    if not self.autoqk:
      return None

    self.autoqk.hypermodel.target.print_stats()
    print_qmodel_summary(self.best_model)

    return self.best_model

  def get_learning_rate(self):
    return self.learning_rate
