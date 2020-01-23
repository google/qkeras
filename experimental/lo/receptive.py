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

import math

from .utils import get_padding_value


def print_rf(layer_name, x):
  print("Layer {}:".format(layer_name))
  print(
      "\theight/width: {}\n\tstride: {}\n\teq_kernel_size: {}\n\tstart: {}\n".format(
          *x)
  )


def rf_computation_for_layer(layer, layer_in):
  k, s, p = layer
  n_in, j_in, r_in, start_in = layer_in

  n_out = int(math.floor((n_in + 2*p - k)/s)) + 1

  if s == 1 and p == 1:
    n_out = n_in

  actual_p = (n_out-1)*s - n_in + k
  p_r = math.ceil(actual_p/2)
  p_l = math.floor(actual_p/2)

  j_out = j_in * s

  r_out = r_in + (k-1)*j_in

  start_out = start_in + (int((k-1)/2) - p_l) * j_in

  return n_out, j_out, r_out, start_out


def model_to_receptive_field(model, i_name, o_name):
  layers_h = []
  layers_w = []

  i_layer = model.get_layer(i_name)
  o_layer = model.get_layer(o_name)

  # right now this only works for sequential layers

  i_index = model.layers.index(i_layer)
  o_index = model.layers.index(o_layer)

  for i in range(i_index, o_index+1):
    k_h, k_w = (1, 1)
    s_h, s_w = (1, 1)
    p_h, p_w = (0, 0)

    if hasattr(model.layers[i], "kernel_size"):
      kernel = model.layers[i].kernel_size

      if isinstance(kernel, int):
        kernel = [kernel, kernel]

      k_h, k_w = kernel[0], kernel[1]

    if hasattr(model.layers[i], "strides"):
      strides = model.layers[i].strides

      if isinstance(strides, int):
        strides = [strides, strides]

      s_h, s_w = strides[0], strides[1]

    if hasattr(model.layers[i], "padding"):
      padding = model.layers[i].padding

      if isinstance(padding, str):
        padding = [padding, padding]

      p_h = get_padding_value(padding[0], k_h)
      p_w = get_padding_value(padding[1], k_w)

    layers_h.append((k_h, s_h, p_h))
    layers_w.append((k_w, s_w, p_w))

  x_h = (i_layer.input.shape[1], 1, 1, 0.5)
  x_w = (i_layer.input.shape[2], 1, 1, 0.5)

  for l_h, l_w in zip(layers_h, layers_w):
    x_h = rf_computation_for_layer(l_h, x_h)
    x_w = rf_computation_for_layer(l_w, x_w)

  strides = (x_h[1], x_w[1])
  kernel = (x_h[2], x_w[2])
  padding = ("valid", "valid")

  return (strides, kernel, padding)

