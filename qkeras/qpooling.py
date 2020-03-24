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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import AveragePooling2D
from .qlayers import QActivation


def QAveragePooling2D(  # pylint: disable=invalid-name
    pool_size=(2, 2), strides=None, padding="valid", quantizer=None, **kwargs):
  """Computes the quantized version of AveragePooling2D."""

  # this is just a convenient layer, not being actually anything fancy. Just
  # reminds us that we need to quantize average pooling before the next layer.

  def _call(x):
    """Performs inline call to AveragePooling followed by QActivation."""

    x = AveragePooling2D(pool_size, strides, padding, **kwargs)(x)

    if kwargs.get("name", None):
      name = kwargs["name"] + "_act"
    else:
      name = None

    if quantizer:
      x = QActivation(quantizer, name=name)(x)

    return x

  return _call

