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
"""Implements total/partial Binary to Thermometer decoder."""

from tensorflow.keras.utils import to_categorical
import numpy as np


def BinaryToThermometer(
    x, classes, value_range, with_residue=False, merge_with_channels=False,
    use_two_hot_encoding=False):

  """Converts binary to one-hot (with scales).

  Given input matrix x with values (for example) 0, 1, 2, 3, 4, 5, 6, 7, create
  a number of classes as follows:

  classes=2, value_range=8, with_residue=0

  A true one-hot representation, and the remaining bits are truncated, using
  one bit representation.

  0 - [1,0] 1 - [1,0] 2 - [1,0] 3 - [1,0]
  4 - [0,1] 5 - [0,1] 6 - [0,1] 7 - [0,1]

  classes=2, value_range=8, with_residue=1

  In this case, the residue is added to the one-hot class, and the class will
  use 2 bits (for the remainder) + 1 bit (for the one hot)

  0 - [1,0] 1 - [1.25,0] 2 - [1.5,0] 3 - [1.75,0]
  4 - [0,1] 5 - [0,1.25] 6 - [0,1.5] 7 - [0,1.75]

  Arguments:
    x: the input vector we want to convert. typically its dimension will be
      (B,H,W,C) for an image, or (B,T,C) or (B,C) for for a 1D signal, where
      B=batch, H=height, W=width, C=channels or features, T=time for time
      series.
    classes: the number of classes to (or log2(classes) bits) to use of the
      values.
    value_range: max(x) - min(x) over all possible x values (e.g. for 8 bits,
      we would use 256 here).
    with_residue: if true, we split the value range into two sets and add
      the decimal fraction of the set to the one-hot representation for partial
      thermometer representation.
    merge_with_channels: if True, we will not create a separate dimension
      for the resulting matrix, but we will merge this dimension with
      the last dimension.
    use_two_hot_encoding: if true, we will distribute the weight between
      the current value and the next one to make sure the numbers will always
      be < 1.

  Returns:
    Converted x with classes with the last shape being C*classes.

  """

  # just make sure we are processing floats so that we can compute fractional
  # values

  x = x.astype(np.float32)

  # the number of ranges are equal to the span of the original values
  # divided by the number of target classes.
  #
  # for example, if value_range is 256 and number of classes is 16, we have
  # 16 values (remaining 4 bits to redistribute).

  ranges = value_range/classes
  x_floor = np.floor(x / ranges)

  if use_two_hot_encoding:
    x_ceil = np.ceil(x / ranges)

  if with_residue:
    x_mod_f = (x - x_floor * ranges) / ranges

  # convert values to categorical. if use_two_hot_encoding, we may
  # end up with one more class because we need to distribute the
  # remaining bits to the saturation class. For example, if we have
  # value_range = 4 (0,1,2,3) and classes = 2, if we use_two_hot_encoding
  # we will have the classes 0, 1, 2, where for the number 3, we will
  # allocate 0.5 to bin 1 and 0.5 to bin 2 (namelly 3 = 0.5 * (2**2 + 2**1)).

  xc_f = to_categorical(x_floor, classes + use_two_hot_encoding)

  if with_residue:
    xc_f_m = xc_f == 1

    if use_two_hot_encoding:
      xc_c = to_categorical(x_ceil, classes + use_two_hot_encoding)
      xc_c_m = xc_c == 1
      if np.any(xc_c_m):
        xc_c[xc_c_m] = x_mod_f.reshape(xc_c[xc_c_m].shape)
      if np.any(xc_f_m):
        xc_f[xc_f_m] = (1.0 - x_mod_f.reshape(xc_f[xc_f_m].shape))
      xc_f += xc_c
    else:
      if np.any(xc_f_m):
        xc_f[xc_f_m] += x_mod_f.reshape(xc_f[xc_f_m].shape)

  if merge_with_channels and len(xc_f.shape) != len(x.shape):
    sz = xc_f.shape
    sz = sz[:-2] + (sz[-2] * sz[-1],)
    xc_f = xc_f.reshape(sz)

  return xc_f

