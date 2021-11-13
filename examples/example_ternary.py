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
from __future__ import absolute_import  # Not necessary in a Python 3-only module
from __future__ import division  # Not necessary in a Python 3-only module
from __future__ import print_function  # Not necessary in a Python 3-only module

from absl import app
from absl import flags
import matplotlib
import numpy as np

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


FLAGS = flags.FLAGS


def _stochastic_rounding(x, precision, resolution, delta):
  """Stochastic_rounding for ternary.

  Args:
    x:
    precision: A float. The area we want to make this stochastic rounding.
       [delta-precision, delta] [delta, delta+precision]
    resolution: control the quantization resolution.
    delta: the undiscountinued point (positive number)

  Return:
    A tensor with stochastic rounding numbers.
  """
  delta_left = delta - precision
  delta_right = delta + precision
  scale = 1 / resolution
  scale_delta_left = delta_left * scale
  scale_delta_right = delta_right * scale
  scale_2_delta = scale_delta_right - scale_delta_left
  scale_x = x * scale
  fraction = scale_x - scale_delta_left
  # print(precision, scale, x[0], np.floor(scale_x[0]), scale_x[0], fraction[0])

  # we use uniform distribution
  random_selector = np.random.uniform(0, 1, size=x.shape) * scale_2_delta

  # print(precision, scale, x[0], delta_left[0], delta_right[0])
  # print('x', scale_x[0], fraction[0], random_selector[0], scale_2_delta[0])
  # rounddown = fraction < random_selector
  result = np.where(fraction < random_selector,
                    scale_delta_left / scale,
                    scale_delta_right / scale)
  return result


def _ternary(x, sto=False):
  m = np.amax(np.abs(x), keepdims=True)
  scale = 2 * m / 3.0
  thres = scale / 2.0
  ratio = 0.1

  if sto:
    sign_bit = np.sign(x)
    x = np.abs(x)
    prec = x / scale
    x = (
        sign_bit * scale * _stochastic_rounding(
            x / scale,
            precision=0.3, resolution=0.01, # those two are all normalized.
            delta=thres / scale))
    # prec + prec *ratio)
    # mm = np.amax(np.abs(x), keepdims=True)
  return np.where(np.abs(x) < thres, np.zeros_like(x), np.sign(x))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # x = np.arange(-3.0, 3.0, 0.01)
  # x = np.random.uniform(-0.01, 0.01, size=1000)
  x = np.random.uniform(-10.0, 10.0, size=1000)
  # x = np.random.uniform(-1, 1, size=1000)
  x = np.sort(x)
  tr = np.zeros_like(x)
  t = np.zeros_like(x)
  iter_count = 500
  for _ in range(iter_count):
    y = _ternary(x)
    yr = _ternary(x, sto=True)
    t = t + y
    tr = tr + yr

  plt.plot(x, t/iter_count)
  plt.plot(x, tr/iter_count)
  plt.ylabel('mean (%s samples)' % iter_count)
  plt.show()


if __name__ == '__main__':
  app.run(main)
