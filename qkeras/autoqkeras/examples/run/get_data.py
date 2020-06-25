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
"""Extracts sample dataset from tfds."""

import numpy as np
from tensorflow.keras.utils import to_categorical
import tensorflow_datasets as tfds


def get_data(dataset_name, fast=False):
  """Returns dataset from tfds."""
  ds_train = tfds.load(name=dataset_name, split="train", batch_size=-1)
  ds_test = tfds.load(name=dataset_name, split="test", batch_size=-1)

  dataset = tfds.as_numpy(ds_train)
  x_train, y_train = dataset["image"].astype(np.float32), dataset["label"]

  dataset = tfds.as_numpy(ds_test)
  x_test, y_test = dataset["image"].astype(np.float32), dataset["label"]

  if len(x_train.shape) == 3:
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))

  x_train /= 256.0
  x_test /= 256.0

  x_mean = np.mean(x_train, axis=0)

  x_train -= x_mean
  x_test -= x_mean

  nb_classes = np.max(y_train) + 1
  y_train = to_categorical(y_train, nb_classes)
  y_test = to_categorical(y_test, nb_classes)

  print(x_train.shape[0], "train samples")
  print(x_test.shape[0], "test samples")

  if fast:
    i_train = np.arange(x_train.shape[0])
    np.random.shuffle(i_train)
    i_test = np.arange(x_test.shape[0])
    np.random.shuffle(i_test)

    s_x_train = x_train[i_train[0:fast]]
    s_y_train = y_train[i_train[0:fast]]
    s_x_test = x_test[i_test[0:fast]]
    s_y_test = y_test[i_test[0:fast]]
    return ((s_x_train, s_y_train), (x_train, y_train), (s_x_test, s_y_test),
            (x_test, y_test))
  else:
    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
  get_data("mnist")
  get_data("fashion_mnist")
  get_data("cifar10", fast=1000)
  get_data("cifar100")


