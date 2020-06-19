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

from qkeras.autoqkeras.examples.run.networks import ConvBlockNetwork  # pylint: disable=line-too-long

def get_model(dataset):
  """Returns a model for the demo of AutoQKeras."""
  if dataset == "mnist":
    model = ConvBlockNetwork(
        shape=(28, 28, 1),
        nb_classes=10,
        kernel_size=3,
        filters=[16, 32, 48, 64, 128],
        dropout_rate=0.2,
        with_maxpooling=False,
        with_batchnorm=True,
        kernel_initializer="he_uniform",
        bias_initializer="zeros",
    ).build()

  elif dataset == "fashion_mnist":
    model = ConvBlockNetwork(
        shape=(28, 28, 1),
        nb_classes=10,
        kernel_size=3,
        filters=[16, [32]*3, [64]*3],
        dropout_rate=0.2,
        with_maxpooling=True,
        with_batchnorm=True,
        use_separable="mobilenet",
        kernel_initializer="he_uniform",
        bias_initializer="zeros",
        use_xnornet_trick=True
    ).build()

  elif dataset == "cifar10":
    model = ConvBlockNetwork(
        shape=(32, 32, 3),
        nb_classes=10,
        kernel_size=3,
        filters=[16, [32]*3, [64]*3, [128]*3],
        dropout_rate=0.2,
        with_maxpooling=True,
        with_batchnorm=True,
        use_separable="mobilenet",
        kernel_initializer="he_uniform",
        bias_initializer="zeros",
        use_xnornet_trick=True
    ).build()

  elif dataset == "cifar100":
    model = ConvBlockNetwork(
        shape=(32, 32, 3),
        nb_classes=100,
        kernel_size=3,
        filters=[16, [32]*3, [64]*3, [128]*3, [256]*3],
        dropout_rate=0.2,
        with_maxpooling=True,
        with_batchnorm=True,
        use_separable="mobilenet",
        kernel_initializer="he_uniform",
        bias_initializer="zeros",
        use_xnornet_trick=True
    ).build()

  model.summary()

  return model
