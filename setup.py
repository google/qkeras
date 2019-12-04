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
"""Setup script for qkeras."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
    name="QKeras",
    version="0.5.0",
    author="Claudionor N. Coelho",
    author_email="nunescoelho@google.com",
    maintainer="Hao Zhuang",
    maintainer_email="hzhuang@google.com",
    packages=setuptools.find_packages(),
    scripts=[],
    url="",
    license="LICENSE",
    description="Quantization package for Keras",
    long_description=long_description,
    install_requires=[
        "keras>=2.2.4",
        "numpy>=1.14.5,<1.17",
        "pyparser>=1.0",
        "scipy>=1.2.2,<1.3",
        "setuptools>=41.0.0",
        "tensorflow>=1.14.0,<2.0",
    ],
    setup_requires=[
        "pytest-runner",
    ],
    tests_require=[
        "pytest==4.6.5",
    ],
)
