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
"""Exports qkeras modules to quantizer package."""

# We use wildcard import for convenience at this moment, which will be later
# refactored and removed.
from .b2t import *  # pylint: disable=wildcard-import
from .estimate import *  # pylint: disable=wildcard-import
from .qlayers import *  # pylint: disable=wildcard-import
from .quantizers import *  # pylint: disable=wildcard-import
from .qconvolutional import *  # pylint: disable=wildcard-import
from .qnormalization import * # pylint: disable=wildcard-import
from .qoctave import *  # pylint: disable=wildcard-import
from .qpooling import *  # pylint: disable=wildcard-import
from .safe_eval import *  # pylint: disable=wildcard-import
#from .qtools import *  # pylint:disable=wildcard-import

__version__ = "0.8.0"
