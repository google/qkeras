# Copyright 2024 Google LLC
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
"""Registry for QKeras quantizers."""

from . import registry

# Global registry for all QKeras quantizers.
_QUANTIZERS_REGISTRY = registry.Registry()


def register_quantizer(quantizer):
  """Decorator for registering a quantizer."""
  _QUANTIZERS_REGISTRY.register(quantizer)
  # Return the quantizer after registering. This ensures any registered
  # quantizer class is properly defined.
  return quantizer


def lookup_quantizer(name):
  """Retrieves a quantizer from the quantizers registry."""
  return _QUANTIZERS_REGISTRY.lookup(name)
