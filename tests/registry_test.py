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
"""Unit tests for registry."""

from numpy.testing import assert_equal
from numpy.testing import assert_raises
import pytest

from qkeras import registry


def sample_function(arg):
  """Sample function for testing."""
  return arg


class SampleClass(object):
  """Sample class for testing."""

  def __init__(self, arg):
    self._arg = arg

  def get_arg(self):
    return self._arg


def test_register_function():
  reg = registry.Registry()
  reg.register(sample_function)
  registered_function = reg.lookup('sample_function')
  # Call the function to validate.
  assert_equal(registered_function, sample_function)


def test_register_class():
  reg = registry.Registry()
  reg.register(SampleClass)
  registered_class = reg.lookup('SampleClass')
  # Create and call class object to validate.
  assert_equal(SampleClass, registered_class)


def test_register_with_name():
  reg = registry.Registry()
  name = 'NewSampleClass'
  reg.register(SampleClass, name=name)
  registered_class = reg.lookup(name)
  # Create and call class object to validate.
  assert_equal(SampleClass, registered_class)


def test_lookup_missing_item():
  reg = registry.Registry()
  assert_raises(KeyError, reg.lookup, 'foo')


def test_lookup_missing_name():
  reg = registry.Registry()
  sample_class = SampleClass(arg=1)
  # objects don't have a default __name__ attribute.
  assert_raises(AttributeError, reg.register, sample_class)

  # check that the object can be retrieved with a registered name.
  reg.register(sample_class, 'sample_class')
  assert_equal(sample_class, reg.lookup('sample_class'))


if __name__ == '__main__':
  pytest.main([__file__])
