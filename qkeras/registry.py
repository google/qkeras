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
"""General purpose registy for registering classes or functions.

The registry can be used along with decorators to record any class/function.

Sample usage:
  # Setup registry with decorator.
  _REGISTRY = registry.Registry()
  def register(cls):
    _REGISTRY.register(cls)
  def lookup(name):
    return _REGISTRY.lookup(name)

  # Register instances.
  @register
  def foo_task():
    ...

  @register
  def bar_task():
    ...

  # Retrieve instances.
  def my_executor():
   ...
   my_task = lookup("foo_task")
   ...
"""


class Registry(object):
  """A registry class to record class representations or function objects."""

  def __init__(self):
    """Initializes the registry."""
    self._container = {}

  def register(self, item, name=None):
    """Register an item.

    Args:
     item: Python item to be recorded.
     name: Optional name to be used for recording item. If not provided,
       item.__name__ is used.
    """
    if not name:
      name = item.__name__
    self._container[name] = item

  def lookup(self, name):
    """Retrieves an item from the registry.

    Args:
      name: Name of the item to lookup.

    Returns:
      Registered item from the registry.
    """
    return self._container[name]
