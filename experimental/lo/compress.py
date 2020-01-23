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
"""Implements faster version of set on multiple strings."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Compressor:
  """Implements a hierarchical set class with better performance than a set."""

  def __init__(self, hash_only_input=False):
    self.n_dict = {}
    self.hash_only_input = hash_only_input

  def add_entry(self, table_in, table_out=""):
    """Adds entry (table_in, table_out) to the set."""
    line = (table_in, table_out)

    if self.hash_only_input:
      h_line = hash(table_in)
    else:
      h_line = hash(line)

    if self.n_dict.get(h_line, None):
      self.n_dict[h_line] = self.n_dict[h_line].union([line])
    else:
      self.n_dict[h_line] = set([line])

  def has_entry(self, table_in, table_out=""):
    """Checks if table_in is already stored in the set."""

    line = (table_in, table_out)

    if self.hash_only_input:
      h_line = hash(table_in)
    else:
      h_line = hash(line)

    if not self.n_dict.get(h_line, None):
      return None

    set_h_line = self.n_dict[h_line]

    for (ti, to) in set_h_line:
      if table_in == ti:
        return to

    return None

  def __call__(self):
    for key in self.n_dict:
      for line in self.n_dict[key]:
        yield line

