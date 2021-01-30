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
"""Implements a safe evaluation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import operator
import pytest

from qkeras.safe_eval import GetParams
from qkeras.safe_eval import safe_eval


add = operator.add


def test_get_params1():
  s = "(3, 0.3, sep=5  )"
  args, kwargs = GetParams(s)
  assert args == [3, 0.3]
  assert kwargs == {"sep": 5}


def test_get_params2():
  s = "(  )"

  args, kwargs = GetParams(s)

  assert not args
  assert not kwargs


def test_get_params3():
  s = ("(3, 0.3,  -1.0, True, False, 'string1', num1=0.1, num2=-3.0, "
       "str1='string2', bool1=True, bool2=False)")

  args, kwargs = GetParams(s)

  assert args == [3, 0.3, -1.0, True, False, "string1"]
  assert kwargs == {
      "num1": 0.1,
      "num2": -3.0,
      "str1": "string2",
      "bool1": True,
      "bool2": False
  }


def test_safe_eval1():
  s = "add(3,3)"
  assert safe_eval(s, globals()) == 6


def i_func(s):
  return -s


def myadd2(a, b):
  return i_func(a) + i_func(b)


def myadd(a=32, b=10):
  return a + b

class myaddcls(object):
  def __call__(self, a=32, b=10):
    return a + b

def test_safe_eval2():
  s_add = [3, 39]
  assert safe_eval("add", globals(), *s_add) == 42


def test_safe_eval3():
  assert safe_eval("myadd()", globals()) == 42
  assert safe_eval("myadd(a=39)", globals(), b=3) == 42


def test_safe_eval4():
  assert safe_eval("myadd2(a=39)", globals(), b=3) == -42
  assert safe_eval("myadd2(a= 39)", globals(), b=3) == -42
  assert safe_eval("myadd2(a= 39, b = 3)", globals()) == -42

def test_safe_eval5():
  assert safe_eval("myadd", globals())(3,39) == 42
  assert safe_eval("myaddcls", globals())(3,39) == 42
  assert safe_eval("myaddcls()", globals())(3,39) == 42

if __name__ == "__main__":
  pytest.main([__file__])
