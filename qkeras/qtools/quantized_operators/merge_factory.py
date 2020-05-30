# Lint as: python3
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
"""create merge layer output quantizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from qkeras.qtools.quantized_operators import adder_impl
from qkeras.qtools.quantized_operators import multiplier_factory
from qkeras.qtools.quantized_operators import quantizer_impl


class MergeFactory:
  """determine which merge implementation to use."""

  def make_quantizer(self, input_qe_list, layer_type):
    """make quantier."""

    if layer_type == "Add":
      return Add(input_qe_list)
    elif layer_type == "Multiply":
      return Multiply(input_qe_list)
    elif layer_type == "Maximum":
      return Maximum(input_qe_list)
    elif layer_type == "Minimum":
      return Minimum(input_qe_list)
    elif layer_type == "Average":
      return Average(input_qe_list)
    elif layer_type == "Concatenate":
      return Concatenate(input_qe_list)
    elif layer_type == "Dot":
      return Dot(input_qe_list)


class IMerger(abc.ABC):
  """abstract class for merge quantizer."""

  def __init__(self, input_qe_list):
    self.input_quantizers = []
    self.edges = []

    for node in input_qe_list:
      self.input_quantizers.append(node[0])
      self.edges.append(node[1])


class Add(IMerger):
  """add a list of inputs."""

  # It takes as input a list of tensors, all of the same shape,
  # and returns a single tensor (also of the same shape).

  def __init__(self, input_qe_list):
    super().__init__(input_qe_list)

    max_bits = -1
    max_int_bits = -1
    is_signed = False

    bits = 0
    is_floating_point = False
    for quantizer in self.input_quantizers:
      if quantizer.is_floating_point:
        is_floating_point = True
        bits = max(bits, quantizer.bits)
      else:
        if quantizer.is_po2:
          qbits_quantizer = adder_impl.po2_qbits_converter(
              quantizer)
        else:
          qbits_quantizer = quantizer

        if qbits_quantizer.bits > max_bits:
          max_bits = qbits_quantizer.bits

        if qbits_quantizer.int_bits > max_int_bits:
          max_int_bits = qbits_quantizer.int_bits

      is_signed |= quantizer.is_signed

    if is_floating_point:
      self.output = quantizer_impl.FloatingPoint(
          bits=bits)
    else:
      self.output = quantizer_impl.QuantizedBits()
      self.output.bits = max_bits + 1
      self.output.int_bits = max_int_bits + 1
      self.output.is_signed = is_signed
      self.output.mode = 0
      self.output.is_floating_point = False
      self.output.is_po2 = 0

    self.gate_factor = 1
    self.gate_bits = self.output.bits

  def implemented_as(self):
    return "add"


class Multiply(IMerger):
  """multiplies (element-wise) a list of inputs."""

  # It takes as input a list of tensors, all of the same shape,
  # and returns a single tensor (also of the same shape).

  def  __init__(self, input_qe_list):
    super().__init__(input_qe_list)
    multiplier_instance = multiplier_factory.MultiplierFactory()

    quantizer = self.input_quantizers[0]
    for cur in self.input_quantizers[1:]:
      tmp = multiplier_instance.make_multiplier(quantizer, cur)
      quantizer = tmp.output

    self.output = quantizer

    # TODO(lishanok): only use the last multiplier here
    self.impl_class = tmp
    self.gate_factor = tmp.gate_factor
    self.gate_bits = tmp.gate_bits

  def implemented_as(self):
    return self.impl_class.implemented_as()


class Maximum(IMerger):
  """maximum of a list of inputs."""

  # It takes as input a list of tensors, all of the same shape,
  # and returns a single tensor (also of the same shape).

  def __init__(self, input_qe_list):
    super().__init__(input_qe_list)

    is_same = True
    is_floating_point = False
    bits = 0

    quantizer = self.input_quantizers[0]
    for cur in self.input_quantizers[1:]:
      if (quantizer.name != cur.name or quantizer.bits != cur.bits or
          quantizer.int_bits != cur.int_bits or
          quantizer.is_signed != cur.is_signed):
        is_same = False
        break

    if is_same:
      self.output = quantizer
    else:
      max_bits = -1
      max_int_bits = -1
      is_signed = False
      for quantizer in self.input_quantizers:
        if quantizer.is_floating_point:
          is_floating_point = True
          bits = max(bits, quantizer.bits)
        else:
          if quantizer.is_po2:
            qbits_quantizer = adder_impl.po2_qbits_converter(
                quantizer)
          else:
            qbits_quantizer = quantizer

          if qbits_quantizer.bits > max_bits:
            max_bits = qbits_quantizer.bits

          if qbits_quantizer.int_bits > max_int_bits:
            max_int_bits = qbits_quantizer.int_bits

        is_signed |= quantizer.is_signed

      if is_floating_point:
        self.output = quantizer_impl.FloatingPoint(
            bits=bits)
      else:
        self.output = quantizer_impl.QuantizedBits()
        self.output.bits = max_bits
        self.output.int_bits = max_int_bits
        self.output.is_signed = is_signed
        self.output.mode = 0
        self.output.is_floating_point = False
        self.output.is_po2 = 0

    self.gate_factor = 0.2
    self.gate_bits = self.output.bits

  @staticmethod
  def implemented_as():
    return "add"


class Minimum(Maximum):
  """minimum (element-wise) a list of inputs."""

  # It takes as input a list of tensors, all of the same shape,
  # and returns a single tensor (also of the same shape).
  pass


class Average(Maximum):
  """average (element-wise) a list of inputs."""

  # It takes as input a list of tensors, all of the same shape,
  # and returns a single tensor (also of the same shape).
  def __init__(self, input_qe_list):
    super().__init__(input_qe_list)

    self.gate_factor = 1
    self.gate_bits = self.output.bits


class Concatenate(Maximum):
  """Layer that concatenates a list of inputs."""

  # It takes as input a list of tensors, all of the same
  # shape except for the concatenation axis, and returns
  # a single tensor, the concatenation of all inputs..
  def __init__(self, input_qe_list):
    super().__init__(input_qe_list)

    self.gate_factor = 0
    self.gate_bits = self.output.bits


# TODO(lishanok): finish DOT ndimension tensor logic
class Dot(IMerger):
  """dot product between samples in two tensors."""

  # E.g. if applied to a list of two tensors a and b
  # of shape (batch_size, n), the
  # output will be a tensor of shape (batch_size, 1)
  # where each entry i will be\
  # the dot product between a[i] and b[i].

  pass
