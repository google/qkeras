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
"""decides which quantizer implementation to use."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from qkeras import quantizers


from qkeras.qtools.quantized_operators import quantizer_impl
from qkeras.qtools.settings import cfg


class QuantizerFactory:
  """Convert qkeras quantizer to qtools quantizer type."""

  def __init__(self):
    self.quantizer_lookup = {
        quantizers.quantized_bits:
            quantizer_impl.QuantizedBits,
        quantizers.binary:
            quantizer_impl.Binary,
        quantizers.quantized_relu:
            quantizer_impl.QuantizedRelu,
        quantizers.ternary:
            quantizer_impl.Ternary,
        quantizers.quantized_relu_po2:
            quantizer_impl.ReluPowerOfTwo,
        quantizers.quantized_po2:
            quantizer_impl.PowerOfTwo,
        quantizers.stochastic_ternary:
            quantizer_impl.StochasticTernary,
        quantizers.stochastic_binary:
            quantizer_impl.StochasticBinary,
        quantizers.bernoulli:
            quantizer_impl.Bernoulli,
        quantizers.quantized_tanh:
            quantizer_impl.QuantizedTanh,
        quantizers.quantized_ulaw:
            quantizer_impl.QuantizedUlaw,
        
            

        # add following quantizer types for the use in GraphUpdateEdge
        quantizer_impl.QuantizedBits:
            quantizer_impl.QuantizedBits,
        quantizer_impl.Binary:
            quantizer_impl.Binary,
        quantizer_impl.QuantizedRelu:
            quantizer_impl.QuantizedRelu,
        quantizer_impl.Ternary:
            quantizer_impl.Ternary,
        quantizer_impl.ReluPowerOfTwo:
            quantizer_impl.ReluPowerOfTwo,
        quantizer_impl.PowerOfTwo:
            quantizer_impl.PowerOfTwo,
        quantizer_impl.FloatingPoint:
            quantizer_impl.FloatingPoint,
        quantizer_impl.StochasticTernary:
            quantizer_impl.StochasticTernary,
        quantizer_impl.StochasticBinary:
            quantizer_impl.StochasticTernary,
        quantizer_impl.Bernoulli:
            quantizer_impl.StochasticTernary,
        quantizer_impl.QuantizedTanh:
            quantizer_impl.StochasticTernary,
        quantizer_impl.QuantizedUlaw:
            quantizer_impl.StochasticTernary,
        
            
    }

    self._default_interm_quantizer = cfg.default_interm_quantizer

  def _make_quantizer_util(self, quantizer) -> quantizer_impl.IQuantizer:
    """make quantizer util function."""
    if quantizer in ["int8", "int16", "int32", "fp16", "fp32"]:
      return self.make_default_quantizer(mode=quantizer)

    elif isinstance(quantizer, tuple(self.quantizer_lookup.keys())):
      quantizer_class = self.quantizer_lookup[type(quantizer)]
      if quantizer_class == type(quantizer):
        return self.clone_quantizer(quantizer)
      else:
        q = quantizer_class()
        q.convert_qkeras_quantizer(quantizer)
        return q

    return None

  def make_quantizer(self, quantizer) -> quantizer_impl.IQuantizer:
    """create quantizer according to input qkeras quantizer."""

    q = None
    if quantizer is not None:
      q = self._make_quantizer_util(quantizer)

    if q is None:
      return self.make_default_quantizer(
          mode=self._default_interm_quantizer)

    return q

  def is_quantizer_supported(self, quantizer) -> bool:
    if quantizer is None:
      # if None, will use default quantizer defined in config.json
      return True

    return isinstance(quantizer, tuple(self.quantizer_lookup.keys()))

  def make_default_quantizer(self, mode) -> quantizer_impl.IQuantizer:
    """make quantizer given qkeras quantizer type."""
    if mode == "fp32":
      return quantizer_impl.FloatingPoint(
          bits=32)
    elif mode == "fp16":
      return quantizer_impl.FloatingPoint(
          bits=16)
    elif mode == "int8":
      qbits = quantizer_impl.QuantizedBits()
      qbits.convert_qkeras_quantizer(
          quantizers.quantized_bits(8, 0, 1))
      return qbits
    elif mode == "int16":
      qbits = quantizer_impl.QuantizedBits()
      qbits.convert_qkeras_quantizer(
          quantizers.quantized_bits(16, 7, 1))
      return qbits
    elif mode == "int32":
      qbits = quantizer_impl.QuantizedBits()
      qbits.convert_qkeras_quantizer(
          quantizers.quantized_bits(32, 10, 1))
      return qbits
    else:
      try:
        # string to quantizer object
        q_name = "quantizers." + mode
        qkeras_object = eval(q_name)  # pylint: disable=eval-used
        return self._make_quantizer_util(qkeras_object)
      except:  # pylint: disable=bare-except
        raise ValueError("unaccepted quantizer {}!".format(mode))

  def clone_quantizer(
      self, quantizer: quantizer_impl.IQuantizer) -> quantizer_impl.IQuantizer:
    """clone the given quantizer."""
    return copy.deepcopy(quantizer)
