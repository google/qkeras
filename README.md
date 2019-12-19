# QKeras

[github.com/google/qkeras](https://github.com/google/qkeras) 

[![Build Status](https://travis-ci.org/google/qkeras.svg?branch=master)](https://travis-ci.org/google/qkeras)


QKeras is a quantization extension to Keras that provides drop-in
replacement for some of the Keras layers, especially the ones that
creates parameters and activation layers, and perform arithmetic
operations, so that we can quickly create a deep quantized version of
Keras network.

According to Tensorflow documentation, Keras is a high-level API to
build and train deep learning models. It's used for fast prototyping,
advanced research, and production, with three key advantages:

- User friendly

Keras has a simple, consistent interface optimized for common use
cases. It provides clear and actionable feedback for user errors.

- Modular and composable

Keras models are made by connecting configurable building blocks
together, with few restrictions.

- Easy to extend

Write custom building blocks to express new ideas for research. Create
new layers, loss functions, and develop state-of-the-art models.

QKeras is being designed to extend the functionality of Keras using
Keras' design principle, i.e. being user friendly, modular and
extensible, adding to it being "minimally intrusive" of Keras native
functionality.

In order to successfully quantize a model, users need to replace
variable creating layers (Dense, Conv2D, etc) by their counterparts
(QDense, QConv2D, etc), and any layers that perform math operations
need to be quantized afterwards.

## Layers Implemented in QKeras

- QDense

- QConv1D

- QConv2D

- QDepthwiseConv2D

- QSeparableConv2D (depthwise + pointwise expanded, extended from
MobileNet SeparableConv2D implementation)

- QActivation

- QAveragePooling2D (in fact, an AveragePooling2D stacked with a 
QActivation layer for quantization of the result)

- QOctaveConv2D

It is worth noting that not all functionality is safe at this time to
be used with other high-level operations, such as with layer
wrappers. For example, Bidirectional layer wrappers are used with
RNNs.  If this is required, we encourage users to use quantization
functions invoked as strings instead of the actual functions as a way
through this, but we may change that implementation in the future.

QSeparableConv2D is implemented as a depthwise + pointwise quantized
expansions, which is extended from the SeparableConv2D implementation
of MobileNet. Finally, QBatchNormalization is still in its
experimental stage, as we have not seen the need to use this yet due
to the normalization and regularization effects of stochastic
activation functions.

A first attempt to create a safe mechanism in QKeras is the adoption
of QActivation is a wrap-up that provides an encapsulation around the
activation functions so that we can save and restore the network
architecture, and duplicate them using Keras interface, but this
interface has not been fully tested yet.

## Activation Layers Implemented in QKeras

- smooth_sigmoid(x)

- hard_sigmoid(x)

- binary_sigmoid(x)

- binary_tanh(x)

- smooth_tanh(x)

- hard_tanh(x)

- quantized_bits(bits=8, integer=0, symmetric=0, keep_negative=1)(x)

- bernoulli(alpha=1.0)(x)

- stochastic_ternary(alpha=1.0, threshold=0.33)(x)

- ternary(alpha=1.0, threshold=0.33)(x)

- stochastic_binary(alpha=1.0)(x)

- binary(alpha=1.0)(x)

- quantized_relu(bits=8, integer=0, use_sigmoid=0)(x)

- quantized_ulaw(bits=8, integer=0, symmetric=0, u=255.0)(x)

- quantized_tanh(bits=8, integer=0, symmetric=0)(x)

- quantized_po2(bits=8, max_value=-1)(x)

- quantized_relu_po2(bits=8, max_value=-1)(x)

The stochastic_* functions, bernoulli as well as quantized_relu and
quantized_tanh rely on stochastic versions of the activation
functions. They draw a random number with uniform distribution from
_hard_sigmoid of the input x, and result is based on the expected
value of the activation function. Please refer to the papers if you
want to understand the underlying theory, or the documentation in
qkeras/qlayers.py.

The parameters "bits" specify the number of bits for the quantization,
and "integer" specifies how many bits of "bits" are to the left of the
decimal point. Finally, our experience in training networks with
QSeparableConv2D, both quantized_bits and quantized_tanh that
generates values between [-1, 1), required symmetric versions of the
range in order to properly converge and eliminate the bias.

Every time we use a quantization for weights and bias that can
generate numbers outside the range [-1.0, 1.0], we need to adjust the
*_range to the number. For example, if we have a
quantized_bits(bits=6, integer=2) in a weight of a layer, we need to
set the weight range to 2**2. Similarly, for quantization functions
that accept an alpha parameter, we need to specify a range of alpha,
and for po2 type of quantizers, we need to specify the range of
max_value.

### Example

Suppose you have the following network.

An example of a very simple network is given below in Keras.


```python
from keras.layers import *

x = x_in = Input(shape)
x = Conv2D(18, (3, 3), name="first_conv2d")(x)
x = Activation("relu")(x)
x = SeparableConv2D(32, (3, 3))(x)
x = Activation("relu")(x)
x = Flatten()(x)
x = Dense(NB_CLASSES)(x)
x = Activation("softmax")(x)
```

You can easily quantize this network as follows:

```python
from keras.layers import *
from qkeras import *

x = x_in = Input(shape)
x = QConv2D(18, (3, 3),
        kernel_quantizer="stochastic_ternary",
        bias_quantizer="ternary", name="first_conv2d")(x)
x = QActivation("quantized_relu(3)")(x)
x = QSeparableConv2D(32, (3, 3),
        depthwise_quantizer=quantized_bits(4, 0, 1),
        pointwise_quantizer=quantized_bits(3, 0, 1),
        bias_quantizer=quantized_bits(3),
        depthwise_activation=quantized_tanh(6, 2, 1))(x)
x = QActivation("quantized_relu(3)")(x)
x = Flatten()(x)
x = QDense(NB_CLASSES,
        kernel_quantizer=quantized_bits(3),
        bias_quantizer=quantized_bits(3))(x)
x = QActivation("quantized_bits(20, 5)")(x)
x = Activation("softmax")(x)
```

The last QActivation is advisable if you want to compare results later on. 
Please find more cases under the directory examples.



## Related Work

QKeras has been implemented based on the work of "B.Moons et al. -
Minimum Energy Quantized Neural Networks", Asilomar Conference on
Signals, Systems and Computers, 2017 and "Zhou, S. et al. -
DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with
Low Bitwidth Gradients," but the framework should be easily
extensible. The original code from QNN can be found below.

https://github.com/BertMoons/QuantizedNeuralNetworks-Keras-Tensorflow

QKeras extends QNN by providing a richer set of layers (including
SeparableConv2D, DepthwiseConv2D, ternary and stochastic ternary
quantizations), besides some functions to aid the estimation for the
accumulators and conversion between non-quantized to quantized
networks. Finally, our main goal is easy of use, so we attempt to make
QKeras layers a true drop-in replacement for Keras, so that users can
easily exchange non-quantized layers by quantized ones.

### Acknowledgements

Portions of QKeras were derived from QNN.

https://github.com/BertMoons/QuantizedNeuralNetworks-Keras-Tensorflow

Copyright (c) 2017, Bert Moons where it applies

