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
""" Clustering based quantizers """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from sklearn.cluster import KMeans
from tqdm import tqdm


def create_in_out_table(km, quantizer):
  """Create [in, out] table needed to map compressed activations to codebook
  values. Given v: in_table[out_table[v]] => codebook value of v

  Arguments:
    km: KMeans model
    quantizer: quantizer function to apply to out_table

  Returns
    in_table: conversion of compressed table indexes to n-bit numbers
    out_table: conversion of n-bit output activations to compressed table
      indexes
  """
  in_table = km.cluster_centers_.flatten()
  qrange = quantizer.range().reshape(-1, 1).astype(np.float32)
  out_table = km.predict(qrange).ravel()
  return in_table, out_table


def activation_compression(model, compile_config, activation_indexes, bits,
                           X_train, y_train, X_test, y_test, sample_size=1.0):
  """This function applies clustering based non-uniform quantization inspired by
  https://arxiv.org/pdf/1911.02079.pdf

  model: Keras model
  compile_config: Dictionary of arguments to be passed to model.compile()
    for all submodels
  activation_indexes: Index list of layers to be quantized. This will
    used to split the model and create submodels
  bits: Number of bits to compress activations to. This will
    results in 2**bits codebook values
  X_train, y_train: training data used to fit clustering algorithm
  X_test, y_test: validation data
  sample_size:
    fraction of training data activations to be used when computing
    codebook values

  Returns:
    cb_tables: [in, out] tables. See create_in_out_table docs
    models: list of keras submodels
    km_models: list of KMeans fitted models
  """
  assert len(activation_indexes) > 0
  assert 0.0 < sample_size <= 1.0
  km_models = [KMeans(2**bits)] * len(activation_indexes)
  cb_tables = [[]] * len(activation_indexes)
  models = []
  x = x_in = model.layers[0].output
  for i in range(1, len(model.layers)):
    layer = model.layers[i]
    x = layer(x)
    if i in activation_indexes or i == len(model.layers) - 1:
      print("\nCreating submodel...")
      models.append(Model([x_in], [x]))
      x = x_in = Input(layer.output[0].shape,
                       batch_size=layer.output.shape[0],
                       dtype=layer.output.dtype)
      models[-1].compile(**compile_config)
      print(models[-1].summary())
  print('\nsample_size: ', sample_size)
  x = X_train
  for i, model in enumerate(models[:-1]):
    print(f'fitting km[{i}]...')
    x = model.predict(x)
    km = km_models[i]
    temp = x.flatten().reshape(-1, 1)
    if sample_size < 1.0:
      idxs = np.random.choice(x.shape[0], size=int(sample_size * x.shape[0]))
      temp = temp[idxs]
    km.fit(temp)
    quantizer = getattr(model.layers[-1], 'quantizer',
                        getattr(model.layers[-1], 'activation'))
    km.cluster_centers_ = quantizer(km.cluster_centers_).numpy()
    km.cluster_centers_.sort(axis=0)
    cb_tables[i] = create_in_out_table(km, quantizer)
  x = X_test
  for i, model in enumerate(models[:-1]):
    x = model.predict(x)
    km = km_models[i]
    preds = km.predict(x.flatten().reshape(-1, 1))
    x = km.cluster_centers_[preds].reshape(x.shape)
    n_unique = np.unique(x.flatten()).shape[0]
    print(f"Number of unique activations: {n_unique}")
    assert n_unique <= 2**bits

  print('\nEvaluating...')
  models[-1].evaluate(x, y_test, verbose=2)
  return cb_tables, models, km_models


def weight_compression(weights, bits, axis=0, quantizer=None):
  """Creates an in, out table that maps weight values to their codebook values.
  Based on the idea presented by https://arxiv.org/pdf/1911.02079.pdf

  Arguments:
    weights: Numpy array
    bits: Number of bits to compress weights to. This will
      results in 2**bits codebook values
    axis: axis to apply quantization by
    quantizer: quantizer function that will be applied to codebook values

  Returns:
    index_table: array of indices that maps to codebook values for all weights
    codebook_table: array of codebook values
  """
  assert bits <= 8
  n = 2**bits
  index_table = []
  codebook_table = np.zeros((weights.shape[axis], n))
  km_models = [None] * weights.shape[axis]

  for i, w in tqdm(enumerate(np.split(weights, weights.shape[axis], axis))):
    original_shape = w.shape
    w = w.ravel()
    km = KMeans(n)
    km.fit(w.reshape(-1, 1))
    if quantizer:
      km.cluster_centers_ = quantizer(km.cluster_centers_).numpy()
    km.cluster_centers_.sort(axis=0)

    km_models[i] = km
    codebook_table[i, :] = km.cluster_centers_.flatten()
    preds = km.predict(w.reshape(-1, 1))
    index_table.append(preds.reshape(original_shape))

  index_table = np.concatenate(index_table, axis)
  return index_table, codebook_table


def two_tier_embedding_compression(embeddings, bits, quantizer=None):
  """ Creates tables that maps embedding values to their codebook values.
  Based on the idea presented by https://arxiv.org/pdf/1911.02079.pdf

  Arguments:
    weights: Numpy array
    bits: Number of bits to compress weights to. This will
      results in 2**bits codebook values
    quantizer: quantizer function that will be applied to codebook values

  Returns:
    index_table: array of indices that maps to codebook values
    cluster_index_table: array that maps each row to the codebook table
      index
    codebook_table: array of codebook values
    quantized_embeddings: Numpy array MxN of quantized weights
  """
  assert bits <= 8
  n = 2**bits
  quantized_embeddings = embeddings.copy()
  index_table = np.zeros(embeddings.shape, dtype=np.uint8)
  cluster_index_table = np.zeros(index_table.shape[0], dtype=np.uint8)
  codebook_table = np.zeros((n, n))

  km1 = KMeans(n)
  km1.fit(embeddings)
  tier1 = km1.predict(embeddings)

  km_models = [0] * n
  block_sizes = [0] * n
  for block_label in tqdm(range(n)):
    mask = block_label == tier1
    indices = np.arange(embeddings.shape[0])[mask]
    block = embeddings[mask]
    km2 = KMeans(n)
    km2.fit(block.flatten().reshape(-1, 1))
    if quantizer:
      km2.cluster_centers_ = quantizer(km2.cluster_centers_).numpy()
    km2.cluster_centers_.sort(axis=0)

    km_models[block_label] = km2
    codebook_table[block_label, :] = km2.cluster_centers_.flatten()
    cluster_index_table[indices] = block_label
    block_sizes[block_label] = block.shape[0]
    for i in indices:
      preds = km2.predict(embeddings[i, :].reshape(-1, 1))
      index_table[indices, :] = preds
      quantized_embeddings[i, :] = km2.cluster_centers_[preds].flatten()
  print('block_sizes:', block_sizes)
  return index_table, cluster_index_table, codebook_table, quantized_embeddings
