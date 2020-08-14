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

import warnings
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, Embedding, Input
from tensorflow.keras.models import Model
from sklearn.cluster import KMeans

from .quantizers import get_quantizer


def create_in_out_table(km, quantizer):
  """Create [in, out] table needed to map compressed activations to codebook values 
  Given v: in_table[out_table[v]] => codebook value of v

  Arguments:
    km: KMeans model
    quantizer: quantizer function to apply to out_table

  Returns 
    in_table: conversion of compressed table indexes to n-bit numbers
    out_table: conversion of n-bit output activations to compressed table indexes
  """
  in_table = km.cluster_centers_.flatten()
  out_table = km.predict(quantizer.range().reshape(-1, 1).astype(np.float32)).flatten()   
  return [in_table, out_table]


def create_indirect_indexes(model, compile_config, activation_indexes, 
                            bits, X_train, y_train, X_test, y_test, sample_size=1.0):
  """This model applies clustering based non-uniform quantization inspired by 
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
    if i in activation_indexes or i == len(model.layers)-1:
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
      idxs = np.random.choice(x.shape[0], size=int(0.1*x.shape[0]))
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
    x = km.cluster_centers_[km.predict(x.flatten().reshape(-1, 1))].reshape(x.shape)
    n_unique = np.unique(x.flatten()).shape[0]
    print(f"Number of unique activations: {n_unique}")
    assert n_unique <= 2**bits

  print('\nEvaluating...')
  models[-1].evaluate(x, y_test, verbose=2)
  return cb_tables, models, km_models

def codebook_embeddings(embeddings, bits, quantizer, rowwise=False):
  """Creates a quantized embedding matrix based on the
  idea presented by https://arxiv.org/pdf/1911.02079.pdf
  
  Arguments:
    embeddings: Numpy array MxN
    bits: Number of bits to compress weights to. This will 
      results in 2**bits codebook values
    quantizer: quantizer function that will be applied to codebook values
    rowwise: if true, rowwise clustering is applied. Otherwise Two-tier clustering 

  Returns:
    km_models: list of fitted KMeans algorithms 
    quantized_embeddings: Numpy array MxN 
  """
  from tqdm import tqdm
  n = 2**bits
  if rowwise:
    print('Rowwise clustering quantization...')
    quantized_embeddings = np.zeros(embeddings.shape)
    km_models = [None] * embeddings.shape[0]
    for i in tqdm(range(quantized_embeddings.shape[0])):
      km = KMeans(n)
      km.fit(embeddings[i, :].reshape(-1, 1))
      km.cluster_centers_ = quantizer(km.cluster_centers_).numpy()
      km.cluster_centers_.sort(axis=0)
      km_models[i] = km
      quantized_embeddings[i, :] = km.cluster_centers_[km.predict(
                                    embeddings[i,:].reshape(-1, 1))].flatten()
    assert np.unique(quantized_embeddings[32]).shape[0] <= n
    return km_models, quantized_embeddings

  else:
    print('Two-tier clustering quantization...')
    quantized_embeddings = embeddings.copy()
    km1 = KMeans(n)
    km1.fit(embeddings)
    tier1 = km1.predict(embeddings)

    km_models = [0]*n
    block_sizes = [0]*n
    for block_label in tqdm(range(n)):
      mask = block_label == tier1
      indices = np.arange(quantized_embeddings.shape[0])[mask]
      block = quantized_embeddings[mask]
      km2 = KMeans(n)
      km2.fit(block.flatten().reshape(-1, 1))
      km2.cluster_centers_ = quantizer(km2.cluster_centers_).numpy()
      km2.cluster_centers_.sort(axis=0)
      km_models[block_label] = km2
      block_sizes[block_label] = block.shape[0]
      for i in indices:
        quantized_embeddings[i, :] = km2.cluster_centers_[km2.predict(
                                          quantized_embeddings[i,:].reshape(-1, 1))].flatten()
    print('block_sizes:', block_sizes)
    assert np.unique(quantized_embeddings[320]).shape[0] <= n # could be less since it is block based 
    return km_models, quantized_embeddings
     