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



def create_indirect_indexes(model, compile_config, activation_indexes, bits, X_train, y_train, X_test, y_test):
  assert len(activation_indexes) > 0
  
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
  
  x = X_train
  for i, model in enumerate(models[:-1]):
    x = model.predict(x)
    km = km_models[i]
    km.fit(x.flatten().reshape(-1, 1))
    km.cluster_centers_ = model.layers[-1].quantizer(km.cluster_centers_).numpy()
    cb_tables[i] = create_in_out_table(km, model.layers[-1].quantizer)
    
    
  x = X_test
  for i, model in enumerate(models[:-1]):
    x = model.predict(x)
    km = km_models[i]
    x = km.cluster_centers_[km.predict(x.flatten().reshape(-1, 1))].reshape(x.shape)
    n_unique = np.unique(x.flatten()).shape[0]
    print(f"Number of unique activations: {n_unique}")
    assert n_unique <= 2**bits

  print(models[-1].evaluate(x, y_test, verbose=2))
  return cb_tables, models, km_models
  

def create_in_out_table(km, quantizer):
  in_table = km.cluster_centers_.flatten()
  out_table = km.predict(quantizer.range().reshape(-1, 1).astype(np.float32)).flatten()   
  return [in_table, out_table]


def codebook_embeddings(embeddings, bits, quantizer, tier=2):
  from tqdm import tqdm
  if tier == 1:
    cluster_embeddings = np.zeros(embeddings.shape)
    for i in tqdm(range(cluster_embeddings.shape[0])):
      km = KMeans(2**bits)
      km.fit(embeddings[i, :].reshape(-1, 1))
      cluster_embeddings[i, :] = km.cluster_centers_[km.predict(embeddings[i,:].reshape(-1, 1))].flatten()
      
    assert np.unique(cluster_embeddings[32]).shape[0] <= 2**bits
    return cluster_embeddings
  else:
    cluster_cls_embeddings = embeddings.copy()
    km1 = KMeans(2**bits)
    km1.fit(embeddings)
    tier1 = km1.predict(embeddings)

    tier2 = [0]*2**bits
    block_sizes = [0]*2**bits
    for block_label in tqdm(range(2**bits)):
      mask = block_label == tier1
      indices = np.arange(cluster_cls_embeddings.shape[0])[mask]
      block = cluster_cls_embeddings[mask]
      
      km2 = KMeans(2**bits)
      km2.fit(block.flatten().reshape(-1, 1))
      tier2[block_label] = km2
      block_sizes[block_label] = block.shape[0]
      
      for i in indices:
        cluster_cls_embeddings[i, :] = km2.cluster_centers_[km2.predict(
                                                      cluster_cls_embeddings[i,:].reshape(-1, 1))].flatten()
        
    assert np.unique(cluster_cls_embeddings[320]).shape[0] <= 2**bits # could be less since it is block based 
    return cluster_cls_embeddings
  


class QCodebookEmbedding(Embedding):
    """
    Post training quantization


    """
    def __init__(self, embedding, n_blocks=None, quantizer=None):
        if n_blocks is not None:
            assert np.mod(np.log2(n_blocks), 1) == 0
            self.n_blocks = n_blocks
        else:
            n_blocks = embedding.weights[0].shape[0]

        self.quantizer = quantizer
        self.quantizer_internal = get_quantizer(self.quantizer)
        self.built = False

        self.weights = embedding.weights
        self.input_dim = embedding.input_dim
        self.output_dim = embedding.output_dim
        self.mask_zero = embedding.mask_zero


    def build(self, input):
        pass

        


 