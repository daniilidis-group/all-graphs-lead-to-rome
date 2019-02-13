# -*- coding: utf-8 -*-
import numpy as np 
import os
import sys
import tensorflow as tf
import sonnet as snt

import tfutils
import myutils
import options

from model import layers

class GraphConvLayerNetwork(snt.AbstractModule):
  """Basic Graph Convolutional Net, no skip connections or group norms"""
  def __init__(self,
               opts,
               arch,
               use_bias=True,
               initializers=None,
               regularizers=None,
               custom_getter=None,
               name="graphnn"):
    """
    Input:
    - opts (options) - object with all relevant options stored
    - arch (ArchParams) - object with all relevant Architecture options
    - use_bias (boolean, optional) - have biases in the network (default True)
    - intializers (dict, optional) - specify custom initializers
    - regularizers (dict, optional) - specify custom regularizers
    - custom_getter (dict, optional) - specify custom getters
    - name (string, optional) - name for module for scoping (default graphnn)
    """
    super(GraphConvLayerNetwork, self).__init__(custom_getter=custom_getter, name=name)
    self._nlayers = len(arch.layer_lens)
    self._layers = [
      layers.GraphConvLayer(
        output_size=layer_len,
        activation=arch.activ,
        initializers=initializers,
        regularizers=regularizers,
        name="{}/graph_conv".format(name))
      for layer_len in arch.layer_lens
    ] + [
      layers.EmbeddingLinearLayer(
        output_size=opts.final_embedding_dim,
        initializers=initializers,
        regularizers=regularizers,
        name="{}/embed_lin".format(name))
    ]
    self.normalize_emb = arch.normalize_emb

  def _build(self, laplacian, init_embeddings):
    """Applying this graph network to sample
    Inputs:
    - laplacian (tf.Tensor) - laplacian for the input graph
    - init_embeddings (tf.Tensor) - Initial node embeddings of the graph
    Outputs: output (tf.Tensor) - the output of the network 
    """
    output = init_embeddings
    for layer in self._layers:
      output = layer(laplacian, output)
    if self.normalize_emb:
      output = tf.nn.l2_normalize(output, axis=2)
    return output

class GraphAttentionLayerNetwork(snt.AbstractModule):
  """Graph Attention Net, derived from https://arxiv.org/abs/1710.10903"""
  def __init__(self,
               opts,
               arch,
               use_bias=True,
               initializers=None,
               regularizers=None,
               custom_getter=None,
               name="graphnn"):
    super(GraphAttentionLayerNetwork, self).__init__(custom_getter=custom_getter,
                                                     name=name)
    """
    Input:
    - opts (options) - object with all relevant options stored
    - arch (ArchParams) - object with all relevant Architecture options
    - use_bias (boolean, optional) - have biases in the network (default True)
    - intializers (dict, optional) - specify custom initializers
    - regularizers (dict, optional) - specify custom regularizers
    - custom_getter (dict, optional) - specify custom getters
    - name (string, optional) - name for module for scoping (default graphnn)
    """
    self._nlayers = len(arch.layer_lens)
    final_regularizers = None
    if regularizers is not None:
      final_regularizers = { k:v
                             for k, v in regularizers.items()
                             if k in ["w", "b"] }
    self._layers = [
      layers.GraphAttentionLayer(
        output_size=layer_len,
        activation=arch.activ,
        sparse=arch.sparse,
        initializers=initializers,
        regularizers=regularizers,
        name="{}/graph_attn".format(name))
      for layer_len in arch.layer_lens
    ] + [
      layers.EmbeddingLinearLayer(
        output_size=opts.final_embedding_dim,
        initializers=initializers,
        regularizers=final_regularizers,
        name="{}/embed_lin".format(name))
    ]
    self.normalize_emb = arch.normalize_emb

  def _build(self, laplacian, init_embeddings):
    """Applying this graph network to sample
    Inputs:
    - laplacian (tf.Tensor) - laplacian for the input graph
    - init_embeddings (tf.Tensor) - Initial node embeddings of the graph
    Outputs: output (tf.Tensor) - the output of the network 
    """
    output = init_embeddings
    for layer in self._layers:
      output = layer(laplacian, output)
    if self.normalize_emb:
      output = tf.nn.l2_normalize(output, axis=2)
    return output


