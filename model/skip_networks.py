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

class GraphSkipLayerNetwork(snt.AbstractModule):
  """Graph Convolutional Net with short skip connections
  
  Small skip connections are skip connections from one layer to the next.
  """
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
    super(GraphSkipLayerNetwork, self).__init__(custom_getter=custom_getter, name=name)
    self._nlayers = len(arch.layer_lens)
    final_regularizers = None
    if regularizers is not None:
      final_regularizers = { k:v
                             for k, v in regularizers.items()
                             if k in ["w", "b"] }
    self._layers = [
      layers.GraphSkipLayer(
        output_size=layer_len,
        activation=arch.activ,
        initializers=initializers,
        regularizers=regularizers,
        name="{}/graph_skip".format(name))
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

class GraphLongSkipLayerNetwork(snt.AbstractModule):
  """Graph Convolutional Net with short and long skip connections
  
  Long skip connections are skip connections from the start to an intermediate
  layer. This combined with short skip connections make training much smoother.
  """
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
    super(GraphLongSkipLayerNetwork, self).__init__(custom_getter=custom_getter,
                                                    name=name)
    self._nlayers = len(arch.layer_lens)
    final_regularizers = None
    if regularizers is not None:
      lin_regularizers = { k:v
                           for k, v in regularizers.items()
                           if k in ["w", "b"] }
    else:
      lin_regularizers = None
    self._layers = [
      layers.GraphSkipLayer(
        output_size=layer_len,
        activation=arch.activ,
        initializers=initializers,
        regularizers=regularizers,
        name="{}/graph_skip".format(name))
      for layer_len in arch.layer_lens
    ] + [
      layers.EmbeddingLinearLayer(
        output_size=opts.final_embedding_dim,
        initializers=initializers,
        regularizers=lin_regularizers,
        name="{}/embed_lin".format(name))
    ]
    self._skip_layer_idx = arch.skip_layers
    self._skip_layers = [
      layers.EmbeddingLinearLayer(
        output_size=arch.layer_lens[skip_idx],
        initializers=initializers,
        regularizers=lin_regularizers,
        name="{}/skip".format(name))
      for skip_idx in self._skip_layer_idx
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
    sk = 0
    for i, layer in enumerate(self._layers):
      if i in self._skip_layer_idx:
        output = layer(laplacian, output) + self._skip_layers[sk](laplacian, output)
        sk += 1
      else:
        output = layer(laplacian, output)
    if self.normalize_emb:
      output = tf.nn.l2_normalize(output, axis=2)
    return output

class GraphLongSkipNormedNetwork(GraphLongSkipLayerNetwork):
  """Graph Convolutional Net with skip connections and group norm
  
  Group norm is an alternative to batch norm, defined here:
  https://arxiv.org/abs/1803.08494
  """
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
    super(GraphLongSkipNormedNetwork, self).__init__(opts, arch,
                                                     use_bias=use_bias,
                                                     initializers=initializers,
                                                     regularizers=regularizers,
                                                     custom_getter=custom_getter,
                                                     name=name)
    self.start_normed = arch.start_normed
    self.group_size = arch.group_size
    self._group_norm_layers = [
      layers.GraphGroupNorm(
        group_size=32,
        name="{}/group_norm".format(name))
      for _ in range(self.start_normed, self._nlayers)
    ]

  def _build(self, laplacian, init_embeddings):
    """Applying this graph network to sample
    Inputs:
    - laplacian (tf.Tensor) - laplacian for the input graph
    - init_embeddings (tf.Tensor) - Initial node embeddings of the graph
    Outputs: output (tf.Tensor) - the output of the network 
    """
    output = init_embeddings
    sk = 0
    for i, layer in enumerate(self._layers):
      if i in self._skip_layer_idx:
        output = layer(laplacian, output) + self._skip_layers[sk](laplacian, init_embeddings)
        sk += 1
      else:
        output = layer(laplacian, output)
      if self.start_normed <= i < self._nlayers:
        output = self._group_norm_layers[i-self.start_normed](output)
    if self.normalize_emb:
      output = tf.nn.l2_normalize(output, axis=2)
    return output

class GraphSkipHopNormedNetwork(GraphLongSkipNormedNetwork):
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
    super(GraphSkipHopNormedNetwork, self).__init__(opts, arch,
                                                     use_bias=use_bias,
                                                     initializers=initializers,
                                                     regularizers=regularizers,
                                                     custom_getter=custom_getter,
                                                     name=name)
    lin_regularizers = None
    if regularizers is not None:
      lin_regularizers = { k:v
                           for k, v in regularizers.items()
                           if k in ["w", "b"] }
    self._hop_layers = [
      layers.EmbeddingLinearLayer(
        output_size=arch.layer_lens[skip_idx],
        initializers=initializers,
        regularizers=lin_regularizers,
        name="{}/hop".format(name))
      for skip_idx in self._skip_layer_idx[1:]
    ]

  def _build(self, laplacian, init_embeddings):
    """Applying this graph network to sample
    Inputs:
    - laplacian (tf.Tensor) - laplacian for the input graph
    - init_embeddings (tf.Tensor) - Initial node embeddings of the graph
    Outputs: output (tf.Tensor) - the output of the network 
    """
    output = init_embeddings
    sk = 0
    last_skip = None
    for i, layer in enumerate(self._layers):
      if i in self._skip_layer_idx:
        output = layer(laplacian, output)
        skip_add = self._skip_layers[sk](laplacian, init_embeddings)
        output = output + skip_add
        if last_skip is not None:
          hop_add = self._hop_layers[sk-1](laplacian, last_skip)
          output = output + hop_add
        last_skip = output
        sk += 1
      else:
        output = layer(laplacian, output)
      if self.start_normed <= i < self._nlayers:
        output = self._group_norm_layers[i-self.start_normed](output)
    if self.normalize_emb:
      output = tf.nn.l2_normalize(output, axis=2)
    return output


