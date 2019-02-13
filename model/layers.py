# -*- coding: utf-8 -*-
import numpy as np 
import os
import sys
import tensorflow as tf
import sonnet as snt

import tfutils
import myutils
import options


class GraphGroupNorm(snt.AbstractModule):
  """Group Norm for Graph-based inputs module 

  Implemented since the tensorflow one does not work with unknown batch size.
  Assumed input dimensions is [ Batch, Nodes, Features ]
  """
  def __init__(self, group_size=32, name='group_norm'):
    super(GraphGroupNorm, self).__init__(custom_getter=None,
                                    name=name)
    self.group_size = 32
    self.possible_keys = self.get_possible_initializer_keys()
    self._initializers = {
      'gamma' : tf.ones_initializer(),
      'beta'  : tf.zeros_initializer()
    }
    self._gamma = None
    self._beta = None
    self._input_shape = None

  def get_possible_initializer_keys(cls, use_bias=True):
    return {"gamma", "beta"}

  @property
  def gamma(self):
    """Returns the Variable containing the scale parameter, gamma.
    Output: gamma (tf.Tensor) - weights, from the most recent __call__.
    Raises:
      snt.NotConnectedError: If the module has not been connected to the
          graph yet, meaning the variables do not exist.
    """
    self._ensure_is_connected()
    return self._gamma

  @property
  def beta(self):
    """Returns the Variable containing the center parameter, beta.
    Output: beta (tf.Tensor) - biases, from the most recent __call__.
    Raises:
      snt.NotConnectedError: If the module has not been connected to the
          graph yet, meaning the variables do not exist.
    """
    self._ensure_is_connected()
    return self._beta

  @property
  def initializers(self):
    """Returns the initializers dictionary."""
    return self._initializers

  @property
  def partitioners(self):
    """Returns the partitioners dictionary."""
    return self._partitioners

  def clone(self, name=None):
    """Returns a cloned `GraphGroupNorm` module.
    Input:
    - name (string, optional) - name of cloned module. The default name
          is constructed by appending "_clone" to `self.module_name`.
    Output: net (snt.Module) - Cloned `GraphGroupNorm` module.
    """
    if name is None:
      name = self.module_name + "_clone"
    return GraphGroupNorm(group_size=self.group_size)

  def _build(self, inputs):
    # Based on: 
    # https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/layers/python/layers/normalization.py
    input_shape = tuple(inputs.get_shape().as_list())
    if len(input_shape) != 3:
      raise snt.IncompatibleShapeError(
          "{}: rank of shape must be 3 not: {}".format(
              self.scope_name, len(input_shape)))

    if input_shape[2] is None:
      raise snt.IncompatibleShapeError(
          "{}: Input size must be specified at module build time".format(
              self.scope_name))
    self._input_shape = input_shape
    dtype = inputs.dtype
    group_sizes = [ self.group_size, self._input_shape[2] // self.group_size ]
    broadcast_shape = [ 1, 1 ] + group_sizes
    self._gamma = tf.get_variable("gamma",
                                  shape=(self._input_shape[2]),
                                  dtype=dtype,
                                  initializer=self._initializers["gamma"])
    if self._gamma not in tf.get_collection('weights'):
      tf.add_to_collection('weights', self._gamma)
    self._gamma = tf.reshape(self._gamma, broadcast_shape)

    self._beta = tf.get_variable("beta",
                                 shape=(self._input_shape[2],),
                                 dtype=dtype,
                                 initializer=self._initializers["beta"])
    if self._beta not in tf.get_collection('biases'):
      tf.add_to_collection('biases', self._beta)
    self._beta = tf.reshape(self._beta, broadcast_shape)

    ##### Actually perform operations
    # Reshape input
    original_shape = [ -1, self._input_shape[1], self._input_shape[2] ]
    inputs_shape = [ -1, self._input_shape[1] ] + group_sizes
                     
    inputs = tf.reshape(inputs, inputs_shape)

    # Normalize
    mean, variance = tf.nn.moments(inputs, [1, 3], keep_dims=True)
    gain = tf.rsqrt(variance + 1e-7) * self._gamma
    offset = -mean * gain + self._beta
    outputs = inputs * gain + offset

    # Reshape back to output
    outputs = tf.reshape(outputs, original_shape)

    return outputs

class AbstractGraphLayer(snt.AbstractModule):
  """Transformation on an graphe node embedding.
  
  This functions almost exactly like snt.Linear except it is for tensors of
  size batch_size x nodes x input_size. Acts by matrix multiplication on the
  left side of each nodes x input_size matrix.
  """
  def __init__(self,
               output_size,
               use_bias=True,
               initializers=None,
               partitioners=None,
               regularizers=None,
               custom_getter=None,
               name="embed_lin"):
    super(AbstractGraphLayer, self).__init__(custom_getter=None, name=name)
    self._output_size = output_size
    self._use_bias = use_bias
    self._input_shape = None
    self.possible_keys = self.get_possible_initializer_keys(use_bias=use_bias)
    self._initializers = snt.check_initializers(
        initializers, self.possible_keys)
    self._partitioners = snt.check_partitioners(
        partitioners, self.possible_keys)
    self._regularizers = snt.check_regularizers(
        regularizers, self.possible_keys)

  @classmethod
  def get_possible_initializer_keys(cls, use_bias=True):
    raise NotImplemented("Need to overwrite in subclass")

  def _build(self, laplacian, inputs):
    return inputs

  @property
  def output_size(self):
    """Returns the module output size."""
    if callable(self._output_size):
      self._output_size = self._output_size()
    return self._output_size

  @property
  def has_bias(self):
    """Returns `True` if bias Variable is present in the module."""
    return self._use_bias

  @property
  def initializers(self):
    """Returns the initializers dictionary."""
    return self._initializers

  @property
  def partitioners(self):
    """Returns the partitioners dictionary."""
    return self._partitioners

  @property
  def regularizers(self):
    """Returns the regularizers dictionary."""
    return self._regularizers

  def clone(self, name=None):
    """Returns a cloned `AbstractGraphLayer` module.
    Input:
    - name (string, optional) - name of cloned module. The default name
          is constructed by appending "_clone" to `self.module_name`.
    Output: net (snt.Module) - Cloned `AbstractGraphLayer` module.
    """
    if name is None:
      name = self.module_name + "_clone"
    return AbstractGraphLayer(output_size=self.output_size,
                              use_bias=self._use_bias,
                              initializers=self._initializers,
                              partitioners=self._partitioners,
                              regularizers=self._regularizers,
                              name=name)



class EmbeddingLinearLayer(AbstractGraphLayer):
  """Linear transformation on an embedding, each independently.
  
  This functions almost exactly like snt.Linear except it is for tensors of
  size batch_size x nodes x input_size. Acts by matrix multiplication on the
  left side of each nodes x input_size matrix.
  """
  def __init__(self,
               output_size,
               use_bias=True,
               initializers=None,
               partitioners=None,
               regularizers=None,
               custom_getter=None,
               name="embed_lin"):
    super(EmbeddingLinearLayer, self).__init__(
                 output_size,
                 use_bias=use_bias,
                 initializers=initializers,
                 partitioners=partitioners,
                 regularizers=regularizers,
                 custom_getter=custom_getter,
                 name=name)
    self._w = None
    self._b = None
    self.possible_keys = self.get_possible_initializer_keys(use_bias=use_bias)

  @classmethod
  def get_possible_initializer_keys(cls, use_bias=True):
    return {"w", "b"} if use_bias else {"w"}

  def _build(self, laplacian, inputs):
    input_shape = tuple(inputs.get_shape().as_list())
    if len(input_shape) != 3:
      raise snt.IncompatibleShapeError(
          "{}: rank of shape must be 3 not: {}".format(
              self.scope_name, len(input_shape)))

    if input_shape[2] is None:
      raise snt.IncompatibleShapeError(
          "{}: Input size must be specified at module build time".format(
              self.scope_name))

    if self._input_shape is not None and input_shape[2] != self._input_shape[2]:
      raise snt.IncompatibleShapeError(
          "{}: Input shape must be [batch_size, {}, {}] not: [batch_size, {}, {}]"
          .format(self.scope_name,
                  input_shape[2],
                  self._input_shape[2],
                  input_shape[1],
                  input_shape[2]))

    self._input_shape = input_shape
    dtype = inputs.dtype

    if "w" not in self._initializers:
      self._initializers["w"] = tfutils.create_linear_initializer(
                                          self._input_shape[2],
                                          self._output_size,
                                          dtype)

    if "b" not in self._initializers and self._use_bias:
      self._initializers["b"] = tfutils.create_bias_initializer(
                                          self._input_shape[2],
                                          self._output_size,
                                          dtype)

    weight_shape = (self._input_shape[2], self.output_size)
    self._w = tf.get_variable("w",
                              shape=weight_shape,
                              dtype=dtype,
                              initializer=self._initializers["w"],
                              partitioner=self._partitioners.get("w", None),
                              regularizer=self._regularizers.get("w", None))
    if self._w not in tf.get_collection('weights'):
      tf.add_to_collection('weights', self._w)
    outputs = tfutils.matmul(inputs, self._w)

    if self._use_bias:
      bias_shape = (self.output_size,)
      self._b = tf.get_variable("b",
                                shape=bias_shape,
                                dtype=dtype,
                                initializer=self._initializers["b"],
                                partitioner=self._partitioners.get("b", None),
                                regularizer=self._regularizers.get("b", None))
      if self._b not in tf.get_collection('biases'):
        tf.add_to_collection('biases', self._b)
      outputs += self._b

    return outputs

  @property
  def w(self):
    """Returns the Variable containing the weight parameters.
    Output: w (tf.Tensor) - weights, from the most recent __call__.
    Raises:
      snt.NotConnectedError: If the module has not been connected to the
          graph yet, meaning the variables do not exist.
    """
    self._ensure_is_connected()
    return self._w

  @property
  def b(self):
    """Returns the Variable containing the bias parameters.
    Output: b (tf.Tensor) - biases, from the most recent __call__.
    Raises:
      snt.NotConnectedError: If the module has not been connected to the
          graph yet, meaning the variables do not exist.
      AttributeError: If the module does not use bias.
    """
    self._ensure_is_connected()
    if not self._use_bias:
      raise AttributeError(
          "No bias Variable in Linear Module when `use_bias=False`.")
    return self._b

  def clone(self, name=None):
    """Returns a cloned `EmbeddingLinearLayer` module.
    Input:
    - name (string, optional) - name of cloned module. The default name
          is constructed by appending "_clone" to `self.module_name`.
    Output: net (snt.Module) - Cloned `EmbeddingLinearLayer` module.
    """
    if name is None:
      name = self.module_name + "_clone"
    return EmbeddingRightLinear(output_size=self.output_size,
                                use_bias=self._use_bias,
                                initializers=self._initializers,
                                partitioners=self._partitioners,
                                regularizers=self._regularizers,
                                name=name)

class GraphConvLayer(AbstractGraphLayer):
  """Linear transformation on an embedding, each independently.
  
  This functions almost exactly like snt.Linear except it is for tensors of
  size batch_size x nodes x input_size. Acts by matrix multiplication on the
  left side of each nodes x input_size matrix.
  """
  def __init__(self,
               output_size,
               activation='relu',
               use_bias=True,
               initializers=None,
               partitioners=None,
               regularizers=None,
               custom_getter=None,
               name="graph_conv"):
    super(GraphConvLayer, self).__init__(
                 output_size,
                 use_bias=use_bias,
                 initializers=initializers,
                 partitioners=partitioners,
                 regularizers=regularizers,
                 custom_getter=custom_getter,
                 name=name)
    self._activ = tfutils.get_tf_activ(activation)
    self._w = None
    self._b = None
    self.possible_keys = self.get_possible_initializer_keys(use_bias=use_bias)

  @classmethod
  def get_possible_initializer_keys(cls, use_bias=True):
    return {"w", "b"} if use_bias else {"w"}

  def _build(self, laplacian, inputs):
    input_shape = tuple(inputs.get_shape().as_list())
    if len(input_shape) != 3:
      raise snt.IncompatibleShapeError(
          "{}: rank of shape must be 3 not: {}".format(
              self.scope_name, len(input_shape)))

    if input_shape[2] is None:
      raise snt.IncompatibleShapeError(
          "{}: Input size must be specified at module build time".format(
              self.scope_name))

    if input_shape[1] is None:
      raise snt.IncompatibleShapeError(
          "{}: Number of nodes must be specified at module build time".format(
              self.scope_name))

    if self._input_shape is not None and \
        (input_shape[2] != self._input_shape[2] or \
         input_shape[1] != self._input_shape[1]):
      raise snt.IncompatibleShapeError(
          "{}: Input shape must be [batch_size, {}, {}] not: [batch_size, {}, {}]"
          .format(self.scope_name,
                  self._input_shape[1],
                  self._input_shape[2],
                  input_shape[1],
                  input_shape[2]))


    self._input_shape = input_shape
    dtype = inputs.dtype

    if "w" not in self._initializers:
      self._initializers["w"] = tfutils.create_linear_initializer(
                                          self._input_shape[2],
                                          self._output_size,
                                          dtype)

    if "b" not in self._initializers and self._use_bias:
      self._initializers["b"] = tfutils.create_bias_initializer(
                                          self._input_shape[2],
                                          self._output_size,
                                          dtype)

    weight_shape = (self._input_shape[2], self.output_size)
    self._w = tf.get_variable("w",
                              shape=weight_shape,
                              dtype=dtype,
                              initializer=self._initializers["w"],
                              partitioner=self._partitioners.get("w", None),
                              regularizer=self._regularizers.get("w", None))
    if self._w not in tf.get_collection('weights'):
      tf.add_to_collection('weights', self._w)
    outputs_ = tfutils.matmul(inputs, self._w)
    outputs = tfutils.batch_matmul(laplacian, outputs_)

    if self._use_bias:
      bias_shape = (self.output_size,)
      self._b = tf.get_variable("b",
                                shape=bias_shape,
                                dtype=dtype,
                                initializer=self._initializers["b"],
                                partitioner=self._partitioners.get("b", None),
                                regularizer=self._regularizers.get("b", None))
      if self._b not in tf.get_collection('biases'):
        tf.add_to_collection('biases', self._b)
      outputs += self._b


    return self._activ(outputs)

  @property
  def w(self):
    """Returns the Variable containing the weight parameters.
    Output: w (tf.Tensor) - weights, from the most recent __call__.
    Raises:
      snt.NotConnectedError: If the module has not been connected to the
          graph yet, meaning the variables do not exist.
    """
    self._ensure_is_connected()
    return self._w

  @property
  def b(self):
    """Returns the Variable containing the bias parameters.
    Output: b (tf.Tensor) - biases, from the most recent __call__.
    Raises:
      snt.NotConnectedError: If the module has not been connected to the
          graph yet, meaning the variables do not exist.
      AttributeError: If the module does not use bias.
    """
    self._ensure_is_connected()
    if not self._use_bias:
      raise AttributeError(
          "No bias Variable in Linear Module when `use_bias=False`.")
    return self._b

  def clone(self, name=None):
    """Returns a cloned `GraphConvLayer` module.
    Input:
    - name (string, optional) - name of cloned module. The default name
          is constructed by appending "_clone" to `self.module_name`.
    Output: net (snt.Module) - Cloned `GraphConvLayer` module.
    """
    if name is None:
      name = self.module_name + "_clone"
    return GraphConvLayer(output_size=self.output_size,
                           use_bias=self._use_bias,
                           initializers=self._initializers,
                           partitioners=self._partitioners,
                           regularizers=self._regularizers,
                           name=name)

class GraphSkipLayer(AbstractGraphLayer):
  """Linear transformation on an embedding, each independently.
  
  This functions almost exactly like snt.Linear except it is for tensors of
  size batch_size x nodes x input_size. Acts by matrix multiplication on the
  left side of each nodes x input_size matrix.
  """
  def __init__(self,
               output_size,
               activation='relu',
               use_bias=True,
               initializers=None,
               partitioners=None,
               regularizers=None,
               custom_getter=None,
               name="graph_skip"):
    super(GraphSkipLayer, self).__init__(
                 output_size,
                 use_bias=use_bias,
                 initializers=initializers,
                 partitioners=partitioners,
                 regularizers=regularizers,
                 custom_getter=custom_getter,
                 name=name)
    self._activ = tfutils.get_tf_activ(activation)
    self._w = None
    self._u = None
    self._b = None
    self._c = None
    self.possible_keys = self.get_possible_initializer_keys(use_bias=use_bias)

  @classmethod
  def get_possible_initializer_keys(cls, use_bias=True):
    return {"w", "u", "b", "c"} if use_bias else {"w", "u"}

  def _build(self, laplacian, inputs):
    input_shape = tuple(inputs.get_shape().as_list())
    if len(input_shape) != 3:
      raise snt.IncompatibleShapeError(
          "{}: rank of shape must be 3 not: {}".format(
              self.scope_name, len(input_shape)))

    if input_shape[2] is None:
      raise snt.IncompatibleShapeError(
          "{}: Input size must be specified at module build time".format(
              self.scope_name))

    if input_shape[1] is None:
      raise snt.IncompatibleShapeError(
          "{}: Number of nodes must be specified at module build time".format(
              self.scope_name))

    if self._input_shape is not None and \
        (input_shape[2] != self._input_shape[2] or \
         input_shape[1] != self._input_shape[1]):
      raise snt.IncompatibleShapeError(
          "{}: Input shape must be [batch_size, {}, {}] not: [batch_size, {}, {}]"
          .format(self.scope_name,
                  self._input_shape[1],
                  self._input_shape[2],
                  input_shape[1],
                  input_shape[2]))


    self._input_shape = input_shape
    dtype = inputs.dtype

    if "w" not in self._initializers:
      self._initializers["w"] = tfutils.create_linear_initializer(
                                          self._input_shape[2],
                                          self._output_size,
                                          dtype)
    if "u" not in self._initializers:
      self._initializers["u"] = tfutils.create_linear_initializer(
                                          self._input_shape[2],
                                          self._output_size,
                                          dtype)

    if "b" not in self._initializers and self._use_bias:
      self._initializers["b"] = tfutils.create_bias_initializer(
                                          self._input_shape[2],
                                          self._output_size,
                                          dtype)
    if "c" not in self._initializers and self._use_bias:
      self._initializers["c"] = tfutils.create_bias_initializer(
                                          self._input_shape[2],
                                          self._output_size,
                                          dtype)

    weight_shape = (self._input_shape[2], self.output_size)
    self._w = tf.get_variable("w",
                              shape=weight_shape,
                              dtype=dtype,
                              initializer=self._initializers["w"],
                              partitioner=self._partitioners.get("w", None),
                              regularizer=self._regularizers.get("w", None))
    if self._w not in tf.get_collection('weights'):
      tf.add_to_collection('weights', self._w)
    self._u = tf.get_variable("u",
                              shape=weight_shape,
                              dtype=dtype,
                              initializer=self._initializers["u"],
                              partitioner=self._partitioners.get("u", None),
                              regularizer=self._regularizers.get("u", None))
    if self._u not in tf.get_collection('weights'):
      tf.add_to_collection('weights', self._u)
    preactiv_ = tfutils.matmul(inputs, self._w)
    preactiv = tfutils.batch_matmul(laplacian, preactiv_)
    skip = tfutils.matmul(inputs, self._u)

    if self._use_bias:
      bias_shape = (self.output_size,)
      self._b = tf.get_variable("b",
                                shape=bias_shape,
                                dtype=dtype,
                                initializer=self._initializers["b"],
                                partitioner=self._partitioners.get("b", None),
                                regularizer=self._regularizers.get("b", None))
      if self._b not in tf.get_collection('biases'):
        tf.add_to_collection('biases', self._b)
      self._c = tf.get_variable("c",
                                shape=bias_shape,
                                dtype=dtype,
                                initializer=self._initializers["c"],
                                partitioner=self._partitioners.get("c", None),
                                regularizer=self._regularizers.get("c", None))
      if self._c not in tf.get_collection('biases'):
        tf.add_to_collection('biases', self._c)
      preactiv += self._b
      skip += self._c

    activ = self._activ(preactiv) + skip

    return activ

  @property
  def w(self):
    """Returns the Variable containing the weight parameters.
    Output: w (tf.Tensor) - weights, from the most recent __call__.
    Raises:
      snt.NotConnectedError: If the module has not been connected to the
          graph yet, meaning the variables do not exist.
    """
    self._ensure_is_connected()
    return self._w

  @property
  def u(self):
    """Returns the Variable containing the skip weights parameters.
    Output: u (tf.Tensor) - skip weights, from the most recent __call__.
    Raises:
      snt.NotConnectedError: If the module has not been connected to the
          graph yet, meaning the variables do not exist.
    """
    self._ensure_is_connected()
    return self._u

  @property
  def b(self):
    """Returns the Variable containing the bias parameters.
    Output: b (tf.Tensor) - biases, from the most recent __call__.
    Raises:
      snt.NotConnectedError: If the module has not been connected to the
          graph yet, meaning the variables do not exist.
      AttributeError: If the module does not use bias.
    """
    self._ensure_is_connected()
    if not self._use_bias:
      raise AttributeError(
          "No bias Variable in Linear Module when `use_bias=False`.")
    return self._b

  @property
  def c(self):
    """Returns the Variable containing the skip bias parameters.
    Output: c (tf.Tensor) - skip biases, from the most recent __call__.
    Raises:
      snt.NotConnectedError: If the module has not been connected to the
          graph yet, meaning the variables do not exist.
      AttributeError: If the module does not use bias.
    """
    self._ensure_is_connected()
    if not self._use_bias:
      raise AttributeError(
          "No bias Variable in Linear Module when `use_bias=False`.")
    return self._c

  def clone(self, name=None):
    """Returns a cloned `GraphSkipLayer` module.
    Input:
    - name (string, optional) - name of cloned module. The default name
          is constructed by appending "_clone" to `self.module_name`.
    Output: net (snt.Module) - Cloned `GraphSkipLayer` module.
    """
    if name is None:
      name = self.module_name + "_clone"
    return GraphSkipLayer(output_size=self.output_size,
                           use_bias=self._use_bias,
                           initializers=self._initializers,
                           partitioners=self._partitioners,
                           regularizers=self._regularizers,
                           name=name)

class GraphAttentionLayer(AbstractGraphLayer):
  """Linear transformation on an embedding, each independently.
  
  This functions almost exactly like snt.Linear except it is for tensors of
  size batch_size x nodes x input_size. Acts by matrix multiplication on the
  left side of each nodes x input_size matrix.
  """
  def __init__(self,
               output_size,
               activation='relu',
               attn_activation='leakyrelu',
               use_bias=True,
               sparse=False,
               initializers=None,
               partitioners=None,
               regularizers=None,
               custom_getter=None,
               name="graph_attn"):
    super(GraphAttentionLayer, self).__init__(
                 output_size,
                 use_bias=use_bias,
                 initializers=initializers,
                 partitioners=partitioners,
                 regularizers=regularizers,
                 custom_getter=custom_getter,
                 name=name)
    self._sparse = sparse
    self._activ = tfutils.get_tf_activ(activation)
    self._attn_activ = tfutils.get_tf_activ(attn_activation)
    self.weight_keys = { ("w", output_size), ("u", output_size),
                         ("f1", 1), ("f2", 1) }
    self.bias_keys = set()
    self.weights = { x[0] : None for x in self.weight_keys }
    if use_bias:
      self.bias_keys = { ("b", output_size), ("c", output_size),
                         ("d1", 1), ("d2", 1) }
      for x in self.bias_keys:
        self.weights[x[0]] = None
    self.possible_keys = self.get_possible_initializer_keys(use_bias=use_bias)

  @classmethod
  def get_possible_initializer_keys(cls, use_bias=True):
    if use_bias:
      return {"w", "u", "b", "c", "f1", "f2", "d1", "d2"}
    else:
      return {"w", "u", "f1", "f2"}

  def _build(self, laplacian, inputs):
    input_shape = tuple(inputs.get_shape().as_list())
    if len(input_shape) != 3:
      raise snt.IncompatibleShapeError(
          "{}: rank of shape must be 3 not: {}".format(
              self.scope_name, len(input_shape)))

    if input_shape[2] is None:
      raise snt.IncompatibleShapeError(
          "{}: Input size must be specified at module build time".format(
              self.scope_name))

    if input_shape[1] is None:
      raise snt.IncompatibleShapeError(
          "{}: Number of nodes must be specified at module build time".format(
              self.scope_name))

    if self._input_shape is not None and \
        (input_shape[2] != self._input_shape[2] or \
         input_shape[1] != self._input_shape[1]):
      raise snt.IncompatibleShapeError(
          "{}: Input shape must be [batch_size, {}, {}] not: [batch_size, {}, {}]"
          .format(self.scope_name,
                  self._input_shape[1],
                  self._input_shape[2],
                  input_shape[1],
                  input_shape[2]))


    self._input_shape = input_shape
    dtype = inputs.dtype

    for k, s in self.weight_keys:
      if k not in self._initializers:
        self._initializers[k] = tfutils.create_linear_initializer(
                                            self._input_shape[2], s, dtype)

    if self._use_bias:
      for k, s in self.bias_keys:
        if k not in self._initializers:
          self._initializers[k] = tfutils.create_bias_initializer(
                                              self._input_shape[2], s, dtype)

    for k, s in self.weight_keys:
      weight_shape = (self._input_shape[2], s)
      self.weights[k] = tf.get_variable(
              k,
              shape=weight_shape,
              dtype=dtype,
              initializer=self._initializers[k],
              partitioner=self._partitioners.get(k, None),
              regularizer=self._regularizers.get(k, None))
      if self.weights[k] not in tf.get_collection('weights'):
        tf.add_to_collection('weights', self.weights[k])

    if self._use_bias:
      for k, s in self.bias_keys:
        bias_shape = (s,)
        self.weights[k] = tf.get_variable(
                k,
                shape=bias_shape,
                dtype=dtype,
                initializer=self._initializers[k],
                partitioner=self._partitioners.get(k, None),
                regularizer=self._regularizers.get(k, None))
      if self.weights[k] not in tf.get_collection('biases'):
        tf.add_to_collection('biases', self.weights[k])

    preactiv_ = tfutils.matmul(inputs, self.weights["w"])
    f1_ = tfutils.matmul(inputs, self.weights["f1"])
    f2_ = tfutils.matmul(inputs, self.weights["f2"])
    if self._use_bias:
      f1_ += self.weights["d1"]
      f2_ += self.weights["d2"]
    preattn_mat_ = f1_ + tf.transpose(f2_, [0, 2, 1])
    if self._sparse:
      preattn_mat = self._attn_activ(preattn_mat_) * laplacian
    else:
      preattn_mat = self._attn_activ(preattn_mat_) + laplacian
    attn_mat = tf.nn.softmax(preattn_mat, axis=-1)
    preactiv = tfutils.batch_matmul(attn_mat, preactiv_)
    skip = tfutils.matmul(inputs, self.weights["u"])

    if self._use_bias:
      preactiv += self.weights["b"]
      skip += self.weights["c"]

    activ = self._activ(preactiv) + skip

    return activ

  @property
  def w(self):
    """Returns the Variable containing the weight parameters.
    Output: w (tf.Tensor) - weights, from the most recent __call__.
    Raises:
      snt.NotConnectedError: If the module has not been connected to the
          graph yet, meaning the variables do not exist.
    """
    self._ensure_is_connected()
    return self.weights["w"]

  @property
  def u(self):
    """Returns the Variable containing the skip weight parameters.
    Output: u (tf.Tensor) - skip weights, from the most recent __call__.
    Raises:
      snt.NotConnectedError: If the module has not been connected to the
          graph yet, meaning the variables do not exist.
    """
    self._ensure_is_connected()
    return self.weights["u"]

  @property
  def f1(self):
    """Returns the Variable containing the first attention weight parameters.
    Output: f1 (tf.Tensor) - attention weights, from the most recent __call__.
    Raises:
      snt.NotConnectedError: If the module has not been connected to the
          graph yet, meaning the variables do not exist.
    """
    self._ensure_is_connected()
    return self.weights["f1"]

  @property
  def f2(self):
    """Returns the Variable containing the second attention weight parameters.
    Output: f2 (tf.Tensor) - attention weights, from the most recent __call__.
    Raises:
      snt.NotConnectedError: If the module has not been connected to the
          graph yet, meaning the variables do not exist.
    """
    self._ensure_is_connected()
    return self.weights["f2"]

  @property
  def b(self):
    """Returns the Variable containing the bias parameters.
    Output: b (tf.Tensor) - biases, from the most recent __call__.
    Raises:
      snt.NotConnectedError: If the module has not been connected to the
          graph yet, meaning the variables do not exist.
      AttributeError: If the module does not use bias.
    """
    self._ensure_is_connected()
    if not self._use_bias:
      raise AttributeError(
          "No bias Variable in Linear Module when `use_bias=False`.")
    return self.weights["b"]

  @property
  def c(self):
    """Returns the Variable containing the skip bias parameters.
    Output: b (tf.Tensor) - skip biases, from the most recent __call__.
    Raises:
      snt.NotConnectedError: If the module has not been connected to the
          graph yet, meaning the variables do not exist.
      AttributeError: If the module does not use bias.
    """
    self._ensure_is_connected()
    if not self._use_bias:
      raise AttributeError(
          "No bias Variable in Linear Module when `use_bias=False`.")
    return self.weights["c"]

  @property
  def d1(self):
    """Returns the Variable containing first attention bias.
    Output: d1 (tf.Tensor) - attention biases, from the most recent __call__.
    Raises:
      snt.NotConnectedError: If the module has not been connected to the
          graph yet, meaning the variables do not exist.
      AttributeError: If the module does not use bias.
    """
    self._ensure_is_connected()
    if not self._use_bias:
      raise AttributeError(
          "No bias Variable in Linear Module when `use_bias=False`.")
    return self.weights["d1"]

  @property
  def d2(self):
    """Returns the Variable containing second attention bias.
    Output: d2 (tf.Tensor) - attention biases, from the most recent __call__.
    Raises:
      snt.NotConnectedError: If the module has not been connected to the
          graph yet, meaning the variables do not exist.
      AttributeError: If the module does not use bias.
    """
    self._ensure_is_connected()
    if not self._use_bias:
      raise AttributeError(
          "No bias Variable in Linear Module when `use_bias=False`.")
    return self.weights["d2"]

  def clone(self, name=None):
    """Returns a cloned `GraphAttentionLayer` module.
    Input:
    - name (string, optional) - name of cloned module. The default name
          is constructed by appending "_clone" to `self.module_name`.
    Output: net (snt.Module) - Cloned `GraphAttentionLayer` module.
    """
    if name is None:
      name = self.module_name + "_clone"
    return GraphAttentionLayer(output_size=self.output_size,
                               use_bias=self._use_bias,
                               initializers=self._initializers,
                               partitioners=self._partitioners,
                               regularizers=self._regularizers,
                               name=name)


if __name__ == "__main__":
  pass


