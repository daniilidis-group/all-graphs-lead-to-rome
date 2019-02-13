"""
Various utility functions related to tensorflow
"""
import tensorflow as tf
import math

def matmul(x,y):
  """Multiplies batch x with a single matrix y
  Input:
  - x (tf.Tensor size BxNxM) - batch of matrices
  - y (tf.Tensor size MxP) - single matrix to multiply with
  Output: z (tf.Tensor size BxNxP) - satisfies z[i] = x[i] * y
  """
  return tf.einsum('bik,kj->bij', x, y)

def batch_matmul(x,y):
  """Multiplies batch x with with batch y
  Input:
  - x (tf.Tensor size BxNxM) - batch of matrices
  - y (tf.Tensor size BxMxP) - batch of matrices to multiply with
  Output: z (tf.Tensor size BxNxP) - satisfies z[i] = x[i] * y[i]
  """
  return tf.einsum('bik,bkj->bij', x, y)

def get_sim(x):
  """Get similarity matrices from batch of embeddings x
  Input: x (tf.Tensor size BxNxM) - batch of matrices, embeddings
  Output: y (tf.Tensor size BxNxN) - satisfies y[i] = x[i] * x[i].T
  """
  x_T = tf.transpose(x, perm=[0, 2, 1])
  return batch_matmul(x, x_T)

def get_tf_activ(activ):
  """Get activation function based on activation string
  Input: activ (string) - string describing activation function (usually relu)
  Output: activ_fn (callable) - callable object that performs the activation
  """
  if activ == 'relu':
    return tf.nn.relu
  elif activ == 'leakyrelu':
    return tf.nn.leaky_relu
  elif activ == 'tanh':
    return tf.nn.tanh
  elif activ == 'elu':
    return tf.nn.elu

def create_linear_initializer(input_size, output_size, dtype=tf.float32):
  """Returns a default initializer for weights of a linear module
  Input:
  - input_size (int) - number of filters in the input matrix
  - output_size (int) - number of filters in the output matrix
  - dtype (dtype, optional) - type of tensor output will be (default tf.float32)
  Output: initializer (callable) - function to initialize weights in network
  """
  stddev = math.sqrt((1.3 * 2.0) / (input_size + output_size))
  return tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)

def create_bias_initializer(unused_in, unused_out, dtype=tf.float32):
  """Returns a default initializer for the biases of a linear/AddBias module
  Input:
  - unused_in (int) - unused
  - unused_out (int) - unused
  - dtype (dtype, optional) - type of tensor output will be (default tf.float32)
  Output: initializer (callable) - function to initialize biases in network
  Bias initializer made to fit interface
  """
  return tf.zeros_initializer(dtype=dtype)

def bce_loss(labels, logits, add_loss=True):
  """Binary cross entropy loss funcion
  Inputs:
  - labels (tf.Tensor) - ground truth labels
  - logits (tf.Tensor) - output of the network in log space same size as labels
  - add_loss (boolean, optional) - add loss to tf.losses or not (default true)
  Output: bce (tf.Tensor) - scalar value of the mean BCE error
  """
  bce_elements = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
  bce_ = tf.reduce_mean(bce_elements)
  if add_loss:
    tf.losses.add_loss(bce_)
  return bce_

def l1_loss(x, y, add_loss=True):
  """L1 loss funcion
  Inputs:
  - x (tf.Tensor) - ground truth labels
  - y (tf.Tensor) - output of the network size as x
  - add_loss (boolean, optional) - add loss to tf.losses or not (default true)
  Output: l1 (tf.Tensor) - scalar value of the mean absolute error
  """
  l1_ = tf.reduce_mean(tf.abs(x - y))
  if add_loss:
    tf.losses.add_loss(l1_)
  return l1_

def l2_loss(x, y, add_loss=True):
  """L2 loss funcion
  Inputs:
  - x (tf.Tensor) - ground truth labels
  - y (tf.Tensor) - output of the network size as x
  - add_loss (boolean, optional) - add loss to tf.losses or not (default true)
  Output: l2 (tf.Tensor) - scalar value of the mean squared error
  """
  l2_ = tf.reduce_mean(tf.square(x - y))
  if add_loss:
    tf.losses.add_loss(l2_)
  return l2_

def l1_l2_loss(x, y, add_loss=True):
  """Addition of L1 and L2 loss funcions
  Inputs:
  - x (tf.Tensor) - ground truth labels
  - y (tf.Tensor) - output of the network size as x
  - add_loss (boolean, optional) - add loss to tf.losses or not (default true)
  Output: l1l2 (tf.Tensor) - scalar value of the loss
  """
  l1_ = tf.reduce_mean(tf.abs(x - y))
  l2_ = tf.reduce_mean(tf.square(x - y))
  l1l2_ = l1_ + l2_
  if add_loss:
    tf.losses.add_loss(l1l2_)
  return l1l2_

