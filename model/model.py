# -*- coding: utf-8 -*-
import numpy as np 
import os
import sys
import tensorflow as tf
import sonnet as snt

import tfutils
import myutils
import options

from model import networks
from model import skip_networks

def get_regularizers(opts):
  """Get regularizers for the weights and biases for Sonnet networks
  Input: opts (options) - object with all relevant options
  Output: regularizers (dict) - regulraizing functions with right keys
  """
  regularizer_fn = None
  bias_fn = tf.contrib.layers.l2_regularizer(0.0)
  if opts.weight_decay <= 0 and opts.weight_l1_decay <= 0:
    return None
  elif opts.weight_decay > 0 and opts.weight_l1_decay <= 0:
    regularizer_fn = \
        lambda r_l2, r_l1: tf.contrib.layers.l2_regularizer(r_l2)
  elif opts.weight_decay <= 0 and opts.weight_l1_decay > 0:
    regularizer_fn = \
        lambda r_l2, r_l1: tf.contrib.layers.l1_regularizer(r_l1)
  elif opts.weight_decay <= 0 and opts.weight_l1_decay > 0:
    regularizer_fn = \
        lambda r_l2, r_l1: tf.contrib.layers.l1_l2_regularizer(r_l1/r_l2, 1.0)
  all_regs = {
          "w" : regularizer_fn(opts.weight_decay, opts.weight_l1_decay), 
          "u" : regularizer_fn(opts.weight_decay, opts.weight_l1_decay), 
          "f1" : regularizer_fn(opts.weight_decay, opts.weight_l1_decay), 
          "f2" : regularizer_fn(opts.weight_decay, opts.weight_l1_decay), 
          "b" :  bias_fn,
          "c" :  bias_fn,
          "d1" : bias_fn, 
          "d2" : bias_fn, 
      }
  if opts.architecture in ['vanilla', 'vanilla0', 'vanilla1']:
    return { k: all_regs[k] for k in [ "w", "b" ] }
  elif opts.architecture in ['skip', 'skip0', 'skip1', \
                             'longskip0', 'longskip1', \
                             'normedskip0', 'normedskip1', \
                             'normedskip2', 'normedskip3', ]:
    return { k: all_regs[k] for k in [ "w", "u", "b", "c" ] }
  elif opts.architecture in ['attn0', 'attn1', 'attn2', \
                             'spattn0', 'spattn1', 'spattn2']:
    return all_regs

def get_network(opts, arch):
  """Get Sonnet networks for training and testing
  Input:
  - opts (options) - object with all relevant options
  - arch (ArchParams) - object with all relevant Architecture options
  Output: Network (snt.Module) - network to train
  """
  regularizers = None
  if opts.architecture in ['vanilla', 'vanilla0', 'vanilla1']:
    network = networks.GraphConvLayerNetwork(
                    opts,
                    arch,
                    regularizers=get_regularizers(opts))
  elif opts.architecture in ['skip', 'skip0', 'skip1']:
    network = skip_networks.GraphSkipLayerNetwork(
                    opts,
                    arch,
                    regularizers=get_regularizers(opts))
  elif opts.architecture in ['longskip0', 'longskip1']:
    network = skip_networks.GraphLongSkipLayerNetwork(
                    opts,
                    arch,
                    regularizers=get_regularizers(opts))
  elif opts.architecture in ['normedskip0', 'normedskip1']:
    network = skip_networks.GraphLongSkipNormedNetwork(
                    opts,
                    arch,
                    regularizers=get_regularizers(opts))
  elif opts.architecture in ['normedskip2', 'normedskip3']:
    network = skip_networks.GraphSkipHopNormedNetwork(
                    opts,
                    arch,
                    regularizers=get_regularizers(opts))
  elif opts.architecture in ['attn0', 'attn1', 'attn2', \
                             'spattn0', 'spattn1', 'spattn2']:
    network = networks.GraphAttentionLayerNetwork(
                    opts,
                    arch,
                    regularizers=get_regularizers(opts))
  return network

if __name__ == "__main__":
  import data_util
  opts = options.get_opts()






