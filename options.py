# -*- coding: utf-8 -*-
"""
Get all options for training network
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import collections
import types
import yaml
import re

import myutils

class DatasetParams(argparse.Namespace):
  """Stores information about dataset"""
  def __init__(self, opts):
    super(DatasetParams, self).__init__()
    self.data_dir='{}/{}'.format(opts.datasets_dir, opts.dataset)
    self.sizes={ 'train': 40000, 'test': 3000 }
    self.fixed_size=True
    self.views=[3]
    self.points=[25]
    self.points_scale=1
    self.knn=4
    self.scale=3
    self.sparse=False
    self.soft_edges=False
    self.descriptor_dim=12
    self.descriptor_var=1.0
    self.descriptor_noise_var=0
    self.noise_level=0.1
    self.num_repeats=1
    self.num_outliers=0
    self.dtype='float32'

# All types of dataset we have considered
dataset_choices = [
  'synth_3view', 'synth_small', 'synth_4view', 'synth_5view', 'synth_6view',
  'noise_3view',
  'noise_gauss', 'noise_symgauss',
  'noise_pairwise', 'noise_pairwise3', 'noise_pairwise5',
  'noise_pairwise3view5', 'noise_pairwise3view6', 'noise_pairwise5view5',
  'noise_largepairwise3', 'noise_largepairwise5',
  'synth_pts50', 'synth_pts100',
  'noise_outlier1', 'noise_outlier2', 'noise_outlier4', 'noise_outlier8',
  'rome16kknn0',
  'rome16kgeom0', 'rome16kgeom4view0',
]

class ArchParams(argparse.Namespace):
  """Stores information about network architecture"""
  def __init__(self, opts):
    super(ArchParams, self).__init__()
    self.layer_lens = [ 32, 64 ]
    self.activ = opts.activation_type
    self.attn_lens = []
    self.skip_layers = []
    self.start_normed = 1
    self.group_size = 32
    self.normalize_emb = True
    self.sparse = False

# All types of architectures we have considered
arch_choices = [
  'vanilla', 'vanilla0', 'vanilla1',
  'skip', 'skip0', 'skip1',
  'longskip0', 'longskip1',
  'normedskip0', 'normedskip1', 'normedskip2', 'normedskip3',
  'attn0', 'attn1', 'attn2',
  'spattn0', 'spattn1', 'spattn2',
]

activation_types = ['relu','leakyrelu','tanh', 'elu']
loss_types = [ 'l2', 'bce', 'l1', 'l1l2' ]
optimizer_types = ['sgd','adam','adadelta','momentum','adamw']
lr_decay_types = ['exponential','fixed','polynomial']


def get_opts():
  """Parse arguments from command line and get all options for training.
  Inputs: None
  Outputs: opts (options) - object with all relevant options stored
  Also saves out options in yaml file in the save_dir directory
  """
  parser = argparse.ArgumentParser(description='Train motion estimator')
  # Directory and dataset options
  parser.add_argument('--save_dir',
                      default=None,
                      help='Directory to save out logs and checkpoints')
  parser.add_argument('--checkpoint_start_dir',
                      default=None,
                      help='Place to load from if not loading from save_dir')
  parser.add_argument('--data_dir',
                      default='/NAS/data/stephen/',
                      help='Directory for saving/loading dataset')
  parser.add_argument('--rome16k_dir',
                      default='/NAS/data/stephen/Rome16K',
                      help='Directory for storing Rome16K dataset (Very specific)')
  # 'synth_noise1', 'synth_noise2'
  parser.add_argument('--dataset',
                      default=dataset_choices[0],
                      choices=dataset_choices,
                      help='Choose which dataset to use')
  parser.add_argument('--datasets_dir',
                      default='/NAS/data/stephen',
                      help='Directory where all the datasets are')
  parser.add_argument('--load_data',
                      default=True,
                      type=myutils.str2bool,
                      help='Load data or just generate it on the fly. '
                           'Generating slower but you get infinite data.')
  parser.add_argument('--shuffle_data',
                      default=True,
                      type=myutils.str2bool,
                      help='Shuffle the dataset or no?')

  # Architecture parameters
  parser.add_argument('--architecture',
                      default=arch_choices[0],
                      choices=arch_choices,
                      help='Network architecture to use')
  parser.add_argument('--final_embedding_dim',
                      default=12,
                      type=int,
                      help='Dimensionality of the output')
  parser.add_argument('--activation_type',
                      default=activation_types[0],
                      choices=activation_types,
                      help='What type of activation to use')

  # Machine learning parameters
  parser.add_argument('--batch_size',
                      default=32,
                      type=int,
                      help='Size for batches')
  parser.add_argument('--use_unsupervised_loss',
                      default=False,
                      type=myutils.str2bool,
                      help='Use true adjacency or noisy one in loss')
  parser.add_argument('--use_clamping',
                      default=False,
                      type=myutils.str2bool,
                      help='Use clamping to [0, 1] on the output similarities')
  parser.add_argument('--use_abs_value',
                      default=False,
                      type=myutils.str2bool,
                      help='Use absolute value on the output similarities')
  parser.add_argument('--loss_type',
                      default=loss_types[0],
                      choices=loss_types,
                      help='Loss function to use for training')
  parser.add_argument('--reconstruction_loss',
                      default=1.0,
                      type=float,
                      help='Use true adjacency or noisy one in loss')
  parser.add_argument('--geometric_loss',
                      default=-1,
                      type=float,
                      help='Weight to use on the geometric loss')
  parser.add_argument('--weight_decay',
                      default=4e-5,
                      type=float,
                      help='Weight decay regularization')
  parser.add_argument('--weight_l1_decay',
                      default=0,
                      type=float,
                      help='L1 weight decay regularization')
  parser.add_argument('--optimizer_type',
                      default=optimizer_types[0],
                      choices=optimizer_types,
                      help='Optimizer type for adaptive learning methods')
  parser.add_argument('--learning_rate',
                      default=1e-3,
                      type=float,
                      help='Learning rate for gradient descent')
  parser.add_argument('--momentum',
                      default=0.6,
                      type=float,
                      help='Learning rate for gradient descent')
  parser.add_argument('--learning_rate_decay_type',
                      default=lr_decay_types[0],
                      choices=lr_decay_types,
                      help='Learning rate decay policy')
  parser.add_argument('--min_learning_rate',
                      default=1e-5,
                      type=float,
                      help='Minimum learning rate after decaying')
  parser.add_argument('--learning_rate_decay_rate',
                      default=0.95,
                      type=float,
                      help='Learning rate decay rate')
  parser.add_argument('--learning_rate_continuous',
                      default=False,
                      type=myutils.str2bool,
                      help='Number of epochs before learning rate decay')
  parser.add_argument('--learning_rate_decay_epochs',
                      default=4,
                      type=float,
                      help='Number of epochs before learning rate decay')

  # Training options
  parser.add_argument('--train_time',
                      default=-1,
                      type=int,
                      help='Time in minutes the training procedure runs')
  parser.add_argument('--num_epochs',
                      default=-1,
                      type=int,
                      help='Number of epochs to run training')
  parser.add_argument('--test_freq',
                      default=8,
                      type=int,
                      help='Minutes between running loss on test set')
  parser.add_argument('--test_freq_steps',
                      default=0,
                      type=int,
                      help='Number of steps between running loss on test set')
  parser.add_argument('--num_runs',
                      default=1,
                      type=int,
                      help='Number of times training runs (length determined '
                           'by run_time)')

  # Logging options
  parser.add_argument('--verbose',
                      default=False,
                      type=myutils.str2bool,
                      help='Print out everything')
  parser.add_argument('--full_tensorboard',
                      default=True,
                      type=myutils.str2bool,
                      help='Display everything on tensorboard?')
  parser.add_argument('--save_summaries_secs',
                      default=120,
                      type=int,
                      help='How frequently in seconds we save training summaries')
  parser.add_argument('--save_interval_secs',
                      default=600,
                      type=int,
                      help='Frequency in seconds to save model while training')
  parser.add_argument('--log_steps',
                      default=5,
                      type=int,
                      help='How frequently we print training loss')

  # Debugging options
  parser.add_argument('--debug',
                      default=False,
                      type=myutils.str2bool,
                      help='Run in debug mode')


  opts = parser.parse_args()

  # Get save directory default
  if opts.save_dir is None:
    save_idx = 0
    while os.path.exists('save/save-{:03d}'.format(save_idx)):
      save_idx += 1
    opts.save_dir = 'save/save-{:03d}'.format(save_idx)

  # Determine dataset
  if not opts.load_data and opts.dataset in [ 'rome16kknn0' ]:
    print("ERROR: Cannot generate samples on the fly for this dataset: {}".format(opts.dataset))
    sys.exit(1)

  dataset_params = DatasetParams(opts)
  if opts.dataset == 'synth_3view':
    pass
  elif opts.dataset == 'noise_3view':
    dataset_params.noise_level = 0.2
  elif opts.dataset == 'synth_small':
    dataset_params.sizes={ 'train': 400, 'test': 300 }
  elif opts.dataset == 'synth_4view':
    dataset_params.views = [4]
  elif opts.dataset == 'synth_5view':
    dataset_params.views = [5]
  elif opts.dataset == 'synth_6view':
    dataset_params.views = [6]
  elif opts.dataset == 'noise_gauss':
    dataset_params.noise_level = 0.1
  elif opts.dataset == 'noise_symgauss':
    dataset_params.noise_level = 0.1
    dataset_params.num_repeats = 1
  elif 'noise_pairwise' in opts.dataset:
    dataset_params.noise_level = 0.1
    regex0 = re.compile('noise_pairwise([0-9]+)view([0-9]+)$')
    regex1 = re.compile('noise_pairwise([0-9]+)$')
    nums0 = regex0.findall(opts.dataset)
    nums1 = regex1.findall(opts.dataset)
    if len(nums0) > 0:
      nums = [ int(x) for x in nums0[0] ]
      dataset_params.num_repeats = nums[0]
      dataset_params.views = [nums[1]]
    elif len(nums1) > 0:
      nums = [ int(x) for x in nums1[0] ]
      dataset_params.num_repeats = nums[0]
  elif 'noise_largepairwise' in opts.dataset:
    dataset_params.noise_level = 0.1
    dataset_params.sizes['train'] = 400000
    num_rep = re.search(r'[0-9]+', opts.dataset)
    if num_rep:
      dataset_params.num_repeats = int(num_rep.group(0))
  elif 'synth_pts' in opts.dataset:
    dataset_params.noise_level = 0.1
    num_pts = re.search(r'[0-9]+', opts.dataset)
    if num_pts:
      dataset_params.points = [ int(num_pts.group(0)) ]
  elif 'noise_outlier' in opts.dataset:
    num_out = re.search(r'[0-9]+', opts.dataset)
    if num_out:
      dataset_params.num_outliers = int(num_out.group(0))
  elif opts.dataset == 'rome16kknn0':
    dataset_params.points=[80] 
    dataset_params.descriptor_dim=128
    # The dataset size is undermined until loading
    dataset_params.sizes={ 'train': -1, 'test': -1 }
  elif opts.dataset == 'rome16kgeom0':
    dataset_params.points=[80] 
    dataset_params.descriptor_dim=128
    # The dataset size is undermined until loading
    dataset_params.sizes={ 'train': -1, 'test': -1 }
  elif opts.dataset == 'rome16kgeom4view0':
    dataset_params.views = [4]
    dataset_params.points=[80] 
    dataset_params.descriptor_dim=128
    # The dataset size is undermined until loading
    dataset_params.sizes={ 'train': -1, 'test': -1 }
  else:
    pass
  opts.data_dir = dataset_params.data_dir
  setattr(opts, 'dataset_params', dataset_params)

  # Set up architecture
  arch = ArchParams(opts)
  if opts.architecture in ['vanilla', 'skip', 'attn0', 'spattn0']:
    arch.layer_lens=[ 2**min(5+k,9) for k in range(5) ]
  elif opts.architecture in ['vanilla0', 'skip0', 'attn1', 'spattn1']:
    arch.layer_lens=[ 2**min(5+k,9) for k in range(5) ]
  elif opts.architecture in ['vanilla1', 'skip1', 'attn2', 'spattn2']:
    arch.layer_lens=[ 2**min(5+k,9) for k in range(5) ]
  elif opts.architecture in ['longskip0', 'normedskip0', 'normedskip2']:
    arch.layer_lens=[ 32, 64, 128, 256, 512, 512, 512,
                      512, 512, 512, 1024, 1024 ]
    arch.skip_layers = [ len(arch.layer_lens)//2, len(arch.layer_lens) - 1 ]
  elif opts.architecture in ['longskip1', 'normedskip1', 'normedskip3']:
    arch.layer_lens=[ 32, 64, 128, 256, 512, 512,
                      512, 512, 512, 1024, 1024,
                      512, 512, 512, 1024, 1024 ]
    arch.skip_layers = [ 5, 10, len(arch.layer_lens) - 1 ]
  if opts.architecture in [ 'spattn0', 'spattn1', 'spattn2' ]:
    arch.sparse = True
  if opts.loss_type == 'bce':
    arch.normalize_emb = False
  if opts.dataset not in [ 'rome16kgeom0', 'rome16kgeom4view0' ]:
    opts.geometric_loss = 0
  setattr(opts, 'arch', arch)

  # Post processing
  if arch.normalize_emb:
    setattr(opts, 'embedding_offset', 1)
  # Save out options
  if not os.path.exists(opts.save_dir):
    os.makedirs(opts.save_dir)
  if opts.checkpoint_start_dir and not os.path.exists(opts.checkpoint_start_dir):
    print("ERROR: Checkpoint Directory {} does not exist".format(opts.checkpoint_start_dir))
    return

  # Save out yaml file with options stored in it
  yaml_fname = os.path.join(opts.save_dir, 'options.yaml')
  if not os.path.exists(yaml_fname):
    with open(yaml_fname, 'w') as yml:
      yml.write(yaml.dump(opts.__dict__))

  # Finished, return options
  return opts

def parse_yaml_opts(opts):
  """Parse the options.yaml to reload options as saved
  Input: opts (options) - object with all relevant options
  Output: opts (options) - object with all relevant options loaded
  """
  with open(os.path.join(opts.save_dir, 'options.yaml'), 'r') as yml:
    yaml_opts = yaml.load(yml)
  opts.__dict__.update(yaml_opts)
  return opts


