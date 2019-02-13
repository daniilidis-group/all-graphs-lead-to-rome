# -*- coding: utf-8 -*-
"""
Run test functions and get list of all outputs
"""
import os
import sys
import glob
import numpy as np
import time

import tensorflow as tf

import data_util.datasets
import model
import myutils
import tfutils
import options

loss_fns = {
  'bce': tfutils.bce_loss,
  'l1': tfutils.l1_loss,
  'l2': tfutils.l2_loss,
  'l1l2': tfutils.l1_l2_loss,
}

def get_test_losses(opts, sample, output, name='loss'):
  """Get testing loss funcion
  Input:
  - opts (options) - object with all relevant options stored
  - sample (dict) - sample from training dataset
  - output (tf.Tensor) - output from network on sample
  - name (string, optional) - name prefix for tensorflow scoping (default loss)
  Output:
  - gt_l1_loss (tf.Tensor) - L1 loss against ground truth 
  - gt_l2_loss (tf.Tensor) - L2 loss against ground truth
  - gt_bce_loss (tf.Tensor) - BCE loss against ground truth
  - ssame_m (tf.Tensor) - Mean similarity of corresponding points
  - ssame_var (tf.Tensor) - Standard dev. of similarity of corresponding points
  - sdiff_m (tf.Tensor) - Mean similarity of non-corresponding points
  - sdiff_var (tf.Tensor) - Standard dev. of similarity of non-corresponding
                            points 
  """
  emb = sample['TrueEmbedding']
  output_sim = tfutils.get_sim(output)
  sim_true = tfutils.get_sim(emb)
  if opts.loss_type == 'bce':
    osim = tf.sigmoid(output_sim)
    osim_log = output_sim
  else:
    osim = output_sim
    osim_log = tf.log(tf.abs(output_sim) + 1e-9)
  gt_l1_loss = loss_fns['l1'](sim_true, osim, add_loss=False)
  gt_l2_loss = loss_fns['l2'](sim_true, osim, add_loss=False)
  gt_bce_loss = loss_fns['bce'](sim_true, osim, add_loss=False)
  num_same = tf.reduce_sum(sim_true)
  num_diff = tf.reduce_sum(1-sim_true)
  ssame_m, ssame_var = tf.nn.weighted_moments(osim, None, sim_true)
  sdiff_m, sdiff_var = tf.nn.weighted_moments(osim, None, 1-sim_true)

  return gt_l1_loss, gt_l2_loss, gt_bce_loss, ssame_m, ssame_var, sdiff_m, sdiff_var

def build_test_session(opts):
  """Build tf.Session with relevant configuration for testing
  Input: opts (options) - object with all relevant options stored
  Output: session (tf.Session)
  """
  config = tf.ConfigProto(device_count = {'GPU': 0})
  # config.gpu_options.allow_growth = True
  return tf.Session(config=config)

def test_values(opts):
  """Run testing on the network
  Input: opts (options) - object with all relevant options stored
  Output: None
  Saves all output in opts.save_dir, given by the user. It loads the saved
  configuration from the options.yaml file in opts.save_dir, so only the
  opts.save_dir needs to be specified. Will test and save out all test values
  in the test set into test_output.log in opts.save_dir
  """
  # Get data and network
  dataset = data_util.datasets.get_dataset(opts)
  network = model.get_network(opts, opts.arch)
  # Sample
  sample = dataset.get_placeholders()
  print(sample)
  output = network(sample['Laplacian'], sample['InitEmbeddings'])
  losses = get_test_losses(opts, sample, output)

  # Tensorflow and logging operations
  disp_string =  '{:06d} Errors: ' \
                 'L1: {:.03e},  L2: {:.03e}, BCE: {:.03e}, ' \
                 'Same sim: {:.03e} +/- {:.03e}, ' \
                 'Diff sim: {:.03e} +/- {:.03e}, ' \
                 'Time: {:.03e}, '


  # Build session
  glob_str = os.path.join(opts.dataset_params.data_dir, 'np_test', '*npz')
  npz_files = sorted(glob.glob(glob_str))
  vars_restore = [ v for v in tf.get_collection('weights') ] + \
                 [ v for v in tf.get_collection('biases') ]
  print(vars_restore)
  saver = tf.train.Saver(vars_restore)
  with open(os.path.join(opts.save_dir, 'test_output.log'), 'a') as log_file:
    with build_test_session(opts) as sess:
      saver.restore(sess, tf.train.latest_checkpoint(opts.save_dir))
      for i, npz_file in enumerate(npz_files):
        sample_ = { k : np.expand_dims(v,0) for k, v in np.load(npz_file).items() }
        start_time = time.time()
        vals = sess.run(losses, { sample[k] : sample_[k] for k in sample.keys() })
        end_time = time.time()
        dstr = disp_string.format(i, *vals, end_time - start_time)
        print(dstr)
        log_file.write(dstr)
        log_file.write('\n')

if __name__ == "__main__":
  opts = options.get_opts()
  print("Getting options from run...")
  opts = options.parse_yaml_opts(opts)
  print("Done")
  test_values(opts)

