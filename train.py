# -*- coding: utf-8 -*-
"""
Train the network.
"""
import os
import sys
import collections
import signal
import time
import numpy as np

import tensorflow as tf
from tensorflow.core.util.event_pb2 import SessionLog                 
                 
import data_util.datasets
import model
import myutils
import tfutils
import options

# Dictionary of loss functions for easy switching
loss_fns = {
  'bce': tfutils.bce_loss,
  'l1': tfutils.l1_loss,
  'l2': tfutils.l2_loss,
  'l1l2': tfutils.l1_l2_loss,
}

log_file = None
def log(string):
  """Log to both standard output and global log file
  Input: string (string) - value to log out
  Output: None
  """
  tf.logging.info(string)
  log_file.write(string)
  log_file.write('\n')

def get_geometric_loss(opts, sample, output_sim, name='geo_loss'):
  """Get geometric loss funcion (epipolar constraints)
  Input:
  - opts (options) - object with all relevant options stored
  - sample (dict) - sample output from dataset
  - output_sim (tf.Tensor) - output similarity from network
  - name (string, optional) - name prefix for tensorflow scoping (default geo_loss)
  Output: None
  """
  b = opts.batch_size
  v = opts.dataset_params.views[-1]
  p = opts.dataset_params.points[-1]
  # Build rotation matrices
  batch_size = tf.shape(sample['Rotations'])[0]
  R = tf.reshape(tf.tile(sample['Rotations'], [ 1, 1, p, 1 ]), [-1, v*p, 3, 3])
  T = tf.reshape(tf.tile(sample['Translations'], [ 1, 1, p ]), [-1, v*p, 3])
  X = tf.concat([ sample['InitEmbeddings'][...,-4:-2],
                  tf.tile(tf.ones((1,v*p,1)), [ batch_size, 1, 1 ]) ], axis=-1)
  # Compute absolute essential losses
  RX = tf.einsum('bvik,bvk->bvi',R,X)
  TcrossRX = tf.cross(T, RX)
  E_part = tfutils.batch_matmul(RX, tf.transpose(TcrossRX, perm=[0, 2, 1]))
  # Compute mask of essential losses to not include self loops
  npmask = np.kron(1-np.eye(v),np.ones((p,p))).reshape(1,v*p,v*p).astype(opts.dataset_params.dtype)
  mask = tf.convert_to_tensor(npmask, name='mask_{}'.format(name))
  # Compute symmetric part
  E = tf.multiply(tf.abs(E_part + tf.transpose(E_part, [0, 2, 1])), mask)
  if opts.full_tensorboard: # Add to tensorboard
    tf.summary.image('Geometric matrix {}'.format(name), tf.expand_dims(E, -1))
    tf.summary.histogram('Geometric matrix hist {}'.format(name), E)
    tf.summary.scalar('Geometric matrix norm {}'.format(name), tf.norm(E, ord=np.inf))
  return tf.reduce_mean(tf.multiply(output_sim, E), name=name)

def get_loss(opts, sample, output, return_gt=False, name='train'):
  """Get total loss funcion with main loss functions and regularizers
  Input:
  - opts (options) - object with all relevant options stored
  - sample (dict) - sample from training dataset
  - output (tf.Tensor) - output from network on sample
  - return_gt (boolean, optional) - return ground truth losses (default False)
  - name (string, optional) - name prefix for tensorflow scoping (default train)
  Output:
  - loss (tf.Tensor) - total summed loss value
  - gt_l1_loss (tf.Tensor) - L1 loss against ground truth (only returned
                             if return_gt=True)
  - gt_l2_loss (tf.Tensor) - L2 loss against ground truth (only returned
                             if return_gt=True)
  """
  emb = sample['TrueEmbedding']
  output_sim = tfutils.get_sim(output)
  if opts.use_abs_value:
    output_sim = tf.abs(output_sim)
  sim_true = tfutils.get_sim(emb)
  # Figure out if we are using unsupervised to know if we can use GT Adjcency
  if opts.use_unsupervised_loss:
    v = opts.dataset_params.views[-1]
    p = opts.dataset_params.points[-1]
    b = opts.batch_size 
    sim = sample['AdjMat'] + tf.eye(num_rows=v*p, batch_shape=[b])
  else:
    sim = sim_true
  if opts.full_tensorboard: # Add to tensorboard
    tf.summary.image('Output Similarity {}'.format(name), tf.expand_dims(output_sim, -1))
    tf.summary.image('Embedding Similarity {}'.format(name), tf.expand_dims(sim, -1))
  # Our main loss, the reconstruction loss
  reconstr_loss = loss_fns[opts.loss_type](sim, output_sim)
  if opts.full_tensorboard: # Add to tensorboard
    tf.summary.scalar('Reconstruction Loss {}'.format(name), reconstr_loss)
  # Geometric loss terms and tensorboard
  if opts.geometric_loss > 0:
    geo_loss = get_geometric_loss(opts, sample, output_sim, name='geom_loss_{}'.format(name))
    if opts.full_tensorboard: # Add to tensorboard
      tf.summary.scalar('Geometric Loss {}'.format(name), geo_loss)
      geo_loss_gt = get_geometric_loss(opts, sample, sim_true)
      tf.summary.scalar('Geometric Loss GT {}'.format(name), geo_loss_gt)
    loss = opts.reconstruction_loss * reconstr_loss + opts.geometric_loss * geo_loss
  else:
    loss = reconstr_loss
  tf.summary.scalar('Total Loss {}'.format(name), loss)
  # Compare to loss vs ground truth (almost always do that)
  if return_gt:
    output_sim_gt = output_sim
    if opts.loss_type == 'bce':
      output_sim_gt = tf.sigmoid(output_sim)
    gt_l1_loss = loss_fns['l1'](sim_true, output_sim_gt, add_loss=False)
    gt_l2_loss = loss_fns['l2'](sim_true, output_sim_gt, add_loss=False)
    if opts.full_tensorboard and opts.use_unsupervised_loss: # Add to tensorboard
      tf.summary.scalar('GT L1 Loss {}'.format(name), gt_l1_loss)
      tf.summary.scalar('GT L2 Loss {}'.format(name), gt_l2_loss)
    return loss, gt_l1_loss, gt_l2_loss
  else:
    return loss

def build_optimizer(opts, global_step):
  """Build optimizer for training using options in opts
  Input:
  - opts (options) - object with all relevant options stored
  - global_step (tf.Variable) - global_step variable from tensorflow
  Output: optimizer (tf.train.optimizer) - optimizer build based on opts
  """
  # Learning parameters post-processing
  num_batches = 1.0 * opts.dataset_params.sizes['train'] / opts.batch_size
  decay_steps = int(num_batches * opts.learning_rate_decay_epochs)
  use_staircase = (not opts.learning_rate_continuous)
  if opts.learning_rate_decay_type == 'fixed':
    learning_rate = tf.constant(opts.learning_rate, name='fixed_learning_rate')
  elif opts.learning_rate_decay_type == 'exponential':
    learning_rate = tf.train.exponential_decay(opts.learning_rate,
                                               global_step,
                                               decay_steps,
                                               opts.learning_rate_decay_rate,
                                               staircase=use_staircase,
                                               name='learning_rate')
  elif opts.learning_rate_decay_type == 'polynomial':
    learning_rate = tf.train.polynomial_decay(opts.learning_rate,
                                              global_step,
                                              decay_steps,
                                              opts.min_learning_rate,
                                              power=1.0,
                                              cycle=False,
                                              name='learning_rate')

  if opts.full_tensorboard: # Add to tensorboard
    tf.summary.scalar('learning_rate', learning_rate)
  # Different descent algorithms
  if opts.optimizer_type == 'adam':
    optimizer = tf.train.AdamOptimizer(learning_rate)
  elif opts.optimizer_type == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(learning_rate)
  elif opts.optimizer_type == 'momentum':
    optimizer = tf.train.MomentumOptimizer(learning_rate,opts.momentum)
  elif opts.optimizer_type == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  elif opts.optimizer_type == 'adamw':
    optimizer = tf.contrib.opt.AdamWOptimizer(learning_rate)

  return optimizer

def get_train_op(opts, loss):
  """Build train_op for training using options in opts
  Input:
  - opts (options) - object with all relevant options stored
  - loss (tf.Tensor) - loss value to train on
  Output: train_op (tf.op) - Training operation to run to train network
  """
  global_step = tf.train.get_or_create_global_step()
  optimizer = build_optimizer(opts, global_step)
  train_op = None
  if opts.weight_decay > 0 or opts.weight_l1_decay > 0:
    reg_loss = tf.losses.get_regularization_loss()
    reg_optimizer = tf.train.GradientDescentOptimizer(
                            learning_rate=opts.weight_decay)
    reg_step = reg_optimizer.minimize(reg_loss, global_step=global_step)
    with tf.control_dependencies([reg_step]):
      gvs = optimizer.compute_gradients(loss)
      capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
      train_op = optimizer.apply_gradients(capped_gvs)
  else:
    gvs = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(capped_gvs)
  return train_op
  
def build_session(opts):
  """Build tf.train.SingularMonitoredSession with relevant hooks
  Input: opts (options) - object with all relevant options stored
  Output: session (tf.train.SingularMonitoredSession)
  """
  checkpoint_dir = opts.save_dir
  if opts.checkpoint_start_dir is not None:
    checkpoint_dir = opts.checkpoint_start_dir
  saver_hook = tf.train.CheckpointSaverHook(opts.save_dir,
                                            save_secs=opts.save_interval_secs)
  merged = tf.summary.merge_all()
  summary_hook = tf.train.SummarySaverHook(output_dir=opts.save_dir, 
                                           summary_op=merged,
                                           save_secs=opts.save_summaries_secs)
  all_hooks = [saver_hook, summary_hook]
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  return tf.train.SingularMonitoredSession(
            checkpoint_dir=checkpoint_dir,
            hooks=all_hooks,
            config=config)

def get_intervals(opts):
  """Get relevant training and testing time/step intervals
  Input: opts (options) - object with all relevant options stored
  Output:
  - train_steps (int) - steps to continue training for
  - train_time (int) - wall clock time to keep training for
  - test_freq_steps (int) - how frequently in steps to run test loss
  - test_freq (int) - how frequently in wall clock time to run test loss
  """
  if opts.num_epochs > 0:
    num_batches = 1.0 * opts.dataset_params.sizes['train'] / opts.batch_size
    train_steps = int(num_batches * opts.num_epochs)
  else:
    train_steps = None
  if opts.train_time > 0:
    train_time = opts.train_time * 60
  else:
    train_time = None
  if opts.test_freq > 0:
    test_freq = opts.test_freq * 60
  else:
    test_freq = None
  if opts.test_freq_steps > 0:
    test_freq_steps = opts.test_freq_steps
  else:
    test_freq_steps = None
  return train_steps, train_time, test_freq_steps, test_freq

def get_test_dict(opts, dataset, network):
  """Build dictionary with relevant tensors for evaluating test loss
  Input:
  - opts (options) - object with all relevant options stored
  - dataset (data_util.datasets.MyDataset) - dataset to evaluate on
  - network (callable) - Sonnet network we are training/testing 
  Output:
  - test_data (dict) - Dictionary with following items
    * sample (dict) - test sample from dataset
    * output (tf.Tensor) - network output from sample
    * loss (tf.Tensor) - test loss value
    * loss_gt_l1 (tf.Tensor) - loss against ground-truth (L1)
    * loss_gt_l2 (tf.Tensor) - loss against ground-truth (L2)
    * num_steps (int) - number of steps to go in training
  """
  test_data = {}
  test_data['sample'] = dataset.load_batch('test')
  test_data['output'] = network(test_data['sample']['Laplacian'],
                                test_data['sample']['InitEmbeddings'])
  if opts.use_unsupervised_loss:
    test_loss, test_gt_l1_loss, test_gt_l2_loss = \
                        get_loss(opts,
                                 test_data['sample'],
                                 test_data['output'],
                                 return_gt=True,
                                 name='test')
    test_data['loss'] = test_loss
    test_data['loss_gt_l1'] = test_gt_l1_loss
    test_data['loss_gt_l2'] = test_gt_l2_loss
  else:
    test_data['loss'] = get_loss(opts,
                                 test_data['sample'],
                                 test_data['output'],
                                 name='test')
  num_batches = 1.0 * opts.dataset_params.sizes['test'] / opts.batch_size
  test_data['nsteps'] = int(num_batches)
  return test_data

def run_test(opts, sess, test_data, verbose=True):
  """Run test evaluation
  Input:
  - opts (options) - object with all relevant options stored
  - sess (tf.train.SingularMonitoredSession) - session from get_session
  - test_data (dict) - Dictionary from get_test_dict
  - verbose (boolean, optional) - display everything (default True)
  Output: None
  Prints out the final values of the test evaluation
  """
  # Setup
  npsave = {}
  teststr = " ------------------- "
  teststr += " Test loss = {:.4e} "
  npsave_keys = [ 'output', 'input', 'adjmat', 'gt' ]
  test_data_vals = [ test_data['output'], test_data['sample']['InitEmbeddings'], 
                     test_data['sample']['AdjMat'], test_data['sample']['TrueEmbedding'] ]
  test_vals = [ test_data['loss'] ]
  start_time = time.time()
  if opts.use_unsupervised_loss:
    teststr += ", GT L1 Loss = {:4e} , GT L2 Loss  = {:4e} "
    test_vals += [ test_data['loss_gt_l1'], test_data['loss_gt_l2'] ]
  teststr += "({:.03} sec)"
  summed_vals = [ 0 for x in range(len(test_vals)) ]
  # Run experiment
  run_outputs = sess.run(test_vals + test_data_vals)
  for t in range(len(test_vals)):
    summed_vals[t] += run_outputs[t]
  npsave = { k: v for k, v in zip(npsave_keys, run_outputs[len(test_vals):]) }
  for _ in range(test_data['nsteps']-1):
    run_outputs = sess.run(test_vals)
    for t in range(len(test_vals)):
      summed_vals[t] += run_outputs[t]
  strargs = (sv / test_data['nsteps'] for sv in summed_vals)
  np.savez(myutils.next_file(opts.save_dir, 'test', '.npz'), **npsave)
  ctime = time.time()
  log(teststr.format(*strargs, ctime-start_time))

def train(opts):
  """Train the network
  Input: opts (options) - object with all relevant options stored
  Output: None
  Saves all output in opts.save_dir, given by the user. For how to modify the
  training procedure, look at options.py
  """
  # Get data and network
  dataset = data_util.datasets.get_dataset(opts)
  network = model.get_network(opts, opts.arch)
  # Training
  with tf.device('/cpu:0'):
    if opts.load_data:
      sample = dataset.load_batch('train')
    else:
      sample = dataset.gen_batch('train')
  output = network(sample['Laplacian'], sample['InitEmbeddings'])
  loss = get_loss(opts, sample, output, name='train')
  train_op = get_train_op(opts, loss)
  # Testing
  test_data = get_test_dict(opts, dataset, network)

  # Tensorflow and logging operations
  step = 0
  train_steps, train_time, test_freq_steps, test_freq = get_intervals(opts)
  trainstr = "global step {}: loss = {} ({:.04} sec/step, time {:.04})"
  tf.logging.set_verbosity(tf.logging.INFO)
  # Build session
  with build_session(opts) as sess:
    # Train loop
    for run in range(opts.num_runs):
      stime = time.time()
      ctime = stime
      ttime = stime
      # Main loop
      while step != train_steps and ctime - stime <= train_time:
        start_time = time.time()
        _, loss_ = sess.run([ train_op, loss ])
        ctime = time.time()
        # Check for logging
        if (step % opts.log_steps) == 0:
          log(trainstr.format(step,
                                loss_,
                                ctime - start_time,
                                ctime - stime))
        # Check if time to evaluate test
        if ((test_freq_steps and step % test_freq_steps == 0) or \
            (ctime - ttime > test_freq)):
          raw_sess = sess.raw_session()
          run_test(opts, raw_sess, test_data)
          ttime = time.time()
        step += 1

if __name__ == "__main__":
  opts = options.get_opts()
  log_file = open(os.path.join(opts.save_dir, 'logfile.log'), 'a')
  train(opts)
  log_file.close()

