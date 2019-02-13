"""
Run and save out plots from a trained model
"""
import os
import sys
import numpy as np
import argparse
import argcomplete
import matplotlib.pyplot as plt
from matplotlib import cm
import tqdm
# import yaml

import myutils
# import options

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_experiment_opts():
  parser = argparse.ArgumentParser(description='Experiment with output')
  argcomplete.autocomplete(parser)
  parser.add_argument('--verbose',
                      default=False,
                      type=str2bool,
                      help='Print everything or not')
  parser.add_argument('--index',
                      default=1,
                      type=int,
                      help='Test data index to experiment with')
  parser.add_argument('--data_path',
                      default='test001.npz',
                      help='Path to test data to experiment with')
  parser.add_argument('--save_dir',
                      default=None,
                      help='Directory to save plot files in')
  plot_options = [ 'none', 'plot', 'unsorted', 'baseline', 'random', 'save_all' ]
  parser.add_argument('--plot_style',
                      default=plot_options[0],
                      choices=plot_options,
                      help='Plot things in experiment')
  parser.add_argument('--viewer_size',
                      default=8,
                      type=int,
                      help='Run in debug mode')

  opts = parser.parse_args()
  # Finished, return options
  return opts

def npload(fdir,idx):
  return dict(np.load("{}/np_test-{:04d}.npz".format(fdir,idx)))

def get_sorted(labels):
  idxs = np.argmax(labels, axis=1)
  sorted_idxs = np.argsort(idxs)
  slabels = labels[sorted_idxs]
  return slabels, sorted_idxs

def plot_hist(save_dir, sim_mats, names, true_sim):
  fig, ax = plt.subplots(nrows=1, ncols=2)
  diags = [ np.reshape(v[true_sim==1],-1) for v in sim_mats ]
  off_diags = [ np.reshape(v[true_sim==0],-1) for v in sim_mats ]
  ax[0].hist(diags, bins=20, density=True, label=names)
  ax[0].legend()
  ax[0].set_title('Diagonal Similarity Rate')
  ax[1].hist(off_diags, bins=20, density=True, label=names)
  ax[1].set_title('Off Diagonal Similarity Rate')
  ax[1].legend()
  if save_dir:
    fig.savefig(os.path.join(save_dir, 'hist.png'))
  else:
    plt.show()

def plot_baseline(save_dir, emb_init, emb_gt, emb_out):
  slabels, sorted_idxs = get_sorted(emb_gt)
  srand = myutils.dim_normalize(emb_init[sorted_idxs])
  lsim = np.abs(np.dot(slabels, slabels.T))
  rsim = np.abs(np.dot(srand, srand.T))
  print('Sorted labels')
  fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
  im0 = ax0.imshow(slabels)
  im1 = ax1.imshow(srand)
  fig.colorbar(im0, ax=ax0)
  fig.colorbar(im1, ax=ax1)
  if save_dir:
    fig.savefig(os.path.join(save_dir, 'labels_sort.png'))
  else:
    plt.show()
  print('Sorted similarites')
  fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
  im0 = ax0.imshow(lsim)
  im1 = ax1.imshow(rsim)
  fig.colorbar(im0, ax=ax0)
  fig.colorbar(im1, ax=ax1)
  if save_dir:
    fig.savefig(os.path.join(save_dir, 'sim_sort.png'))
  else:
    plt.show()

def plot_index(save_dir, emb_init, emb_gt, emb_out):
  # Embeddings
  slabels, sorted_idxs = get_sorted(emb_gt)
  soutput = emb_out[sorted_idxs]
  srand = myutils.dim_normalize(emb_init[sorted_idxs])
  lsim = np.abs(np.dot(slabels, slabels.T))
  osim = np.abs(np.dot(soutput, soutput.T))
  rsim = np.abs(np.dot(srand, srand.T))
  fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
  u, s, v = np.linalg.svd(np.dot(soutput.T, slabels))
  o_ = np.ones_like(s)
  o_[-1] = np.linalg.det(np.dot(u,v))
  Q = np.dot(u, np.dot(np.diag(o_), v))
  im0 = ax0.imshow(np.abs(np.dot(soutput, Q)))
  im1 = ax1.imshow(osim)
  fig.colorbar(im0, ax=ax0)
  fig.colorbar(im1, ax=ax1)
  if save_dir:
    fig.savefig(os.path.join(save_dir, 'output.png'))
  else:
    plt.show()
  # Histogram
  diag = np.reshape(osim[lsim==1],-1)
  off_diag = np.reshape(osim[lsim==0],-1)
  baseline_diag = np.reshape(rsim[lsim==1],-1)
  baseline_off_diag = np.reshape(rsim[lsim==0],-1)
  fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
  ax0.hist([ diag, baseline_diag ], bins=20, density=True,
           label=[ 'diag', 'baseline_diag' ])
  ax0.legend()
  ax0.set_title('Diagonal Similarity Rate')
  ax1.hist([ off_diag, baseline_off_diag ], bins=20, density=True,
           label=[ 'off_diag', 'baseline_off_diag' ])
  ax1.set_title('Off Diagonal Similarity Rate')
  ax1.legend()
  if save_dir:
    fig.savefig(os.path.join(save_dir, 'sim_hist.png'))
  else:
    plt.show()

def plot_index_unsorted(save_dir, emb_init, emb_gt, emb_out, adjmat):
  labels = emb_gt
  rand = myutils.dim_normalize(emb_init)
  lsim = np.abs(np.dot(labels, labels.T))
  osim = np.abs(np.dot(emb_out, emb_out.T))
  rsim = np.abs(np.dot(rand, rand.T))
  fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3)
  im0 = ax0.imshow(output)
  im1 = ax1.imshow(osim)
  im2 = ax2.imshow(adjmat + np.eye(adjmat.shape[0]))
  fig.colorbar(im0, ax=ax0)
  fig.colorbar(im1, ax=ax1)
  if save_dir:
    fig.savefig(os.path.join(save_dir, 'unsorted_output.png'))
  else:
    plt.show()
  # diag = np.reshape(osim[lsim==1],-1)
  # off_diag = np.reshape(osim[lsim==0],-1)
  # baseline_diag = np.reshape(rsim[lsim==1],-1)
  # baseline_off_diag = np.reshape(rsim[lsim==0],-1)
  # fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
  # ax0.hist([ diag, baseline_diag ], bins=20, density=True,
  #          label=[ 'diag', 'baseline_diag' ])
  # ax0.legend()
  # ax0.set_title('Diagonal Similarity Rate')
  # ax1.hist([ off_diag, baseline_off_diag ], bins=20, density=True,
  #          label=[ 'off_diag', 'baseline_off_diag' ])
  # ax1.set_title('Off Diagonal Similarity Rate')
  # ax1.legend()
  # fig.savefig(os.path.join(save_dir, 'sim_hist_unsorted.png'))

def plot_random(save_dir, emb_init, emb_gt, emb_out):
  slabels, sorted_idxs = get_sorted(emb_gt)
  soutput = emb_out[sorted_idxs]
  srand = myutils.dim_normalize(emb_init[sorted_idxs])
  lsim = np.abs(np.dot(slabels, slabels.T))
  osim = np.abs(np.dot(soutput, soutput.T))
  rsim = np.abs(np.dot(srand, srand.T))
  plots = [ rsim, osim, osim**9 ]
  names = [ 'rsim', 'osim', 'osim**9' ]
  fig, ax = plt.subplots(nrows=1, ncols=len(plots))
  diags = [ np.reshape(v[lsim==1],-1) for v in plots ]
  off_diags = [ np.reshape(v[lsim==0],-1) for v in plots ]
  for i in range(len(plots)):
    ax[i].hist([ diags[i], off_diags[i] ], bins=20, density=True, label=['diag', 'off_diag'])
    ax[i].legend()
    ax[i].set_title(names[i])
    print(np.min(diags[i]))
    print(np.max(off_diags[i]))
    print('--')
  if save_dir:
    fig.savefig(os.path.join(save_dir, 'random.png'))
  else:
    plt.show()

def get_stats(emb_init, emb_gt, emb_out):
  slabels, sorted_idxs = get_sorted(emb_gt)
  soutput = emb_out[sorted_idxs]
  srand = myutils.dim_normalize(emb_init[sorted_idxs])
  lsim = np.abs(np.dot(slabels, slabels.T))
  osim = np.abs(np.dot(soutput, soutput.T))
  rsim = np.abs(np.dot(srand, srand.T))
  diag = np.reshape(osim[lsim==1],-1)
  off_diag = np.reshape(osim[lsim==0],-1)
  baseline_diag = np.reshape(rsim[lsim==1],-1)
  baseline_off_diag = np.reshape(rsim[lsim==0],-1)
  return (np.mean(diag), np.std(diag), \
          np.mean(off_diag), np.std(off_diag), \
          np.mean(baseline_diag), np.std(baseline_diag), \
          np.mean(baseline_off_diag), np.std(baseline_off_diag))


if __name__ == "__main__":
  # Build options
  # opts = options.get_opts()
  opts = get_experiment_opts()
  # Run experiment
  ld = np.load(opts.data_path)
  emb_init = ld['input']
  emb_gt = ld['gt']
  emb_out = ld['output']
  adjmat = ld['adjmat']
  n = len(emb_gt)
  if opts.plot_style == 'none':
    stats = np.zeros((n,8))
    if opts.verbose:
      for i in tqdm.tqdm(range(n)):
        stats[i] = get_stats(emb_init[i], emb_gt[i], emb_out[i])
    else:
      for i in range(n):
        stats[i] = get_stats(emb_init[i], emb_gt[i], emb_out[i])
    meanstats = np.mean(stats,0)
    print("Diag: {:.2e} +/- {:.2e}, Off Diag: {:.2e} +/- {:.2e}, " \
          "Baseline Diag: {:.2e} +/- {:.2e}, " \
          "Baseline Off Diag: {:.2e} +/- {:.2e}".format(*list(meanstats)))
    sys.exit()
  if opts.index > n:
    print("ERROR: index out of bounds")
    sys.exit()
  i = opts.index
  if opts.plot_style == 'plot':
    plot_index(None, emb_init[i], emb_gt[i], emb_out[i])
  elif opts.plot_style == 'unsorted':
    plot_index_unsorted(None, emb_init[i], emb_gt[i], emb_out[i])
  elif opts.plot_style == 'baseline':
    plot_baseline(None, emb_init[i], emb_gt[i], emb_out[i])
  elif opts.plot_style == 'random':
    plot_random(None, emb_init[i], emb_gt[i], emb_out[i])
  elif opts.plot_style == 'save_all':
    if opts.save_dir:
      save_dir = opts.save_dir
    else:
      save_dir = os.path.dirname(os.path.abspath(opts.data_path))
    plot_index(save_dir, emb_init[i], emb_gt[i], emb_out[i])
    plot_baseline(save_dir, emb_init[i], emb_gt[i], emb_out[i])
    plot_random(save_dir, emb_init[i], emb_gt[i], emb_out[i])



