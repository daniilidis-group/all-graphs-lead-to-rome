# -*- coding: utf-8 -*-
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt 
import time
import sklearn.metrics as metrics
import tqdm
import itertools

def scatterplot_matrix(data, names, **kwargs):
  """Plots a scatterplot matrix of subplots.  Each row of "data" is plotted
  against other rows, resulting in a nrows by nrows grid of subplots with the
  diagonal subplots labeled with "names".  Additional keyword arguments are
  passed on to matplotlib's "plot" command. Returns the matplotlib figure
  object containg the subplot grid."""
  numvars, numdata = data.shape
  fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(8,8))
  fig.subplots_adjust(hspace=0.05, wspace=0.05)

  for ax in axes.flat:
    # Hide all ticks and labels
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    # Set up ticks only on one side for the "edge" subplots...
    if ax.is_first_col():
      ax.yaxis.set_ticks_position('left')
    if ax.is_last_col():
      ax.yaxis.set_ticks_position('right')
    if ax.is_first_row():
      ax.xaxis.set_ticks_position('top')
    if ax.is_last_row():
      ax.xaxis.set_ticks_position('bottom')

  # Plot the data.
  for i, j in zip(*np.triu_indices_from(axes, k=1)):
    for x, y in [(i,j), (j,i)]:
      axes[x,y].scatter(data[x], data[y], **kwargs)

  # Label the diagonal subplots...
  for i, label in enumerate(names):
    axes[i,i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                       ha='center', va='center')

  # Turn on the proper x or y axes ticks.
  for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
    axes[j,i].xaxis.set_visible(True)
    axes[i,j].yaxis.set_visible(True)

  return fig

def main(verbose):
  # Constants
  N_ = 2048
  # MATLAB Output Files
  opt_names = [ x[:-len('TestErrors.log')] for x in sorted(glob.glob('*.log')) ]
  all_names = opt_names + ['GCN', 'AlmostPerfect', 'Random' ]
  # all_names = [ 'MatchALS015Iter', 'MatchALS025Iter', 'MatchALS050Iter', 'MatchALS100Iter' ]
  num_outputs = len(os.listdir('{}Outputs'.format(opt_names[0])))
  # Tensorflow output file
  fname = 'GCN12Layer.npz'
  with open(fname, 'rb') as f:
    ld = dict(np.load(fname))
  temb = ld['trueemb']
  outemb = ld['out']
  assert len(temb) == num_outputs
  os.makedirs('ROC-Curves', exist_ok=True)
  os.makedirs('P-R-Curves', exist_ok=True)

  roc_ = { k : [] for k in all_names }
  p_r_ = { k : [] for k in all_names }
  for i in tqdm.tqdm(range(num_outputs), disable=(verbose != 1)):
    adjmat = np.dot(temb[i], temb[i].T).reshape(-1) # They are all the same
    # MATLAB Outputs
    fig_roc, ax_roc = plt.subplots()
    fig_p_r, ax_p_r = plt.subplots()
    ax_roc.scatter([0,0,1,1],[0,1,0,1])
    ax_p_r.scatter([0,0,1,1],[0,1,0,1])
    for k in all_names:
      # Compute things
      if k == 'GCN':
        output = np.abs(np.dot(outemb[i], outemb[i].T)).reshape(-1)
      elif k == 'AlmostPerfect':
        output = np.abs(adjmat + np.random.randn(*adjmat.shape)*0.05)
      elif k == 'Random':
        output = np.abs(np.random.randn(*adjmat.shape)*0.25)
      else:
        fname = '{}Outputs/{:04d}.npy'.format(k, i+1)
        # print(fname)
        with open(fname, 'rb') as f:
          o = np.load(f)
        output = o.reshape(-1)
      # Get areas
      roc_[k].append(metrics.roc_auc_score(adjmat, output))
      p_r_[k].append(metrics.average_precision_score(adjmat, output))
      if verbose > 1:
        print('{0:04d} {1:<20}: ROC: {2:.03e}, P-R: {3:.03e}'.format(i, k,
                                                                     roc_[k][-1],
                                                                     p_r_[k][-1]))
      # PLot lines
      FPR, TPR, _ = metrics.roc_curve(adjmat, output)
      precision, recall, _ = metrics.precision_recall_curve(adjmat, output)
      ax_roc.plot(FPR, TPR, label='{} ROC ({:.03e})'.format(k, roc_[k][-1]))
      ax_p_r.plot(precision, recall, label='{} P-R ({:.03e})'.format(k, p_r_[k][-1]))

    # Finish plots
    ax_roc.set_xlabel('False Positive Rate')
    ax_p_r.set_xlabel('Precision')
    ax_roc.set_ylabel('True Positive Rate')
    ax_p_r.set_ylabel('Recall')
    ax_roc.set_title('ROC Curves')
    ax_p_r.set_title('Precision Recall Curves')
    ax_roc.legend()
    ax_p_r.legend()
    fig_roc.savefig('ROC-Curves/{:04d}.png'.format(i))
    fig_p_r.savefig('P-R-Curves/{:04d}.png'.format(i))
    plt.close(fig_roc)
    plt.close(fig_p_r)
  dispstr = '{:<15}: ROC: {:.03e} +/- {:.03e} ; P-R: {:.03e} +/- {:.03e}'
  for k in all_names:
    roc_mean = np.mean(roc_[k])
    roc_std = np.std(roc_[k])
    p_r_mean = np.mean(p_r_[k])
    p_r_std = np.std(p_r_[k])
    print(dispstr.format(k, roc_mean, roc_std, p_r_mean, p_r_std))

  # plot_names = [ 'MatchALS015Iter', 'PGDDS015Iter', 'Spectral', 'GCN' ]
  plot_names = all_names
  plot_vars = np.stack([ p_r_[k] for k in plot_names ], axis=0) 
  # # scatter_fig = scatterplot_matrix(plot_vars, plot_names)
  # plt.scatter(p_r_['MatchALS100Iter'], p_r_['GCN'])
  # plt.scatter([0,0,1,1], [0,1,0,1])
  # plt.plot([0,1], [0,1])
  # plt.show()
  # print(np.corrcoef(plot_vars))
  better_than = np.zeros((len(plot_names), len(plot_names)))
  for i in range(len(plot_names)):
    for j in range(len(plot_names)):
      better_than[i,j] = np.sum(np.array(p_r_[plot_names[i]]) < np.array(p_r_[plot_names[j]]))
  print(better_than)

if __name__ == '__main__':
  main(1)

