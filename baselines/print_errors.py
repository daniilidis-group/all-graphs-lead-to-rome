import os
import sys
import numpy as np
import re

import matplotlib.pyplot as plt

egstr = '000008 Errors: L1: 1.430e-02, L2: 5.834e-03, BCE: 5.971e-02, ' \
        'Same sim: 4.281e-01 +/- 1.905e-01, Diff sim: 7.239e-03 +/- 3.317e-02, ' \
        'Area under ROC: 9.161e-01, Area under P-R: 7.879e-01, ' \
        'Time: 2.072e+00'

def stdagg(x):
  return np.sqrt(np.mean(np.array(x)**2))

def myformat2(x):
  return '{:.05e}'.format(x)

def myformat(x):
  return '{:.03f}'.format(x)

def myformat_old(x):
  y = "{:.03e}".format(x).split('e')
  return "{}e-{}".format(y[0], y[1][-1])

efmt = '[-+]?\d+\.\d*e[-+]\d+'
disp_match = re.compile(efmt)
names = [
  'l1', 'l2', 'bce', \
  'ssame_m', 'ssame_s', 'sdiff_m', 'sdiff_s', \
  'roc', 'pr', \
  'time']
def parse(line):
  return dict(zip(names, [ float(x) for x in disp_match.findall(line) ]))

agg_names = [ 'l1', 'l2', 'bce', 'ssame', 'sdiff' ]
def agg(vals):
  aggs = dict(zip(agg_names, [ None for nm in agg_names ]))
  for k in [ 'l1', 'l2', 'bce', 'roc', 'pr', 'time' ]:
    aggs[k] = (np.mean(vals[k]), np.std(vals[k]))
  for k in [ 'ssame', 'sdiff' ]:
    aggs[k] = ( np.mean(vals[k + '_m']), stdagg(vals[k + '_s']) )
  return aggs

def disp_val(aggs):
  # fstr = "{:40}, L1: {} +/- {} , L2: {} +/- {} , BCE: {} +/- {}"
  # print(fstr.format(fname, 
  #                   myformat(aggs['l1'][0]), myformat(aggs['l1'][1]),
  #                   myformat(aggs['l2'][0]), myformat(aggs['l2'][1]),
  #                   myformat(aggs['bce'][0]), myformat(aggs['bce'][1])))
  # return 
  fstr = "{:40} & {} $\pm$ {} & {} $\pm$ {} & {} $\pm$ {} & {} $\pm$ {} & {} $\pm$ {} \\\\ \\hline"
  print(fstr.format(fname, 
                    myformat(aggs['l1'][0]), myformat(aggs['l1'][1]),
                    myformat(aggs['l2'][0]), myformat(aggs['l2'][1]),
                    myformat(aggs['roc'][0]), myformat(aggs['roc'][1]),
                    myformat(aggs['pr'][0]), myformat(aggs['pr'][1]),
                    myformat(aggs['time'][0]), myformat(aggs['time'][1])))



topstr = "Method                                   &" \
         " $L_1$             &" \
         " $L_2$             &" \
         " Area under ROC    &" \
         " Area Prec.-Recall &" \
         " Time (sec)        \\ \hline"
print(topstr)
for fname in sys.argv[1:]:
  vals = dict(zip(names, [ [] for nm in names ]))
  f = open(fname, 'r')
  for line in f:
    vals_ = parse(line)
    for k, v in vals_.items():
      vals[k].append(v)

  f.close()

  # Latex
  aggs = agg(vals)
  disp_val(aggs)
