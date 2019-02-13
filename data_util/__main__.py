# -*- coding: utf-8 -*-
"""
Generate datasets
"""
import numpy as np
import os

import options
from data_util import datasets
from data_util import noisy_dataset
from data_util import real_dataset

opts = options.get_opts()
print("Generating Pose Graphs")
if not os.path.exists(opts.data_dir):
  os.makedirs(opts.data_dir)
mydataset = datasets.get_dataset(opts)

# Create train and test tfrecords
types = [
  'train',
  'test'
]
for t in types:
  dname = os.path.join(opts.data_dir,t)
  if not os.path.exists(dname):
    os.makedirs(dname)
  mydataset.convert_dataset(dname, t)

# Generate numpy test files
out_dir = os.path.join(opts.data_dir,'np_test')
if not os.path.exists(out_dir):
  os.makedirs(out_dir)
mydataset.create_np_dataset(out_dir, opts.dataset_params.sizes['test'])


