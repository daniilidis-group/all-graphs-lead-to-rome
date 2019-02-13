"""
Generate the Rome16K dataset and the n-tuples associating frames. Need to specify
a top_directory where the Rome16K files have been unzipped
(http://s3.amazonaws.com/LocationRecognition/Datasets/Rome16K.tar.gz).
"""
import numpy as np 
import os
import sys
import argparse
import pickle
import tqdm
import time
import itertools as it

from data_util.rome16k import scenes
from data_util.rome16k import parse
import myutils

def get_build_scene_opts():
  """Parse arguments from command line and get all options for training."""
  parser = argparse.ArgumentParser(description='Train motion estimator')
  parser.add_argument('--build_tuples',
                      type=myutils.str2bool,
                      default=True,
                      help='Name of pickle file to load - None if no loading')
  parser.add_argument('--save_imsizes',
                      type=myutils.str2bool,
                      default=False,
                      help='Save files of images, which requires internet connectivity')
  parser.add_argument('--overwrite_tuples',
                      type=myutils.str2bool,
                      default=False,
                      help='If tuple file exists, overwrite it')
  parser.add_argument('--top_dir',
                      default='/NAS/data/stephen/Rome16K',
                      help='Storage location for pickle files')
  parser.add_argument('--save',
                      choices=parse.bundle_files + [ 'all' ],
                      default='all',
                      help='Save out bundle file to pickle file')
  parser.add_argument('--min_points',
                      default=80,
                      type=int,
                      help='Minimum overlap of points for connection')
  parser.add_argument('--max_points',
                      default=150,
                      type=int,
                      help='Maximum overlap of points for connection')
  parser.add_argument('--max_tuple_size',
                      default=4,
                      type=int,
                      help='Maximum tuple size')
  parser.add_argument('--max_num_tuples',
                      type=int,
                      default=-1,
                      help='Maximum number of tuples you want to generate')
  parser.add_argument('--verbose',
                      default=1,
                      type=int,
                      help='Print out everything')

  opts = parser.parse_args()
  return opts

def factorial(n, stop=0):
  o = 1
  while n > stop:
    o *= n
    n -= 1
  return o

def choose(n, k):
  return factorial(n, stop=k) // factorial(n - k)

def silent(x):
  pass

def process_scene_bundle(opts, bundle_file, scene_fname, tuples_fname):
  if opts.verbose > 0:
    myprint = lambda x: print(x)
  else:
    myprint = lambda x: silent(x)
  ######### Build and save out scene file ###########
  scene = parse.parse_bundle(bundle_file,
                             opts.top_dir,
                             get_imsize=opts.save_imsizes,
                             verbose=opts.verbose > 1)
  parse.save_scene(scene, scene_fname, opts.verbose > 0)

  if not opts.build_tuples:
    return 

  ######### Build and save out k-tuples ###########
  n = len(scene.cams)
  cam_pts = lambda i: set([ f.point for f in scene.cams[i].features ])
  tuples_full = []
  tuples_sizes = []
  # Length 2 is a special case
  myprint("Building pairs...")
  start_time = time.time()
  pairs, tsizes = [], []
  for x in tqdm.tqdm(it.combinations(range(n),2), total=choose(n,2), disable=opts.verbose < 1):
    p = len(cam_pts(x[0]) & cam_pts(x[1]))
    if p >= opts.min_points:
      pairs.append(x)
      tsizes.append(p)
      if opts.max_num_tuples > 0 and len(pairs) > opts.max_num_tuples:
        break
      
  tuples_full.append(pairs)
  tuples_sizes.append(tsizes)
  end_time = time.time()
  myprint("Done with pairs ({} sec)".format(end_time-start_time))
  # Length 3 and above
  for k in range(3,opts.max_tuple_size+1):
    myprint("Selecting {}-tuples...".format(k))
    start_time = time.time()
    tlist, tsizes = [], []
    tvals = tuples_full[-1]
    for (i, x) in tqdm.tqdm(enumerate(tvals), total=len(tvals), disable=opts.verbose < 1):
      xpts = cam_pts(x[0])
      for xx in x[1:]:
        xpts = xpts & cam_pts(xx)
      for j in range(x[-1]+1,n):
        p = len(cam_pts(j) & xpts)
        if p >= opts.min_points:
          tlist.append(x + (j,))
          tsizes.append(p)
          if opts.max_num_tuples > 0 and len(tlist) > opts.max_num_tuples:
            break
      if opts.max_num_tuples > 0 and len(tlist) > opts.max_num_tuples:
        break
    tuples_full.append(tlist)
    tuples_sizes.append(tsizes)
    end_time = time.time()
    myprint("Done with {}-tuples ({} sec)".format(k, end_time-start_time))

  tuples = [ [ x for i, x in enumerate(tups) if tsizes[i] <= opts.max_points ]
              for tups, tsizes in zip(tuples_full, tuples_sizes) ]

  myprint("Saving tuples...")
  with open(tuples_fname,'wb') as f:
    pickle.dump(tuples, f, protocol=pickle.HIGHEST_PROTOCOL)
  # with open(triplets_name(bundle_file, lite=True),'wb') as f:
  #   pickle.dump(triplets[:100].tolist(), f, protocol=pickle.HIGHEST_PROTOCOL)
  myprint("Done")



opts = get_build_scene_opts()
if opts.save == 'all':
  N = len(parse.bundle_files)
  for i, bundle_file in enumerate(parse.bundle_files):
    scene_fname=os.path.join(opts.top_dir,'scenes',parse.scene_fname(bundle_file))
    tuples_fname=os.path.join(opts.top_dir,'scenes',parse.tuples_fname(bundle_file))
    if opts.verbose > 0:
      print('Computing {} ({} of {})...'.format(bundle_file,i+1,N))
    if not opts.overwrite_tuples and os.path.exists(tuples_fname):
      if opts.verbose > 0:
        print('Already computed tuples, skipping...')
      continue
    start_time = time.time()
    process_scene_bundle(opts, bundle_file, scene_fname, tuples_fname)
    end_time = time.time()
    if opts.verbose > 0:
      print('Finished {} ({:0.3f} sec)'.format(bundle_file,end_time-start_time))
else:
  scene_fname=os.path.join(opts.top_dir,'scenes',parse.scene_fname(opts.save))
  tuples_fname=os.path.join(opts.top_dir,'scenes',parse.tuples_fname(opts.save))
  process_scene_bundle(opts, opts.save, scene_fname, tuples_fname)

