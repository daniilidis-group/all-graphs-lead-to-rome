"""
Various utility functions for os and numpy related stuff
"""
import os
import sys
import numpy as np 

# Numpy
def mysincf(x):
  """Numerically stable sinc function (sin(x)/x)
  Input: x (float)
  Output: z (float) - sin(x)/x, numerically stable around 0
  """
  z = x if x != 0 else 1e-16
  return np.sin(z) / z

def mysinc(x):
  """Numerically stable sinc function (sin(x)/x)
  Input: x (numpy array)
  Output: z (numpy array) - sin(x)/x, numerically stable around 0
  """
  z = np.select(x == 0, 1e-16, x)
  return np.sin(z) / z

def normalize(x):
  """Return the unit vector in the direction of x"""
  return x / (1e-16 + np.linalg.norm(x))

def sph_rot(x):
  """Takes unit vector and create rotation matrix from it
  Input: x (3x1 or 1x3 matrix)
  Output:
  - R (3x3 matrix) - rotation matrix such that dot(R,x) = x (not
    deterministically made)
  """
  x = x.reshape(-1)
  u = normalize(np.random.randn(3))
  R = np.array([
          normalize(np.cross(np.cross(u,x),x)),
          normalize(np.cross(u,x)),
          x,
     ])
  return R

def dim_norm(X):
  """Norms of the vectors along the last dimension of X
  Input: X (NxM numpy array)
  Output: X_norm (Nx1 numpy array) - norm of each row of X
  """
  return np.expand_dims(np.sqrt(np.sum(X**2, axis=-1)), axis=-1)

def dim_normalize(X):
  """Return X with vectors along last dimension normalized to unit length
  Input: X (NxM numpy array)
  Output: X_norm (Nx1 numpy array) - norm of each row of X
  """
  return X / dim_norm(X)

def planer_proj(X):
  """Return X divided by the last element of its dimension
  Input: X (NxM numpy array)
  Output: X_proj (NxM numpy array) - X with each row divided by its last element
  """
  return X / np.expand_dims(X[...,-1], axis=-1)

# Miscellaneous
def str2bool(v):
  """Convert a string into a boolean
  Input: v (string) - string to convert to boolean
  Output: v_bool (boolean) - appropriate boolean matching the string
  """
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    import argparse
    raise argparse.ArgumentTypeError('Boolean value expected.')

def next_file(directory, fname, suffix):
  """Returns name of file in directory with a number suffix incremented.
  Input: 
  - directory (string) - name of directory that file will be in
  - fname (string) - prefix to the number that will get incremented
  - suffix (string) - file suffix (e.g. .png, .jpg, .txt)
  Output: None
  When this is called, it will check all files in directory with prefix fname
  in order (i.e. fname000.png, fname001.png, etc.) until it hits the highest
  number and then returns the value with a number 1 higher than that.
  """
  fidx = 1
  name = lambda i: os.path.join(directory,"{}{:03d}{}".format(fname,i,suffix))
  while os.path.exists(name(fidx)):
    fidx += 1
  return name(fidx)


