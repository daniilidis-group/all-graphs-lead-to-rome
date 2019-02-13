# -*- coding: utf-8 -*-
import numpy as np
import os

from data_util import parent_dataset # import GraphSimDataset


class GraphSimNoisyDataset(parent_dataset.GraphSimDataset):
  """Dataset for syntehtic cycle consistency graphs
  Generates synthetic graphs from sim_graphs.py then stores/loads them to/from
  tfrecords. Adds 'neighbor' noise to output of parent_dataset.GraphSimDataset,
  essentially adding a matrix to confound the factorization in between
  """
  MAX_IDX=7000

  def __init__(self, opts, params):
    """
    Inputs:
    - opts (options) - object with all relevant options stored
    - params (DatasetParams) - object with all dataset parameters stored
    Outputs: GraphSimNoisyDataset
    """
    parent_dataset.GraphSimDataset.__init__(self, opts, params)

  def gen_sample(self):
    # Pose graph and related objects
    sample = parent_dataset.GraphSimDataset.gen_sample(self)

    # Graph objects
    p = self.n_pts
    noise = self.dataset_params.noise_level
    TEmb = sample['TrueEmbedding']
    Noise = np.eye(p) + noise*(np.eye(p, k=-1) + np.eye(p, k=-1))
    AdjMat = np.dot(np.dot(TEmb, Noise), TEmb.T)
    AdjMat = np.minimum(1, AdjMat)
    Degrees = np.diag(np.sum(AdjMat,0))
    sample['AdjMat'] = AdjMat.astype(self.dtype)
    sample['Degrees'] = Degrees.astype(self.dtype)

    # Laplacian objects
    Ahat = AdjMat + np.eye(*AdjMat.shape)
    Dhat_invsqrt = np.diag(1/np.sqrt(np.sum(Ahat,0)))
    Laplacian = np.dot(Dhat_invsqrt, np.dot(Ahat, Dhat_invsqrt))
    sample['Laplacian'] = Laplacian.astype(self.dtype)

    return sample

class GraphSimGaussDataset(parent_dataset.GraphSimDataset):
  """Dataset for syntehtic cycle consistency graphs
  Generates synthetic graphs from sim_graphs.py then stores/loads them to/from
  tfrecords. Adds gaussian noise to output of parent_dataset.GraphSimDataset
  """
  MAX_IDX=7000

  def __init__(self, opts, params):
    parent_dataset.GraphSimDataset.__init__(self, opts, params)

  def gen_sample(self):
    # Pose graph and related objects
    sample = parent_dataset.GraphSimDataset.gen_sample(self)

    # Graph objects
    p = self.n_pts
    n = self.n_views 
    noise = self.dataset_params.noise_level
    TEmb = sample['TrueEmbedding']
    Noise = np.abs(np.random.randn(p*n,p*n)*noise)
    AdjMat = np.dot(TEmb, TEmb.T) + Noise - np.eye(p*n)
    AdjMat = np.minimum(1, AdjMat)
    Degrees = np.diag(np.sum(AdjMat,0))
    sample['AdjMat'] = AdjMat.astype(self.dtype)
    sample['Degrees'] = Degrees.astype(self.dtype)

    # Laplacian objects
    Ahat = AdjMat + np.eye(*AdjMat.shape)
    Dhat_invsqrt = np.diag(1/np.sqrt(np.sum(Ahat,0)))
    Laplacian = np.dot(Dhat_invsqrt, np.dot(Ahat, Dhat_invsqrt))
    sample['Laplacian'] = Laplacian.astype(self.dtype)

    return sample

class GraphSimSymGaussDataset(parent_dataset.GraphSimDataset):
  """Dataset for syntehtic cycle consistency graphs
  Generates synthetic graphs from sim_graphs.py then stores/loads them to/from
  tfrecords. Adds symmetric gaussian noise to output of
  parent_dataset.GraphSimDataset
  """
  MAX_IDX=7000

  def __init__(self, opts, params):
    parent_dataset.GraphSimDataset.__init__(self, opts, params)

  def gen_sample(self):
    # Pose graph and related objects
    sample = parent_dataset.GraphSimDataset.gen_sample(self)

    # Graph objects
    p = self.n_pts
    n = self.n_views 
    noise = self.dataset_params.noise_level
    TEmb = sample['TrueEmbedding']
    Noise = np.abs(np.random.randn(p*n,p*n)*noise)
    Mask = np.kron(np.ones((n,n))-np.eye(3),np.ones((p,p)))
    AdjMat = np.dot(TEmb, TEmb.T) + ((Noise+Noise.T)/2.0)*Mask - np.eye(p*n)
    AdjMat = np.minimum(1, AdjMat)
    Degrees = np.diag(np.sum(AdjMat,0))
    sample['AdjMat'] = AdjMat.astype(self.dtype)
    sample['Degrees'] = Degrees.astype(self.dtype)

    # Laplacian objects
    Ahat = AdjMat + np.eye(*AdjMat.shape)
    Dhat_invsqrt = np.diag(1/np.sqrt(np.sum(Ahat,0)))
    Laplacian = np.dot(Dhat_invsqrt, np.dot(Ahat, Dhat_invsqrt))
    sample['Laplacian'] = Laplacian.astype(self.dtype)

    return sample

class GraphSimPairwiseDataset(parent_dataset.GraphSimDataset):
  """Dataset for syntehtic cycle consistency graphs
  Generates synthetic graphs from sim_graphs.py then stores/loads them to/from
  tfrecords. Adds individual noise matricies between the image matching matrix
  multiplies.
  """
  MAX_IDX=7000

  def __init__(self, opts, params):
    parent_dataset.GraphSimDataset.__init__(self, opts, params)

  def gen_sample(self):
    # Pose graph and related objects
    sample = parent_dataset.GraphSimDataset.gen_sample(self)

    # Graph objects
    p = self.n_pts
    n = self.n_views 
    r = self.dataset_params.num_repeats
    noise = self.dataset_params.noise_level
    perm = lambda p: np.eye(p)[np.random.permutation(p),:]
    TEmb = sample['TrueEmbedding']
    AdjMat = np.zeros((p*n,p*n))
    for i in range(n):
      TEmb_i = TEmb[p*i:p*i+p,:]
      for j in range(i+1, n):
        TEmb_j = TEmb[p*j:p*j+p,:]
        Noise = (1-noise)*np.eye(p) + noise*sum([ perm(p) for i in range(r) ])
        Val_ij = np.dot(TEmb_i, np.dot(Noise, TEmb_j.T))
        AdjMat[p*i:p*i+p, p*j:p*j+p] = Val_ij
        AdjMat[p*j:p*j+p, p*i:p*i+p] = Val_ij.T
    AdjMat = np.minimum(1, AdjMat)
    Degrees = np.diag(np.sum(AdjMat,0))
    sample['AdjMat'] = AdjMat.astype(self.dtype)
    sample['Degrees'] = Degrees.astype(self.dtype)

    # Laplacian objects
    Ahat = AdjMat + np.eye(*AdjMat.shape)
    Dhat_invsqrt = np.diag(1/np.sqrt(np.sum(Ahat,0)))
    Laplacian = np.dot(Dhat_invsqrt, np.dot(Ahat, Dhat_invsqrt))
    sample['Laplacian'] = Laplacian.astype(self.dtype)

    return sample

class GraphSimOutlierDataset(parent_dataset.GraphSimDataset):
  """Dataset for syntehtic cycle consistency graphs
  Generates synthetic graphs from sim_graphs.py then stores/loads them to/from
  tfrecords. Adds in outlier connections to the adjacency matrix
  """
  MAX_IDX=7000

  def __init__(self, opts, params):
    parent_dataset.GraphSimDataset.__init__(self, opts, params)

  def create_outlier_indeces(self, o, n):
    """Create the selection indeces for the outliers
    Inputs:
    - o (int) - number of outliers
    - n (int) - number of points total
    Outputs: outlier_sel (np.array) - list of indeces to be outliers
    """
    ind_pairs = [ (x,y) for x in range(n) for y in range(x+1,n) ]
    probs = [ 1.0/len(ind_pairs) ] * len(ind_pairs)
    outlier_ind_pairs = np.random.multinomial(o, probs, size=1)[0]
    outlier_sel = np.zeros((n,n), dtype=np.int64)
    for i in range(len(outlier_ind_pairs)):
      outlier_sel[ind_pairs[i]] = int(outlier_ind_pairs[i])
      outlier_sel[ind_pairs[i]] = (outlier_ind_pairs[i])
    # for i in range(n):
    #   for j in range(i+1,n):
    #     outlier_sel[i,j] = outlier_ind_pairs[i*n + j]
    #     outlier_sel[j,i] = outlier_ind_pairs[i*n + j]
    return outlier_sel

  def gen_sample(self):
    # Pose graph and related objects
    sample = parent_dataset.GraphSimDataset.gen_sample(self)

    # Graph objects
    p = self.n_pts
    n = self.n_views 
    r = self.dataset_params.num_repeats
    o = self.dataset_params.num_outliers
    noise = self.dataset_params.noise_level
    perm = lambda p: np.eye(p)[np.random.permutation(p),:]
    TEmb = sample['TrueEmbedding']
    AdjMat = np.zeros((p*n,p*n))
    outlier_sel =  self.create_outlier_indeces(o, n)
    # Generate matrix
    for i in range(n):
      TEmb_i = TEmb[p*i:p*i+p,:]
      for j in range(i+1, n):
        TEmb_j = TEmb[p*j:p*j+p,:]
        if outlier_sel[i,j] > 0:
          Noise = np.eye(p)
          # for _ in range(outlier_sel[i,j]):
          for _ in range(1):
            s0, s1 = np.random.choice(range(p), size=2, replace=False)
            tmp = Noise[s1,:].copy()
            Noise[s1,:] = Noise[s0,:]
            Noise[s0,:] = tmp
          Val_ij = np.dot(TEmb_i, np.dot(Noise, TEmb_j.T))
        else:
          Val_ij = np.dot(TEmb_i, TEmb_j.T)
        AdjMat[p*i:p*i+p, p*j:p*j+p] = Val_ij
        AdjMat[p*j:p*j+p, p*i:p*i+p] = Val_ij.T
    AdjMat = np.minimum(1, AdjMat)
    Degrees = np.diag(np.sum(AdjMat,0))
    sample['AdjMat'] = AdjMat.astype(self.dtype)
    sample['Degrees'] = Degrees.astype(self.dtype)

    # Laplacian objects
    Ahat = AdjMat + np.eye(*AdjMat.shape)
    Dhat_invsqrt = np.diag(1/np.sqrt(np.sum(Ahat,0)))
    Laplacian = np.dot(Dhat_invsqrt, np.dot(Ahat, Dhat_invsqrt))
    sample['Laplacian'] = Laplacian.astype(self.dtype)

    return sample

