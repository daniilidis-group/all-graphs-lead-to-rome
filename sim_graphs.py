# -*- coding: utf-8 -*-
import numpy as np 
import os
import sys
import collections
import scipy.linalg as la
from tqdm import tqdm

from myutils import *
import options

# Classes
Points = collections.namedtuple("Points", ["p","d"]) # position and descriptor
Pose = collections.namedtuple("Pose", ["R","T"])
PoseEdge = collections.namedtuple("PoseEdge", ["idx", "g_ij"])
class PoseGraph(object):
  """Generation for the synthetic training data, with some visualization aides
  """
  def __init__(self, params, n_pts, n_views):
    """Create PoseGraph
    Inputs:
    - params - dataset_params from options.py
    - n_pts - number of points per view
    - n_views - number of views
    Outputs: PoseGraph
    """
    self.params = params
    self.n_pts = n_pts
    self.n_views = n_views
    # Generate poses
    sph = dim_normalize(np.random.randn(self.n_views,3))
    rot = [ sph_rot(-sph[i]) for i in range(self.n_views) ]
    trans = params.scale*sph
    # Create variables
    pts = params.points_scale*np.random.randn(self.n_pts,3)
    self.desc_dim = params.descriptor_dim
    self.desc_var = params.descriptor_var
    desc = self.desc_var*np.random.randn(self.n_pts, self.desc_dim)
    self.pts_w = Points(p=pts,d=desc)
    self.g_cw = [ Pose(R=rot[i],T=trans[i]) for i in range(self.n_views) ]
    # Create graph
    eye = np.eye(self.n_views)
    dist_mat = 2 - 2*np.dot(sph, sph.T) + 3*eye
    AdjList0 = [ dist_mat[i].argsort()[:params.knn].tolist() 
                 for i in range(self.n_views) ]
    A = np.array([ sum([ eye[j] for j in AdjList0[i] ])
                   for i in range(self.n_views) ])
    self.adj_mat = np.minimum(1, A.T + A)
    get_adjs = lambda adj: np.argwhere(adj.reshape(-1) > 0).T.tolist()[0]
    self.adj_list = []
    for i in range(self.n_views):
      pose_edges = []
      for j in get_adjs(self.adj_mat[i]):
        Rij = np.dot(rot[i].T,rot[j]),
        Tij = normalize(np.dot(rot[i].T, trans[j] - trans[i])).reshape((3,1))
        pose_edges.append(PoseEdge(idx=j, g_ij=Pose(R=Rij, T=Tij)))
      self.adj_list.append(pose_edges)

  def get_random_state(self, pts):
    """Get random state determined by 3d points pts"""
    seed = (np.sum(np.abs(pts**5)))
    return np.random.RandomState(int(seed))

  def get_proj(self, i):
    """Get the 2d projection for view i"""
    pts_c = np.dot(self.pts_w.p - self.g_cw[i].T, self.g_cw[i].R.T)
    s = self.get_random_state(pts_c)
    perm = s.permutation(self.n_pts)
    proj_pos = planer_proj(pts_c)[perm,:2]
    var = self.params.descriptor_noise_var
    desc_noise = var*s.randn(self.n_pts, self.desc_dim)
    descs = self.pts_w.d[perm,:] + desc_noise
    return Points(p=proj_pos, d=descs)

  def get_perm(self, i):
    """Get the permutation of ground truth points for view i"""
    pts_c = np.dot(self.pts_w.p - self.g_cw[i].T, self.g_cw[i].R.T)
    s = self.get_random_state(pts_c)
    return s.permutation(self.n_pts)

  def get_all_permutations(self):
    """Get list of all permutations from all views"""
    return [ self.get_perm(i) for i in range(self.n_views) ]

  def get_feature_matching_mat(self):
    """Get matching matrix using the synthetic features"""
    n = self.n_pts
    m = self.n_views
    perms = [ self.get_perm(i) for i in range(m) ]
    sigma = 2
    total_graph = np.zeros((n*m, n*m))
    for i in range(m):
      for j in ([ e.idx for e in self.adj_list[i] ]):
        s_ij = np.zeros((n, n))
        descs_i = self.get_proj(i).d
        descs_j = self.get_proj(j).d
        for x in range(n):
          u = perms[i][x]
          for y in range(n):
            v = perms[j][y]
            s_ij[u,v] = np.exp(-np.linalg.norm(descs_i[u] - descs_j[v])/(sigma))
        total_graph[i*n:(i+1)*n, j*n:(j+1)*n] = s_ij
    return total_graph # + np.eye(n*m)




