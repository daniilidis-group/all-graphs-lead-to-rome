import numpy as np
import scipy.linalg as la
import os
import sys
import glob
import datetime
import tqdm
import pickle

import tensorflow as tf

import sim_graphs
from data_util import parent_dataset
from data_util import tf_helpers
from data_util.rome16k import parse

class Rome16KTupleDataset(parent_dataset.GraphSimDataset):
  """Abstract base class for Rome16K cycle consistency graphs
  Generates graphs from Rome16K dataset and then stores/loads them to/from
  tfrecords.
  """

  def __init__(self, opts, params, tuple_size=3):
    parent_dataset.GraphSimDataset.__init__(self, opts, params)
    self.rome16k_dir = opts.rome16k_dir
    self.tuple_size = tuple_size
    del self.features['Mask']
    del self.features['MaskOffset']
    self.dataset_params.sizes['train'] = \
        sum([ min(int(x[1]*1.5), x[tuple_size-2])
              for _, x in parse.bundle_file_info['train'].items() ])
    self.dataset_params.sizes['test'] = \
        sum([ min(int(x[1]*1.5), x[tuple_size-2])
              for _, x in parse.bundle_file_info['test'].items()  ])

  def gen_sample(self):
    print("ERROR: Cannot generate sample - need to load data")
    sys.exit(1)

  def gen_sample_from_tuple(self, scene, tupl):
    print("ERROR: Not implemented in abstract base class")
    sys.exit(1)

  def scene_fname(self, bundle_file):
    """Scene file name based on bundle number
    Inputs: bundle_file (string) - bundle file number, from
            rome16k.parse.bundle_files
    Outputs: scene_fname (string) - path to the scene file
    """
    return os.path.join(self.rome16k_dir, 'scenes', parse.scene_fname(bundle_file))

  def tuples_fname(self, bundle_file):
    """Tuples file name based on bundle number
    Inputs: bundle_file (string) - bundle file number, from
            rome16k.parse.bundle_files
    Outputs: tuple_fname (string) - path to the tuple file
    A tuples file store the n-tuples of views in the scene that have overlapping views
    """
    return os.path.join(self.rome16k_dir, 'scenes', parse.tuples_fname(bundle_file))

  def get_tuples(self, bundle_file):
    """Load tuples file based on bundle number
    Inputs: bundle_file (string) - bundle file number, from
            rome16k.parse.bundle_files
    Outputs: tuple (list of lists of tuples) - all tuples of various sizes
    """
    tuples_fname = self.tuples_fname(bundle_file)
    with open(tuples_fname, 'rb') as f:
      tuples_all = pickle.load(f)
    if self.tuple_size == 3:
      tuples = tuples_all[1]
    else:
      tuples_sel = tuples_all[self.tuple_size-2]
      n_select = int(1.5*len(tuples_all[1]))
      if n_select > len(tuples_sel):
        tuples = tuples_all[self.tuple_size-2]
      else:
        tuples = np.array(tuples_sel)
        tuples_idx = np.random.choice(np.arange(len(tuples)),
                                      size=n_select, replace=False)
        tuples = np.sort(tuples[np.sort(tuples_idx)]).tolist()
    return tuples

  def convert_dataset(self, out_dir, mode):
    params = self.dataset_params
    fname = '{}-{:02d}.tfrecords'
    outfile = lambda idx: os.path.join(out_dir, fname.format(mode, idx))
    if not os.path.isdir(out_dir):
      os.makedirs(out_dir)

    print('Writing dataset to {}/{}'.format(out_dir, mode))
    writer = None
    scene = None
    record_idx = 0
    file_idx = self.MAX_IDX + 1

    pbar = tqdm.tqdm(total=params.sizes[mode])
    for bundle_file in parse.bundle_file_info[mode]:
      scene_name = self.scene_fname(bundle_file)
      np.random.seed(hash(scene_name) % 2**32)
      scene = parse.load_scene(scene_name)
      for tupl in self.get_tuples(bundle_file):
        if file_idx > self.MAX_IDX:
          file_idx = 0
          if writer: writer.close()
          writer = tf.python_io.TFRecordWriter(outfile(record_idx))
          record_idx += 1
        loaded_features = self.gen_sample_from_tuple(scene, tupl)
        features = self.process_features(loaded_features)
        example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(example.SerializeToString())
        file_idx += 1
        pbar.update()

    if writer: writer.close()
    # And save out a file with the creation time for versioning
    timestamp_file = '{}_timestamp.txt'.format(mode)
    with open(os.path.join(out_dir, timestamp_file), 'w') as date_file:
      date_file.write('TFrecord created {}'.format(str(datetime.datetime.now())))

  def create_np_dataset(self, out_dir, num_entries):
    del num_entries
    fname = 'np_test-{:04d}.npz'
    outfile = lambda idx: os.path.join(out_dir, fname.format(idx))
    print('Writing dataset to {}'.format(out_dir))
    record_idx = 0
    pbar = tqdm.tqdm(total=self.dataset_params.sizes['test'])
    index = 0
    for bundle_file in parse.bundle_file_info['test']:
      scene_name = self.scene_fname(bundle_file)
      np.random.seed(hash(scene_name) % 2**32)
      scene = parse.load_scene(scene_name)
      for tupl in self.get_tuples(bundle_file):
        features = self.gen_sample_from_tuple(scene, tupl)
        np.savez(outfile(index), **features)
        index += 1
        pbar.update()

    # And save out a file with the creation time for versioning
    timestamp_file = 'np_test_timestamp.txt'
    with open(os.path.join(out_dir, timestamp_file), 'w') as date_file:
      date_file.write('Numpy Dataset created {}'.format(str(datetime.datetime.now())))

  def gen_batch(self, mode):
    params = self.dataset_params
    opts = self.opts
    assert mode in params.sizes, "Mode {} not supported".format(mode)
    batch_size = opts.batch_size
    keys = sorted(list(self.features.keys()))
    shapes = [ self.features[k].shape for k in keys ]
    types = [ self.features[k].dtype for k in keys ]
    tfshapes = [ tuple([batch_size] + s) for s in shapes ]
    tftypes = [ tf.as_dtype(t) for t in types ]
    def generator_fn():
      while True:
        vals = [ np.zeros([batch_size] + s, types[i])
                 for i, s in enumerate(shapes) ]
        for b in range(batch_size):
          s = self.gen_sample()
          for i, k in enumerate(keys):
            vals[i][b] = s[k]
        yield tuple(vals)
    dataset = tf.data.Dataset.from_generator(generator_fn,
                                             tuple(tftypes),
                                             tuple(tfshapes))
    batches = dataset.prefetch(2 * batch_size)

    iterator = batches.make_one_shot_iterator()
    values = iterator.get_next()
    return dict(zip(keys, values))

class KNNRome16KDataset(Rome16KTupleDataset):
  """Abstract base class for Rome16K cycle consistency graphs
  Generates graphs from Rome16K dataset and then stores/loads them to/from
  tfrecords. Build Graphs using simple k-nearest neighbor scheme. Initial
  embeddings are SIFT features as well as the x,y position, log-scale, and
  orientation.
  """
  def __init__(self, opts, params):
    super(KNNRome16KDataset, self).__init__(opts, params, tuple_size=3)

  def gen_sample_from_tuple(self, scene, tupl):
    # Parameters
    k = self.dataset_params.knn
    n = self.dataset_params.points[-1]
    v = self.dataset_params.views[-1]
    mask = np.kron(np.ones((v,v))-np.eye(v),np.ones((n,n)))
    cam_pt = lambda i: set([ f.point for f in scene.cams[i].features ])
    point_set = cam_pt(tupl[0]) & cam_pt(tupl[1]) & cam_pt(tupl[2])
    # Build features
    feat_perm = np.random.permutation(len(point_set))[:n]
    features = [] 
    for camid in tupl:
      fset = [ ([ f for f in p.features if f.cam.id == camid  ])[0] for p in point_set ]
      fset = sorted(fset, key=lambda x: x.id)
      features.append([ fset[x] for x in feat_perm ])
    # Build descriptors
    descs_ = [ np.array([ f.desc for f in feats ]) for feats in features ]
    rids = [ np.random.permutation(len(ff)) for ff in descs_ ]
    perm_mats = [ np.eye(len(perm))[perm] for perm in rids ]
    perm = la.block_diag(*perm_mats)
    descs = np.dot(perm,np.concatenate(descs_))

    # Build Graph
    desc_norms = np.sqrt(np.sum(descs**2, 1).reshape(-1, 1))
    ndescs = descs / desc_norms
    Dinit = np.dot(ndescs,ndescs.T)
    # Rescaling
    Dmin = Dinit.min()
    Dmax = Dinit.max()
    D = (Dinit - Dmin)/(Dmax-Dmin)
    L = np.copy(D)
    for i in range(v):
      for j in range(v):
        Lsub = L[n*i:n*(i+1),n*j:n*(j+1)]
        for u in range(n):
          Lsub[u,Lsub[u].argsort()[:-k]] = 0
    LLT = np.maximum(L,L.T)

    # Build dataset options
    InitEmbeddings = np.concatenate(ndescs, axis=1)
    AdjMat = LLT*mask
    Degrees = np.diag(np.sum(AdjMat,0))
    TrueEmbedding = np.concatenate(perm_mats,axis=0)
    Ahat = AdjMat + np.eye(*AdjMat.shape)
    Dhat_invsqrt = np.diag(1/np.sqrt(np.sum(Ahat,0)))
    Laplacian = np.dot(Dhat_invsqrt, np.dot(Ahat, Dhat_invsqrt))

    return {
      'InitEmbeddings': InitEmbeddings.astype(self.dtype),
      'AdjMat': AdjMat.astype(self.dtype),
      'Degrees': Degrees.astype(self.dtype),
      'Laplacian': Laplacian.astype(self.dtype),
      'TrueEmbedding': TrueEmbedding.astype(self.dtype),
      'NumViews': v,
      'NumPoints': n,
    }

class GeomKNNRome16KDataset(Rome16KTupleDataset):
  """Abstract base class for Rome16K cycle consistency graphs
  Generates graphs from Rome16K dataset and then stores/loads them to/from
  tfrecords. Build Graphs using simple k-nearest neighbor scheme, and also
  stores the pose information for each view. Initial embeddings are SIFT 
  features as well as the x,y position, log-scale, and orientation.
  """
  def __init__(self, opts, params):
    super(GeomKNNRome16KDataset, self).__init__(opts, params, tuple_size=params.views[-1])
    d = self.n_pts*self.n_views
    e = params.descriptor_dim
    self.features.update({
      'InitEmbeddings':
           tf_helpers.TensorFeature(
                         key='InitEmbeddings',
                         shape=[d, e + 2 + 1 + 1],
                         dtype=self.dtype,
                         description='Initial embeddings for optimization'),
      'Rotations':
           tf_helpers.TensorFeature(
                         key='Rotations',
                         shape=[self.tuple_size, 3, 3],
                         dtype=self.dtype,
                         description='Mask offset for loss'),
      'Translations':
           tf_helpers.TensorFeature(
                         key='Translations',
                         shape=[self.tuple_size, 3],
                         dtype=self.dtype,
                         description='Mask offset for loss'),
    })

  def build_mask(self):
    p = self.n_pts
    v = self.n_views
    return tf.convert_to_tensor(1-np.kron(np.eye(v), np.ones((p,p))))

  def gen_sample_from_tuple(self, scene, tupl):
    # Parameters
    k = self.dataset_params.knn
    n = self.dataset_params.points[-1]
    v = self.dataset_params.views[-1]
    mask = np.kron(np.ones((v,v))-np.eye(v),np.ones((n,n)))
    cam_pt = lambda i: set([ f.point for f in scene.cams[i].features ])
    point_set = set.intersection(*[ cam_pt(t) for t in tupl ])
    # Build features
    feat_perm = np.random.permutation(len(point_set))[:n]
    features = [] 
    for camid in tupl:
      fset = [ ([ f for f in p.features if f.cam.id == camid  ])[0] for p in point_set ]
      fset = sorted(fset, key=lambda x: x.id)
      features.append([ fset[x] for x in feat_perm ])
    # Build descriptors
    xy_pos_ = [ np.array([ f.pos for f in feats ]) for feats in features ]
    scale_ = [ np.array([ f.scale for f in feats ]) for feats in features ]
    orien_ = [ np.array([ f.orien for f in feats ]) for feats in features ]
    descs_ = [ np.array([ f.desc for f in feats ]) for feats in features ]
    # Apply permutation to features
    rids = [ np.random.permutation(len(ff)) for ff in descs_ ]
    perm_mats = [ np.eye(len(perm))[perm] for perm in rids ]
    perm = la.block_diag(*perm_mats)
    descs = np.dot(perm,np.concatenate(descs_))
    xy_pos = np.dot(perm,np.concatenate(xy_pos_))
    # We have to manually normalize these values as they are much larger than the others
    logscale = np.dot(perm, np.log(np.concatenate(scale_)) - 1.5).reshape(-1,1)
    orien = np.dot(perm,np.concatenate(orien_)).reshape(-1,1) / np.pi
    # Build Graph
    desc_norms = np.sqrt(np.sum(descs**2, 1).reshape(-1, 1))
    ndescs = descs / desc_norms
    Dinit = np.dot(ndescs,ndescs.T)
    # Rescaling
    Dmin = Dinit.min()
    Dmax = Dinit.max()
    D = (Dinit - Dmin)/(Dmax-Dmin)
    L = np.copy(D)
    for i in range(v):
      for j in range(v):
        Lsub = L[n*i:n*(i+1),n*j:n*(j+1)]
        for u in range(n):
          Lsub[u,Lsub[u].argsort()[:-k]] = 0
    LLT = np.maximum(L,L.T)

    # Build dataset options
    InitEmbeddings = np.concatenate([ndescs,xy_pos,logscale,orien], axis=1)
    AdjMat = LLT*mask
    Degrees = np.diag(np.sum(AdjMat,0))
    TrueEmbedding = np.concatenate(perm_mats,axis=0)
    Ahat = AdjMat + np.eye(*AdjMat.shape)
    Dhat_invsqrt = np.diag(1/np.sqrt(np.sum(Ahat,0)))
    Laplacian = np.dot(Dhat_invsqrt, np.dot(Ahat, Dhat_invsqrt))
    Rotations = np.stack([ scene.cams[i].rot.T for i in tupl ], axis=0)
    Translations = np.stack([ -np.dot(scene.cams[i].rot.T, scene.cams[i].trans)
                              for i in tupl ], axis=0)

    return {
      'InitEmbeddings': InitEmbeddings.astype(self.dtype),
      'AdjMat': AdjMat.astype(self.dtype),
      'Degrees': Degrees.astype(self.dtype),
      'Laplacian': Laplacian.astype(self.dtype),
      'TrueEmbedding': TrueEmbedding.astype(self.dtype),
      'Rotations': Rotations,
      'Translations': Translations,
      'NumViews': v,
      'NumPoints': n,
    }

