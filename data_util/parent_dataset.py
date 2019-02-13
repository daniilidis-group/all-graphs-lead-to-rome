# -*- coding: utf-8 -*-
import numpy as np
import os
import glob
import datetime
import tqdm

import tensorflow as tf

import sim_graphs
from data_util import tf_helpers

class GraphSimDataset(object):
  """Dataset for syntehtic cycle consistency graphs
  Generates synthetic graphs from sim_graphs.py then stores/loads them to/from
  tfrecords. Generates the initial embedings using random ground truth features
  with added noise for each image. Parent to all other Dataset classes.
  """
  MAX_IDX=7000

  def __init__(self, opts, params):
    """
    Inputs:
    - opts (options) - object with all relevant options stored
    - params (DatasetParams) - object with all dataset parameters stored
    Outputs: GraphSimDataset
    """
    self.opts = opts
    self.dataset_params = params
    self.data_dir = params.data_dir
    self.dtype = params.dtype
    self.n_views = np.random.randint(params.views[0], params.views[1]+1)
    self.n_pts = np.random.randint(params.points[0], params.points[1]+1)
    d = self.n_pts*self.n_views
    e = params.descriptor_dim
    p = params.points[-1]
    f = opts.final_embedding_dim
    self.features = {
      'InitEmbeddings':
           tf_helpers.TensorFeature(
                         key='InitEmbeddings',
                         shape=[d, e],
                         dtype=self.dtype,
                         description='Initial embeddings for optimization'),
      'AdjMat':
           tf_helpers.TensorFeature(
                         key='AdjMat',
                         shape=[d, d],
                         dtype=self.dtype,
                         description='Adjacency matrix for graph'),
      'Degrees':
           tf_helpers.TensorFeature(
                         key='Degrees',
                         shape=[d, d],
                         dtype=self.dtype,
                         description='Degree matrix for graph'),
      'Laplacian':
           tf_helpers.TensorFeature(
                         key='Laplacian',
                         shape=[d, d],
                         dtype=self.dtype,
                         description='Alternate Laplacian matrix for graph'),
      'Mask':
           tf_helpers.TensorFeature(
                         key='Mask',
                         shape=[d, d],
                         dtype=self.dtype,
                         description='Mask for valid values of matrix'),
      'MaskOffset':
           tf_helpers.TensorFeature(
                         key='MaskOffset',
                         shape=[d, d],
                         dtype=self.dtype,
                         description='Mask offset for loss'),
      'TrueEmbedding':
           tf_helpers.TensorFeature(
                         key='TrueEmbedding',
                         shape=[d, p],
                         dtype=self.dtype,
                         description='True values for the low dimensional embedding'),
      'NumViews':
           tf_helpers.Int64Feature(
                         key='NumViews',
                         description='Number of views used in this example'),
      'NumPoints':
           tf_helpers.Int64Feature(
                         key='NumPoints',
                         description='Number of points used in this example'),
    }

  def process_features(self, loaded_features):
    """Augmentation after generation
    Input: keys (list of strings) - keys and actual values for the dataset
    Output: sample (dict) - sample for this dataset
    """
    features = {}
    for k, feat in self.features.items():
      features[k] = feat.get_feature_write(loaded_features[k])
    return features

  def augment(self, keys, values):
    """Augmentation after generation
    Input:
    - keys (list of strings) - keys for the dataset values
    - values (list of np.array) - actual values for the dataset
    Output: 
    - keys (list of strings) - keys for the dataset values, augmented
    - values (list of np.array) - actual values for the dataset, augmented
    """
    return keys, values

  def gen_sample(self):
    """Return a single sample generated for this dataset
    Input: None
    Output: sample (dict) - sample for this dataset
    """
    # Pose graph and related objects
    params = self.dataset_params
    pose_graph = sim_graphs.PoseGraph(self.dataset_params,
                                      n_pts=self.n_pts,
                                      n_views=self.n_views)
    sz = (pose_graph.n_pts, pose_graph.n_pts)
    sz2 = (pose_graph.n_views, pose_graph.n_views)
    if params.sparse:
      mask = np.kron(pose_graph.adj_mat,np.ones(sz))
    else:
      mask = np.kron(np.ones(sz2)-np.eye(sz2[0]),np.ones(sz))

    perms_ = [ np.eye(pose_graph.n_pts)[:,pose_graph.get_perm(i)]
               for i in range(pose_graph.n_views) ]
    # Embedding objects
    TrueEmbedding = np.concatenate(perms_, 0)
    InitEmbeddings = np.concatenate([ pose_graph.get_proj(i).d
                                      for i in range(pose_graph.n_views) ], 0)

    # Graph objects
    if not params.soft_edges:
      if params.descriptor_noise_var == 0:
        AdjMat = np.dot(TrueEmbedding,TrueEmbedding.T)
        if params.sparse:
          AdjMat = AdjMat * mask
        else:
          AdjMat = AdjMat - np.eye(len(AdjMat))
        Degrees = np.diag(np.sum(AdjMat,0))
    else:
      if params.sparse and params.descriptor_noise_var > 0:
        AdjMat = pose_graph.get_feature_matching_mat()
        Degrees = np.diag(np.sum(AdjMat,0))

    # Laplacian objects
    Ahat = AdjMat + np.eye(*AdjMat.shape)
    Dhat_invsqrt = np.diag(1/np.sqrt(np.sum(Ahat,0)))
    Laplacian = np.dot(Dhat_invsqrt, np.dot(Ahat, Dhat_invsqrt))

    # Mask objects
    neg_offset = np.kron(np.eye(sz2[0]),np.ones(sz)-np.eye(sz[0]))
    Mask = AdjMat - neg_offset
    MaskOffset = neg_offset
    return {
      'InitEmbeddings': InitEmbeddings.astype(self.dtype),
      'AdjMat': AdjMat.astype(self.dtype),
      'Degrees': Degrees.astype(self.dtype),
      'Laplacian': Laplacian.astype(self.dtype),
      'Mask': Mask.astype(self.dtype),
      'MaskOffset': MaskOffset.astype(self.dtype),
      'TrueEmbedding': TrueEmbedding.astype(self.dtype),
      'NumViews': pose_graph.n_views,
      'NumPoints': pose_graph.n_pts,
    }

  def get_placeholders(self):
    """Writes data into a TF record file
    Input: None
    Output: sample (dict) - placeholders for all relevant fields
    """
    return { k:v.get_placeholder() for k, v in self.features.items() }

  def convert_dataset(self, out_dir, mode):
    """Writes data into a TF record file
    Input:
    - out_dir (string) - directory to store tf files in
    - mode (string) - train or test to load appropriate dataset
    Output: None
    Calls gen_sample many times to generate a tfrecord file for tf.Dataset
    to load from
    """
    params = self.dataset_params
    fname = '{}-{:02d}.tfrecords'
    outfile = lambda idx: os.path.join(out_dir, fname.format(mode, idx))
    if not os.path.isdir(out_dir):
      os.makedirs(out_dir)

    print('Writing dataset to {}/{}'.format(out_dir, mode))
    writer = None
    record_idx = 0
    file_idx = self.MAX_IDX + 1
    for index in tqdm.tqdm(range(params.sizes[mode])):
      if file_idx > self.MAX_IDX:
        file_idx = 0
        if writer: writer.close()
        writer = tf.python_io.TFRecordWriter(outfile(record_idx))
        record_idx += 1
      loaded_features = self.gen_sample()
      features = self.process_features(loaded_features)
      example = tf.train.Example(features=tf.train.Features(feature=features))
      writer.write(example.SerializeToString())
      file_idx += 1

    if writer: writer.close()
    # And save out a file with the creation time for versioning
    timestamp_file = '{}_timestamp.txt'.format(mode)
    with open(os.path.join(out_dir, timestamp_file), 'w') as date_file:
      date_file.write('TFrecord created {}'.format(str(datetime.datetime.now())))

  def create_np_dataset(self, out_dir, num_entries):
    """Create npz files to store dataset
    Input: 
    - out_dir (string) - directory to store npz files in
    - num_entries (int) - number of entries to generate for npz files
    Output: None
    Save out npz files storing samples into out_dir
    """
    fname = 'np_test-{:04d}.npz'
    outfile = lambda idx: os.path.join(out_dir, fname.format(idx))
    print('Writing dataset to {}'.format(out_dir))
    record_idx = 0
    for index in tqdm.tqdm(range(num_entries)):
      features = self.gen_sample()
      np.savez(outfile(index), **features)

    # And save out a file with the creation time for versioning
    timestamp_file = 'np_test_timestamp.txt'
    with open(os.path.join(out_dir, timestamp_file), 'w') as date_file:
      date_file.write('Numpy Dataset created {}'.format(str(datetime.datetime.now())))

  def gen_batch(self, mode):
    """Return batch generated for this dataset
    Input: mode (string) - train or test to load appropriate dataset
    Output: sample (dict) - sample for this dataset, with a batch dimension
    """
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

  def load_batch(self, mode):
    """Return batch loaded from this dataset
    Input: mode (string) - train or test to load appropriate dataset
    Output: iterator (tf.data.Iterator) - iterator for train/test data
    """
    params = self.dataset_params
    opts = self.opts
    assert mode in params.sizes, "Mode {} not supported".format(mode)
    data_source_name = mode + '-[0-9][0-9].tfrecords'
    data_sources = glob.glob(os.path.join(self.data_dir, mode, data_source_name))
    if opts.shuffle_data and mode != 'test':
      np.random.shuffle(data_sources) # Added to help the shuffle
    # Build dataset provider
    keys_to_features = { k: v.get_feature_read()
                         for k, v in self.features.items() }
    items_to_descriptions = { k: v.description
                              for k, v in self.features.items() }
    def parser_op(record):
      example = tf.parse_single_example(record, keys_to_features)
      return { k : v.tensors_to_item(example) for k, v in self.features.items() }
    dataset = tf.data.TFRecordDataset(data_sources)
    dataset = dataset.map(parser_op)
    dataset = dataset.repeat(None)
    if opts.shuffle_data and mode != 'test':
      dataset = dataset.shuffle(buffer_size=5*opts.batch_size)
    if opts.batch_size > 1:
      dataset = dataset.batch(opts.batch_size)
      dataset = dataset.prefetch(buffer_size=opts.batch_size)

    iterator = dataset.make_one_shot_iterator()
    sample = iterator.get_next()
    return sample


