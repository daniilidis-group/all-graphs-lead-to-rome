import tensorflow as tf

# Tensorflow features
def _bytes_feature(value):
  """Create arbitrary tensor Tensorflow feature."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

class Int64Feature(object):
  """Custom class used for decoding serialized tensors."""
  def __init__(self, key, description):
    super(Int64Feature, self).__init__()
    self._key = key
    self.description = description
    self.shape = []
    self.dtype = 'int64'

  def get_placeholder(self):
    return tf.placeholder(tf.int64, shape=[None])

  def get_feature_write(self, value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

  def get_feature_read(self):
    return tf.FixedLenFeature([], tf.int64)

  def tensors_to_item(self, keys_to_tensors):
    tensor = keys_to_tensors[self._key]
    return tf.cast(tensor, dtype=tf.int64)

class TensorFeature(object):
  """Custom class used for decoding serialized tensors."""
  def __init__(self, key, shape, dtype, description):
    super(TensorFeature, self).__init__()
    self._key = key
    self.shape = shape
    self.dtype = dtype
    self.description = description

  def get_placeholder(self):
    return tf.placeholder(self.dtype, shape=[None] + self.shape)

  def get_feature_write(self, value):
    v = value.astype(self.dtype).tobytes()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))

  def get_feature_read(self):
    return tf.FixedLenFeature([], tf.string)

  def tensors_to_item(self, keys_to_tensors):
    tensor = keys_to_tensors[self._key]
    tensor = tf.decode_raw(tensor, out_type=self.dtype)
    return tf.reshape(tensor, self.shape)

