# coding: utf-8
import os, re, sys, random, copy, time
import subprocess, itertools
import numpy as np
from pprint import pprint
from collections import OrderedDict, defaultdict
from core.utils.common import dotDict, recDotDefaultDict, recDotDict, flatten, batching_dicts, dbgprint, read_jsonlines
from core.vocabulary.base import PAD_ID

def partial_shuffle(data, chunk_size=10000):
  for i in range(0, len(data), chunk_size):
    idx = np.random.permutation(range(i, min(i+chunk_size, len(data))))
    data[i:i+chunk_size] = [data[j] for j in idx]
  return data

class PartitionedDatasetBase(object):
  '''
  A class for dataset divided to train, dev, test portions.
  The class containing three instances of this class is also defined aside.
  '''
  def __init__(self, vocab, config, filename, max_rows, 
               maxlen, minlen):
    '''
    Args:
    - config: A hierarchical dict containing shared settings across train/dev/test set.
    - filename: A string, specifying the filename of the set.
    - vocab: A hierarchical dict containing instances of the classes defined in "core.vocabulary.base". 

    - maxlen, minlen: A dict containing max/min numbers of words, characters, etc.  These kwargs are fed only to training set.
    '''
    self.config = config
    self.source_path = os.path.join(config.source_dir, filename)
    self.iterations_per_epoch = config.iterations_per_epoch

    # Currently the maximum window size of char-CNN is set as 5.
    self.maxlen = maxlen 
    self.minlen = minlen 
    self.max_rows = max_rows
    self.vocab = vocab
    self.data = [] # Lazy loading.

  @property
  def size(self):
    if len(self.data) == 0:
      self.load_data(self.source_path, self.max_rows)
    return len(self.data)

  def tensorize(self, data):
    batch = recDotDefaultDict()
    for d in data:
      batch = batching_dicts(batch, d) # list of dictionaries to dictionary of lists.
    batch = self.padding(batch, minlen=self.minlen, maxlen=self.maxlen)
    return batch

  def get_batch(self, batch_size, do_shuffle=False, chunk_size=10000):
    if not self.data:
      self.load_data(self.source_path, self.max_rows)

    if do_shuffle:
      self.data = partial_shuffle(self.data)

      if hasattr(self, 'iterations_per_epoch') and self.iterations_per_epoch:
        data = self.data[:self.iterations_per_epoch * batch_size]
      else:
        data = self.data
    else:
      data = self.data

    for i, b in itertools.groupby(enumerate(data), 
                                  lambda x: x[0] // (batch_size)):
      sliced_data = [x[1] for x in b] # (id, data) -> data
      batch = self.tensorize(sliced_data)
      yield batch

  def load_data(self, source_path, max_rows):
    raise NotImplementedError

  def preprocess(self, line):
    '''
    Args:
    - line: An instance of recDotDict.
    Return:
    - a preprocessed input.
    '''
    raise NotImplementedError

  def padding(self, batch, minlen=None, maxlen=None):
    raise NotImplementedError


class DatasetBase(object):
  train_class =  None
  test_class = None
  def __init__(self, config, vocab):
    if not self.train_class or not self.test_class:
      raise NotImplementedError('The class variables specifying the classes of partitioned sets, \'train_class\' and \'test_class\' must be defined.')
    self.vocab = vocab
    self.train = self.train_class(vocab,
                                  config,
                                  config.filename.train,
                                  config.max_rows.train,
                                  config.maxlen.train,
                                  config.minlen.train)
    self.valid = self.test_class(vocab,
                                 config,
                                 config.filename.valid,
                                 config.max_rows.valid,
                                 config.maxlen.test,
                                 config.minlen.test)

    self.test = self.test_class(vocab,
                                config,
                                config.filename.test,
                                config.max_rows.test,
                                config.maxlen.test,
                                config.minlen.test)
    
  @property
  def size(self):
    train_size = self.train.size if hasattr(self.train, 'size') else 0
    valid_size = self.valid.size if hasattr(self.valid, 'size') else 0
    test_size = self.test.size if hasattr(self.test, 'size') else 0
    return train_size, valid_size, test_size

# Functions for padding.

def fill_empty_brackets(sequence, max_len):
  """
  - sequence: A 1D list of list.
  """
  return sequence + [[] for _ in range(max_len - len(sequence))]

def fill_zero(sequence, length): # 最長系列が短すぎたときに0埋め
  '''
  Make the length of a sequence at least 'length' by truncating of filling 0.
  Args:
  sequence: A 1D list of integer.
  length: an integer.
  '''
  if len(sequence) < length:
    return sequence + [0 for _ in range(length - len(sequence))]
  else:
    return sequence


def define_length(batch, minlen=None, maxlen=None):
  if not minlen:
    minlen = 0

  if maxlen:
    batch_max = max([len(b) for b in batch] + [minlen])
    return min(max(maxlen, minlen), batch_max)
  else:
    return max([len(b) for b in batch] + [minlen])

def padding_2d(batch, minlen=None, maxlen=None, pad=PAD_ID, pad_type='post'):
  '''
  Args:
  batch: a 2D list. 
  maxlen: an integer.
  Return:
  A 2D tensor of which shape is [batch_size, max_num_word].
  '''
  if type(maxlen) == list:
    maxlen = maxlen[0]
  if type(minlen) == list:
    minlen = minlen[0]

  length_of_this_dim = define_length(batch, minlen, maxlen)
  return np.array([fill_zero(l[:length_of_this_dim], length_of_this_dim) for l in batch])

def padding(batch, minlen, maxlen, pad=PAD_ID):
  '''
  Args:
  - batch: A list of tensors with different shapes.
  - minlen, maxlen: A list of integers or None. Each i-th element specifies the minimum (or maximum) size of the tensor in the rank i+1.
    minlen[i] is considered as 0 if it is None, and maxlen[i] is automatically set to be equal to the maximum size of 'batch', the input tensor.
  
  e.g. 
  [[1], [2, 3], [4, 5, 6]] with minlen=[None], maxlen=[None] should be
  [[1, 0, 0], [2, 3, 0], [4, 5, 6]]
  '''
  assert len(minlen) == len(maxlen)
  assert type(batch) == list
  rank = len(minlen) + 1
  padded_batch = []

  length_of_this_dim = define_length(batch, minlen[0], maxlen[0])
  if rank == 2:
    return padding_2d(batch, minlen=minlen[0], maxlen=maxlen[0], pad=pad)

  for l in batch:
    l = fill_empty_brackets(l[:length_of_this_dim], length_of_this_dim)
    if rank == 3:
      l = padding_2d(l, minlen=minlen[1:], maxlen=maxlen[1:], pad=pad)
    else:
      l = padding(l, minlen=minlen[1:], maxlen=maxlen[1:], pad=pad)

    padded_batch.append(l)
  largest_shapes = [max(n_dims) for n_dims in zip(*[tensor.shape for tensor in padded_batch])]
  target_tensor = np.zeros([len(batch)] + largest_shapes)

  for i, tensor in enumerate(padded_batch):
    pad_lengths = [x - y for x, y in zip(largest_shapes, tensor.shape)]
    pad_shape = [(0, l) for l in pad_lengths] 
    padded_batch[i] = np.pad(tensor, pad_shape, 'constant')
  return np.array(padded_batch)
