# coding: utf-8
import math
from collections import defaultdict
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper, ResidualWrapper

def setup_cell(cell_type, size, use_residual=False, keep_prob=None):
  cell = getattr(tf.contrib.rnn, cell_type)(size)
  if keep_prob is not None:
    cell = DropoutWrapper(cell, 
                          input_keep_prob=1.0,
                          output_keep_prob=keep_prob)
  if use_residual:
    cell = ResidualWrapper(cell)
  return cell

def initialize_embeddings(name, emb_shape, initializer=None, 
                          trainable=True):
  if not initializer:
    initializer = tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3))
  embeddings = tf.get_variable(name, emb_shape, trainable=trainable,
                               initializer=initializer)
  return embeddings


class ModelBase(object):
  def __init__(self, sess, config, trainer, vocab):
    self.sess = sess
    self.scope = tf.get_variable_scope()
    self.scopename = self.scope.name.replace('_', '/')
    self.loss_weight = config.loss_weight if 'loss_weight' in config else 1.0

    self.debug_ops = []
    self.is_training = trainer.is_training
    self.keep_prob = trainer.keep_prob
    self.config = config
    self.vocab = vocab

    # Define operations in advance not to create ops in the loop.
    with tf.name_scope('add_step'):
      self.global_step = tf.get_variable(
        "global_step", trainable=False, shape=[],  dtype=tf.int32,
        initializer=tf.constant_initializer(0, dtype=tf.int32)) 

      self._add_step = tf.assign(self.global_step, tf.add(self.global_step, tf.constant(1, dtype=tf.int32)))
    
    with tf.name_scope('update_max_score'):
      self._next_score = tf.placeholder(tf.float32, name='max_score_ph', shape=[])
      self.max_score = tf.get_variable(
        "max_score", trainable=False, shape=[],  dtype=tf.float32,
        initializer=tf.constant_initializer(0.0, dtype=tf.float32)) 


      self._update_max_score = tf.assign(self.max_score, self._next_score)

  def get_input_feed(self, batch, is_training):
    return {}

  def add_step(self):
    #self.sess.run(tf.assign(self.global_step, tf.add(self.global_step, tf.constant(1, dtype=tf.int32))))
    self.sess.run(self._add_step)

  def update_max_score(self, score):
    #self.sess.run(tf.assign(self.max_score, tf.constant(score, dtype=tf.float32)))
    
    self.sess.run(self._update_max_score, feed_dict={self._next_score:score})

  def compute_gradients(self, loss):
    params = tf.contrib.framework.get_trainable_variables()
    gradients = defaultdict(lambda: None)
    for p, g in zip(params, tf.gradients(self.loss_weight * loss, params)):
      gradients[p] = g
      #gradients = [tf.expand_dims(g, 0) if g is not None else g for g in tf.gradients(loss, params)]

    return gradients

  def define_combination(self, other_models):
    '''
    This is a method called after the layers of all tasks are defined.
    Mainly for adversarial training.
    '''
    pass

class TestModelBase(object):
  '''
  A class for model which does only testing.
  '''
  pass
