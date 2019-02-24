# coding: utf-8
import math
from collections import defaultdict
import tensorflow as tf

def average_gradients(gradients_list):
  '''
  <Args>
  -gradients_list: [task1_grads, task2_grads, ...]

  <Return>
  A default dict of gradients, keyed by their target parameters.
  This is because tf.gradients returns a list of gradients, but it can be called before all parameters used in other tasks are defined so the number and the order of gradients can also be variable. 
  '''
  with tf.name_scope('average_gradients'):
    params = tf.contrib.framework.get_trainable_variables()
    gradients = defaultdict(lambda: None)
    for v in params:
      g = [grads[v] for grads in gradients_list if grads[v] is not None]
      # print(v, g) (Debug: Watch parameters and their propagated gradients)
      if len(g) > 1:
        g = [tf.expand_dims(gg, 0)for gg in g]
        g = tf.reduce_mean(tf.concat(g, axis=0), axis=0) 
      elif len(g) == 1:
        g = g[0]
      else:
        g = None
      gradients[v] = g
    return gradients


class TrainerBase(object):
  def __init__(self, sess, config, vocab):
    self.sess = sess
    self.vocab = vocab
    self.optimizer_type = config.optimizer.optimizer_type
    self.decay_rate_per_epoch = config.optimizer.decay_rate_per_epoch
    self.max_gradient_norm = config.optimizer.max_gradient_norm

    self.is_training = tf.placeholder(tf.bool, name='is_training', shape=[]) 
    with tf.name_scope('keep_prob'):
      self.keep_prob = 1.0 - tf.to_float(self.is_training) * config.dropout_rate

    with tf.name_scope('global_variables'):
      self.global_step = tf.get_variable(
        "global_step", trainable=False, shape=[],  dtype=tf.int32,
        initializer=tf.constant_initializer(0, dtype=tf.int32)) 

      self.epoch = tf.get_variable(
        "epoch", trainable=False, shape=[], dtype=tf.int32,
        initializer=tf.constant_initializer(0, dtype=tf.int32)) 

      # Decay by epoch.
      self.learning_rate = tf.train.exponential_decay(
        config.optimizer.learning_rate, self.epoch,
        1, self.decay_rate_per_epoch, staircase=True)

      self.optimizer = getattr(tf.train, self.optimizer_type)(self.learning_rate)

    # Define operations in advance not to create ops in the loop.
    with tf.name_scope('add_epoch'):
      self._add_epoch = tf.assign(self.epoch, tf.add(self.epoch, tf.constant(1, dtype=tf.int32)))

    # Keep the shared scope to define layers not in the same scope.
    with tf.variable_scope("Shared", reuse=tf.AUTO_REUSE) as scope:
      self.shared_scope = scope

  # TODO: current 'ModelBase.compute_gradients' returns a defaultdict of (val, grad), not a list of gradients.

  # def get_updates(self, loss, global_step):
  #   with tf.name_scope("update"):
  #     params = tf.contrib.framework.get_trainable_variables()
  #     opt = getattr(tf.train, self.optimizer_type)(self.learning_rate)
  #     #gradients = [grad for grad, _ in opt.compute_gradients(loss)]
  #     gradients = tf.gradients(loss)
  #     clipped_gradients, _ = tf.clip_by_global_norm(gradients, 
  #                                                   self.max_gradient_norm)
  #     grad_and_vars = [(g, v) for g, v in zip(clipped_gradients, params)]
  #     updates = opt.apply_gradients(grad_and_vars, global_step=global_step)
  #   return updates

  def add_epoch(self):
    self.sess.run(self._add_epoch)

  def train(self, *args):
    return self.run_epoch(*args, True)

  def valid(self, *args):
    return self.run_epoch(*args, False)

  def test(self, batches):
    raise NotImplementedError("Directly call MTLTrainerBase.tasks[task_name].test()")
