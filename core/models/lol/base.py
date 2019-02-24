# coding: utf-8 
import sys
import numpy as np
from pprint import pprint
import tensorflow as tf
from core.models.base import ModelBase, initialize_embeddings
from core.utils.tf_utils import shape, linear, make_summary, cnn, linear
from core.utils.common import dbgprint, dotDict, recDotDefaultDict, flatten, flatten_batch, flatten_recdict

class LoLBase(ModelBase):
  def __init__(self, sess, config, trainer, vocab):
    super(LoLBase, self).__init__(sess, config, trainer, vocab)

    self.ph = self.setup_placeholder(config)
    self.embeddings = dotDict()
    self.embeddings.champions = initialize_embeddings(
      'champions', [vocab.champion.size, config.emb_size.champion])
    self.embeddings.roles = initialize_embeddings(
      'role', [vocab.role.size, config.emb_size.role])
    pick_repls = tf.nn.embedding_lookup(self.embeddings.champions, self.ph.picks)
    role_repls = tf.nn.embedding_lookup(self.embeddings.roles, self.ph.roles)
    board_repls = tf.concat([pick_repls, role_repls], axis=-1)
    with tf.variable_scope(trainer.shared_scope):
      self.board_repls = self.encode(board_repls, config.num_ffnn_layers)
    

  def encode(self, board_repls, num_ffnn_layers):
    with tf.variable_scope('Encode'):
      board_repls = cnn(board_repls, filter_sizes=[3,4,5,10])
      board_repls = tf.nn.dropout(board_repls, self.keep_prob)
      for _ in range(num_ffnn_layers):
        board_repls = linear(board_repls, activation=tf.nn.relu)
        board_repls = tf.nn.dropout(board_repls, self.keep_prob)
    return board_repls

  def inference(self, *args):
    raise NotImplementedError

  def setup_placeholder(self, config):
    # Placeholders
    with tf.name_scope('Placeholder'):
      ph = recDotDefaultDict()
      # encoder's placeholder
      ph.picks = tf.placeholder(
        tf.int32, name='pick', shape=[None, 10])
      ph.bans = tf.placeholder(
        tf.int32, name='ban', shape=[None, 10])
      ph.roles = tf.placeholder(
        tf.int32, name='role', shape=[None, 10])
      ph.win = tf.placeholder(
        tf.int32, name='win', shape=[None])
    return ph

  def get_input_feed(self, batch, is_training):
    input_feed = {}
    input_feed[self.is_training] = is_training
    input_feed[self.ph.picks] = batch.picks.ids
    input_feed[self.ph.roles] = batch.roles.ids
    input_feed[self.ph.bans] = batch.bans.ids
    input_feed[self.ph.win] = batch.win
    # for k, v in flatten_recdict(batch).items():
    #   if type(v) == np.ndarray:
    #     print(k, v.shape)
    #   else:
    #     print(k, type(v))
    # #exit(1)
    return input_feed
