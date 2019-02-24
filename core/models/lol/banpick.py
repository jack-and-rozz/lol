# coding: utf-8 
import sys
import numpy as np
from pprint import pprint
import tensorflow as tf
from core.models.lol.base import LoLBase
from core.utils.tf_utils import shape, linear, make_summary
from core.utils.common import dbgprint, dotDict, recDotDefaultDict, flatten, flatten_batch, flatten_recdict
from core.dataset.lol.match import print_example

def evaluate_and_print(examples, results):
  win_estimations = np.argmax(results, axis=-1)
  acc = 0.0
  for i, (e, r) in enumerate(zip(examples, results.tolist())):
    print('<%04d>' % i)
    print_example(e)
    if e.win == win_estimations[i]:
      acc += 1
    print()
    print('Winrate Estimation (BLUE, RED):\t', r)
    print('---------------------------------------')
    print()
  acc /= len(examples)
  print('Accuracy:\t', acc)
  return acc

class WinrateEstimater(LoLBase):
  def __init__(self, sess, config, trainer, vocab):
    super(WinrateEstimater, self).__init__(sess, config, trainer, vocab)
    logits = self.inference(config, self.board_repls)
    labels = tf.one_hot(self.ph.win, depth=2)
    self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(
      logits=logits, 
      labels=labels)
    self.gradients = self.compute_gradients(self.loss)
    self.predictions = tf.nn.softmax(logits)

  def inference(self, config, board_repls):
    logits = linear(board_repls, output_size=2, activation=None)
    return logits

  def test(self, batches, mode, logger, output_path):
    results = np.zeros([0, 2])
    used_batches = []
    sys.stderr.write('Start decoding (%s) ...\n' % mode)
    for i, batch in enumerate(batches):
      input_feed = self.get_input_feed(batch, False)
      # output_feed = [
      #   self.predictions,
      # ]
      output_feed = self.predictions
      outputs = self.sess.run(output_feed, input_feed)

      # Flatten the batch and outputs.
      used_batches += flatten_batch(batch)
      results = np.concatenate([results, outputs])
 
    sys.stdout = open(output_path, 'w') if output_path else sys.stdout
    sys.stderr.write('%d %d\n' %(len(results), len(used_batches)))
    acc = evaluate_and_print(used_batches, results)
    sys.stdout = sys.__stdout__
    summary_dict = {}
    summary_dict['%s/%s/accuracy' % (self.scopename, mode)] = acc
    summary = make_summary(summary_dict)
    return acc, summary
