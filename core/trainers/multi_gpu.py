# coding: utf-8 
import math, time, sys
from pprint import pprint
import numpy as np
import tensorflow as tf
from collections import defaultdict
from core.utils import tf_utils
from core.utils.common import dbgprint, dotDict, flatten_recdict
from core.trainers.base import TrainerBase, average_gradients
from core.models import available_models

class MultiModelWrapper(object):
  def __init__(self, models):
    self.models = models
    self.n_models = len(models)

    losses = [m.loss for m in models if m.loss is not None]
    gradients = [m.gradients for m in models if m.gradients is not None]
    self.loss = tf.reduce_mean(losses) if losses else None
    self.gradients = average_gradients(gradients) if gradients else None

  def __getattr__(self, name):
    return getattr(self.models[0], name)

  def get_input_feed(self, batch_list, is_training):
    input_feed = {}
    for b, m in zip(batch_list, self.models):
      input_feed.update(m.get_input_feed(b, is_training))
    return input_feed

class MultiGPUTrainer(TrainerBase):
  def __init__(self, sess, config, vocab, activation=tf.nn.relu):
    super(MultiGPUTrainer, self).__init__(sess, config.trainer, vocab)
    self.tasks = self.setup_tasks(sess, config)
    self.taskname = list(self.tasks.keys())[0]
    self.losses = self.tasks[self.taskname].loss
    self.updates = self.get_updates(self.tasks)

  def get_input_feed(self, batch_list, is_training):
    '''
    - batch_list: A list of batch.
    '''
    input_feed = {}
    input_feed[self.is_training] = is_training
    input_feed.update(self.tasks[self.taskname].get_input_feed(batch_list, is_training))
    return input_feed

  def run_epoch(self, batches, is_training):
    start_time = time.time()
    num_steps = 1
    loss = np.array([0.0 for _ in self.tasks])
    mode = 'train' if is_training else 'valid'
    print('<%s>' % mode)
    while True:
      t = time.time()
      try:
        model = self.tasks[self.taskname]
        batch = [batches[self.taskname].__next__() for _ in range(model.n_models)]
      except StopIteration as e:
        break

      input_feed = self.get_input_feed(batch, is_training)
      task_model = self.tasks[self.taskname]

      if task_model.debug_ops:
        print(task_model)
        for ops, res in zip(task_model.debug_ops, 
                            self.sess.run(task_model.debug_ops, input_feed)):
          print(ops, res.shape)
          print(res)
        exit(1)

      output_feed = []
      output_feed.append(self.losses)
      if is_training:
        output_feed.append(self.updates)

      t = time.time()
      outputs = self.sess.run(output_feed, input_feed)
      t = time.time() - t
      step_loss = outputs[:len(self.tasks)]
      loss += np.array(step_loss)

      print('epoch: %d,' % self.epoch.eval(), 
            'step: %d,' % self.global_step.eval(),
            'task: %s,' % ' '.join(self.tasks.keys()),
            'step_loss: %s,' % ' '.join(["%.3f" % l for l in step_loss]), 
            'step_time: %.3f,' % t)
      sys.stdout.flush()
      num_steps += 1

    epoch_time = (time.time() - start_time)
    step_time = epoch_time / num_steps
    loss = [l/num_steps for l in loss]
    if loss[0] == 0:
      raise ValueError('Set max_rows of the data more than batch_size * num_gpus.')

    mode = 'train' if is_training else 'valid'
    summary_dict = {'%s/%s/loss' % (task_model.scopename, mode): l for task_model, l in zip(self.tasks.values(), loss)}
    summary = tf_utils.make_summary(summary_dict)
    return epoch_time, step_time, loss, summary

  def get_updates(self, tasks):
    with tf.name_scope("update"):
      task = list(tasks.keys())[0]
      params = tf.contrib.framework.get_trainable_variables()
      gradients = self.tasks[task].gradients # gradients[v] = grad
      gradients = [gradients[v] for v in params]
      clipped_gradients, _ = tf.clip_by_global_norm(gradients, 
                                                    self.max_gradient_norm)
      grad_and_vars = [(g, v) for g, v in zip(clipped_gradients, params)]
      opt = getattr(tf.train, self.optimizer_type)(self.learning_rate)
      updates = opt.apply_gradients(grad_and_vars, global_step=self.global_step)
    return updates

  def setup_tasks(self, sess, config):
    try:
      assert len(config.tasks) == 1
    except:
      raise ValueError("%s can execute only one type of task." % (self.__class__.__name__))
    task_name = list(config.tasks.keys())[0]
    task_config =  list(config.tasks.values())[0]
    model_type = available_models[task_config.model_type]
    num_gpus = len(tf_utils.get_available_gpus())
    if not num_gpus:
      with tf.variable_scope(task_name, reuse=tf.AUTO_REUSE) as scope:
        models = [model_type(sess, task_config, self, self.vocab)]
    else:
      models = []
      for i in range(num_gpus):
        device = '/gpu:%d' % (i)
        with tf.variable_scope(task_name, reuse=tf.AUTO_REUSE) as scope:
          with tf.device(device):
            model = model_type(sess, task_config, self, self.vocab)
            models.append(model)
    tasks = dotDict({task_name:MultiModelWrapper(models)})
    return tasks


