# coding: utf-8 
import math, time, sys
from pprint import pprint
import numpy as np
import tensorflow as tf

from core.utils.tf_utils import make_summary, assign_device, get_available_gpus
from core.utils.common import dbgprint, dotDict, flatten_recdict
from core.trainers.base import TrainerBase, average_gradients
from core.models import available_models
from core.trainers.multi_gpu import MultiModelWrapper

##############################
##      MTL Manager
##############################

class MTLTrainerBase(TrainerBase):
  def __init__(self, sess, config, vocab, activation=tf.nn.relu):
    super(MTLTrainerBase, self).__init__(sess, config.trainer, vocab)
    # Define each task.
    self.tasks = self.setup_tasks(sess, config)
    self.trainable_tasks = dotDict({k:v for k,v in self.tasks.items() if hasattr(v, 'loss') and v.loss is not None})

    # Calculate losses of the tasks and gradients.
    self.losses = [t.loss for t in self.trainable_tasks.values()]
    self.updates = self.get_updates(self.trainable_tasks)

  def setup_tasks(self, sess, config):
    # Assign GPUs (for multi-gpu computation).
    devices = [assign_device(i) for i in range(len(config.tasks))]

    tasks = dotDict()
    for i, (task_name, task_config) in enumerate(config.tasks.items()):
      device = devices[i]
      sys.stdout.write('Building %s model to %s...\n' % (task_name, device))
      with tf.variable_scope(task_name, reuse=tf.AUTO_REUSE) as scope:
        with tf.device(device):
          task_class = available_models[task_config.model_type]
          args = [sess, task_config, self, self.vocab]
          tasks[task_name] = task_class(*args)
    return tasks

  def get_input_feed(self, batch, is_training):
    input_feed = {}
    input_feed[self.is_training] = is_training
    for task_name, task_model in self.tasks.items():
      if task_name in batch and batch[task_name]:
        input_feed.update(task_model.get_input_feed(batch[task_name], is_training))
    return input_feed

  def run_epoch(self, batches, is_training):
    raise NotImplementedError

  def get_updates(self, tasks):
    raise NotImplementedError("Define in each subclass how to combine losses")

  def get_updates_by_task(self, tasks):
    updates = dotDict()
    reuse = False

    for task_name, task_model in tasks.items():
      with tf.variable_scope(task_name):
        updates[task_name] = super(MTLTrainerBase, self).get_updates(
          task_model.loss, task_model.global_step) 
      reuse = True
    return updates

class MeanLoss(MTLTrainerBase):
  def get_updates(self, tasks):
    loss = tf.reduce_mean([t.loss_weight * t.loss for t in tasks.values() if t.loss is not None])
    updates = super(MTLTrainerBase, self).get_updates(loss, self.global_step)
    return updates

  def yield_examples(self, batches, tasks):
    batch = {}
    for i, (task_name, task_model) in enumerate(tasks.items()):
      try:
        if task_name in batches:
          raw_batch = batches[task_name].__next__()
          batch.update({task_name:raw_batch})
        else:
          batch.update({task_name:{}})
      except StopIteration as e:
        return None
      except ValueError as e:
        print (e)
        exit(1)
    return batch

  def run_epoch(self, batches, is_training):
    start_time = time.time()
    num_steps = 0
    loss = np.array([0.0 for t in self.tasks.values() if t.loss is not None])
    mode = 'train' if is_training else 'valid'
    sys.stdout.write('<%s>\n' % mode)
    while True:
      t = time.time()

      # Once one of the batches of a task stops iteration in an epoch, go to the next epoch.
      batch = self.yield_examples(batches, self.tasks)
      if batch is None:
        break
      num_steps += 1

      input_feed = self.get_input_feed(batch, is_training)
      output_feed = []
      output_feed.extend(self.losses)

      if is_training:
        output_feed.append(self.updates)

      # for task_model in self.tasks.values():
      #   if task_model.debug_ops:
      #     print(task_model)
      #     print(task_model.debug_ops)
      #     for ops, res in zip(task_model.debug_ops, 
      #                         self.sess.run(task_model.debug_ops, input_feed)):
      #       print(ops, res.shape)
      #       print(res)

      t = time.time()
      outputs = self.sess.run(output_feed, input_feed)
      t = time.time() - t
      step_loss = outputs[:len(loss)]
      loss += np.array(step_loss)

      print(
        'epoch: %d,' % self.epoch.eval(), 
        'step: %d,' % self.global_step.eval(),
        'task: %s,' % ' '.join(self.trainable_tasks.keys()),
        'step_loss: %s,' % ' '.join(["%.3f" % l for l in step_loss]), 
        'step_time: %f' % t)
      sys.stdout.flush()

    epoch_time = (time.time() - start_time)
    step_time = epoch_time / num_steps
    loss = [l/num_steps for l in loss]
    mode = 'train' if is_training else 'valid'
    summary_dict = {'%s/%s/loss' % (task_model.scopename, mode): l for task_model, l in zip(self.trainable_tasks.values(), loss)}
    summary = make_summary(summary_dict)
    return epoch_time, step_time, loss, summary

class GradientSum(MeanLoss):
  def get_updates(self, tasks):
    with tf.name_scope("update"):
      params = tf.contrib.framework.get_trainable_variables()
      gradients = average_gradients([t.gradients for t in tasks.values() 
                                     if t.gradients is not None])
      gradients = [gradients[v] for v in params]
      clipped_gradients, _ = tf.clip_by_global_norm(gradients, 
                                                    self.max_gradient_norm)

      grad_and_vars = [(g, v) for g, v in zip(clipped_gradients, params)]
      updates = self.optimizer.apply_gradients(grad_and_vars, 
                                               global_step=self.global_step)
    return updates



class MTLonMultiGPU(GradientSum):
  '''
  Parallelly run all tasks in all GPUs. (i.e. num_gpu * num_tasks models are defined).
  '''
  def setup_tasks(self, sess, config):
    num_gpus = max(1, len(get_available_gpus()))
    sys.stdout.write('Available GPUs: %s\n' % str(['/gpu:%d' % i for i in range(num_gpus)]))
    tasks = dotDict()
    for task_idx, (task_name, task_config) in enumerate(config.tasks.items()):
      models = [] 
      for gpu_idx in range(num_gpus):
        device = '/gpu:%d' % gpu_idx
        sys.stdout.write('Building %s model to %s...\n' % (task_name, device))
        with tf.variable_scope(task_name, reuse=tf.AUTO_REUSE) as scope:
          with tf.device(device):
            task_class = available_models[task_config.model_type]
            args = [sess, task_config, self, self.vocab]
            model = task_class(*args)
          models.append(model)
      tasks[task_name] = MultiModelWrapper(models)

    return tasks

  def yield_examples(self, batches, tasks):
    '''
    <Args>
    - batches:
    - tasks:
    
    '''
    batch = {}
    for i, (task_name, task_model) in enumerate(tasks.items()):
      try:
        if task_name in batches:
          raw_batch = [batches[task_name].__next__() for _ in task_model.models]
          batch.update({task_name:raw_batch})
        else:
          batch.update({task_name:{}})
      except StopIteration as e:
        return None
      except ValueError as e:
        print (e)
        exit(1)
    return batch


# class BatchIterative(MTLTrainerBase):
#   def get_updates(self, tasks):
#     return self.get_updates_by_task(tasks)

#   def run_epoch(self, batches, is_training):
#     start_time = time.time()
#     num_steps_in_epoch = [0 for _ in self.trainable_tasks]
#     loss = [0.0 for _ in self.trainable_tasks]
#     is_uncomplete = True
#     while is_uncomplete:
#       is_uncomplete = False
#       t = time.time()

#       for i, (task_name, task_model) in enumerate(self.trainable_tasks.items()):
#         try:
#           raw_batch = batches[task_name].__next__()
#           batch = {task_name:raw_batch}
#           input_feed = self.get_input_feed(batch, is_training)
#           if task_model.debug_ops:
#             print(task_model)
#             print(task_model.debug_ops)
#             for ops, res in zip(task_model.debug_ops, 
#                                 self.sess.run(task_model.debug_ops, input_feed)):
#               print(ops, res.shape)
#               print(res)
#             exit(1)
#           output_feed = [task_model.loss]
#           if is_training:
#             output_feed.append(self.updates[task_name])
#           t = time.time()
#           outputs = self.sess.run(output_feed, input_feed)
#           t = time.time() - t
#           step_loss = outputs[0]

#           print('epoch: %d,' % self.epoch.eval(), 
#                 'step: %d,' % num_steps_in_epoch[i],
#                 'task: %s,' % task_name, 
#                 'step_loss: %.3f,' % step_loss, 
#                 'step_time: %f,' % t)
#           sys.stdout.flush()
#           if math.isnan(step_loss):
#             raise ValueError(
#               "Nan loss detection ... (%s: step %d)" % (task_name, num_steps_in_epoch[i])
#             )
#           num_steps_in_epoch[i] += 1
#           loss[i] += step_loss
#           is_uncomplete = True
#         except StopIteration as e:
#           print (e)
#           exit(1)
#         except ValueError as e:
#           print (e)
#           exit(1)

#     epoch_time = (time.time() - start_time)
#     loss = [l/num_steps for l, num_steps in zip(loss, num_steps_in_epoch)]
#     mode = 'train' if is_training else 'valid'
#     summary_dict = {'%s/%s/loss' % (task_model.scopename, mode): l for task_model, l in zip(self.trainable_tasks.values(), loss)}
#     summary = make_summary(summary_dict)
#     return epoch_time, loss, summary


# class OneByOne(MTLTrainerBase):
#   def get_updates(self):
#     return self.get_updates_by_task()

#   def run_epoch(self, *args):
#     return self.run_epoch_one_task(*args)

#   def get_loss(self, task_name):
#     return self.tasks[task_name].loss

#   def run_epoch_one_task(self, task_name, batches, is_training):
#     task_model = self.tasks[task_name]
#     loss = 0.0
#     start_time = time.time()
#     for i, raw_batch in enumerate(batches[task_name]):
#       batch = {task_name:raw_batch}
#       input_feed = self.get_input_feed(batch, is_training)
#       output_feed = [self.tasks[task_name].loss]
#       if is_training:
#         output_feed.append(self.updates[task_name])

#       t = time.time()
#       outputs = self.sess.run(output_feed, input_feed)
#       t = time.time() - t
      
#       step_loss = outputs[0]
#       loss += step_loss

#       print('epoch: %d,' % self.epoch.eval(), 
#             'step: %d,' % i,
#             'task: %s,' % task_name, 
#             'step_loss: %.3f,' % step_loss, 
#             'step_time: %f,' % t)
#       sys.stdout.flush()
#       #break # DEBUG
#       if math.isnan(step_loss):
#         raise ValueError(
#           "Nan loss Nan loss has been detected... (%s: step %d)" % (task_name, i))

#     loss /= i
#     mode = 'train' if is_training else 'valid'
#     summary_dict = {'%s/%s/loss' % (task_model.scopename, mode): loss}
#     summary = make_summary(summary_dict)
#     epoch_time = (time.time() - start_time)
#     return epoch_time, loss, summary
