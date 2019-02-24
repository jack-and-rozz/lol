# coding: utf-8
import sys, os, random, copy, socket, time, re, argparse
from collections import OrderedDict
from pprint import pprint
import tensorflow as tf
import numpy as np

from core.manager import ManagerBase
from core.utils.common import dbgprint, dotDict, recDotDict, recDotDefaultDict, flatten, flatten_batch, flatten_recdict, timewatch, str2bool
from core.vocabulary.base import FeatureVocabulary
import core.dataset 
import core.vocabulary

import core.trainers as trainers
from core.dataset.lol.match import print_example
from core.vocabulary import VocabularyWithEmbedding, PredefinedCharVocab

BEST_CHECKPOINT_NAME = 'model.ckpt.best'

class ExperimentManager(ManagerBase):
  '''
  '''
  @timewatch()
  def __init__(self, args, sess):
    super().__init__(args, sess)
    self.config = config = self.load_config(args)
    # assert 'vocab' in self.config
    # assert 'tasks' in self.config and len(self.config.tasks)
    self.model = None

    # Load pretrained embeddings.
    self.vocab = dotDict() #recDotDefaultDict()

    self.vocab.role = FeatureVocabulary(['TOP', 'MIDDLE', 'ADC', 'SUPPORT', 'JUNGLE'])
    for k, v in self.config.vocab.items():
      vocab_type = getattr(core.vocabulary, v.vocab_type)
      self.vocab[k] = vocab_type(v)
    

    # self.vocab.encoder.word = VocabularyWithEmbedding(config.vocab.encoder.word) 
    # self.vocab.encoder.char = PredefinedCharVocab(config.vocab.encoder.char)

    # if hasattr(config.vocab, 'decoder'):
    #   self.vocab.decoder = dotDict()
    #   self.vocab.decoder.word = VocabularyWithEmbedding(config.vocab.decoder.word)
    # else:
    #   self.vocab.decoder = self.vocab.encoder

    # Load Dataset.x
    self.dataset = recDotDict()
    for k, v in config.tasks.items():
      #t = time.time()
      print(v)
      if 'dataset' in v: # for tasks without data
        dataset_type = getattr(core.dataset, v.dataset.dataset_type)
      else:
        continue

      self.dataset[k] = dataset_type(v.dataset, self.vocab)
      #self.logger.info("Loading %s data: %.2f sec" % (k, time.time() - t))

  def get_batch(self, mode):
    batches = recDotDict({'is_training': False})
    do_shuffle = False

    if mode == 'train':
      batches.is_training = True
      do_shuffle = True

    for task_name in self.config.tasks:
      if not task_name in self.dataset:
        continue
      batch_size = self.config.tasks[task_name].batch_size
      if mode != 'train':
        batch_size /= 10 # Calculating losses for all the predictions expands batch size by beam width, which can cause OOM. (TODO: automatically adjust the batch_size in testing)
      data = getattr(self.dataset[task_name], mode)
      if data.max_rows >= 0:
        batches[task_name] = data.get_batch(
          batch_size, do_shuffle=do_shuffle) 

    return batches

  @timewatch()
  def create_model(self, config, load_best=False):
    if load_best == True:
      checkpoint_path = os.path.join(self.checkpoints_path, BEST_CHECKPOINT_NAME)
    else:
      checkpoint_path = None

    trainer_type = getattr(trainers, config.trainer.trainer_type)

    # Define computation graph.
    if not self.model:
      m = trainer_type(self.sess, config, self.vocab) if not self.model else self.model
      self.model = m
    else:
      m = self.model
      return m

    if not checkpoint_path or not os.path.exists(checkpoint_path + '.index'):
      ckpt = tf.train.get_checkpoint_state(self.checkpoints_path)
      checkpoint_path = ckpt.model_checkpoint_path if ckpt else None

    self.saver = tf.train.Saver(tf.global_variables(), 
                                max_to_keep=config.trainer.max_to_keep)
    self.summary_writer = tf.summary.FileWriter(self.summaries_path, 
                                                self.sess.graph)
    if checkpoint_path and os.path.exists(checkpoint_path + '.index'):
      sys.stdout.write("Reading model parameters from %s\n" % checkpoint_path)
      self.saver.restore(self.sess, checkpoint_path)
    else:
      sys.stdout.write("Created model with fresh parameters.\n")
      tf.global_variables_initializer().run()

    # Store variable names and vocabulary for debug.
    variables_path = self.root_path + '/variables.list'
    #if not os.path.exists(variables_path):
    with open(variables_path, 'w') as f:
      variable_names = sorted([v.name + ' ' + str(v.get_shape()) for v in tf.global_variables()])
      variable_names = [name for name in variable_names if not re.search('Adam', name)]
      f.write('\n'.join(variable_names) + '\n')
    vocab_path = self.root_path + '/vocab.champions'
    if not os.path.exists(vocab_path):
      with open(vocab_path, 'w') as f:
        f.write('\n'.join(self.vocab.champion.rev_vocab) + '\n')

    vocab_path = self.root_path + '/vocab.roles'
    if not os.path.exists(vocab_path):
      with open(vocab_path, 'w') as f:
        f.write('\n'.join(self.vocab.role.rev_vocab) + '\n')

    return m

  def save_model(self, model, save_as_best=False):
    checkpoint_path = self.checkpoints_path + '/model.ckpt'
    self.saver.save(self.sess, checkpoint_path, global_step=model.epoch)
    if save_as_best:
      suffixes = ['data-00000-of-00001', 'index', 'meta']

      # Keep the older best checkpoint to handle failures in saving.
      for sfx in suffixes:
        target_path = self.checkpoints_path + "/%s.%s" % (BEST_CHECKPOINT_NAME, sfx)
        target_path_bak = self.checkpoints_path + "/%s.%s.old" % (BEST_CHECKPOINT_NAME, sfx)
        if os.path.exists(target_path):
          cmd = "mv %s %s" % (target_path, target_path_bak)
          os.system(cmd)

      # Copy the current best checkpoint.
      for sfx in suffixes:
        source_path = self.checkpoints_path + "/model.ckpt-%d.%s" % (model.epoch.eval(), sfx)
        target_path = self.checkpoints_path + "/%s.%s" % (BEST_CHECKPOINT_NAME, sfx)
        if os.path.exists(source_path):
          cmd = "cp %s %s" % (source_path, target_path)
          os.system(cmd)

      # Remove the older one.
      for sfx in suffixes:
        target_path_bak = self.checkpoints_path + "/%s.%s.old" % (BEST_CHECKPOINT_NAME, sfx)
        if os.path.exists(target_path_bak):
          cmd = "rm %s" % (target_path_bak)
          os.system(cmd)

  def train(self):
    model = self.create_model(self.config)
    if not len(model.tasks):
      raise ValueError('Specify at least 1 task in main.conf.')

    if model.epoch.eval() == 0:
      self.logger.info('Loading dataset...')
      for task_name, d in self.dataset.items():
        train_size, dev_size, test_size = d.size
        self.logger.info('<Dataset size>')
        self.logger.info('%s: %d, %d, %d' % (task_name, train_size, dev_size, test_size))

    # if isinstance(model, trainers.OneByOne):
    #   self.train_one_by_one(model)
    # else:
    #   self.train_simultaneously(model)
    self.train_simultaneously(model)

    # Do final validation and testing with the best model.
    m = self.test()
    self.logger.info("The model in epoch %d performs best." % m.epoch.eval())

  def train_simultaneously(self, model):
    m = model
    for epoch in range(m.epoch.eval(), self.config.trainer.max_epoch):
      batches = self.get_batch('train')
      learning_rate = m.learning_rate.eval()
      self.logger.info("Epoch %d: Start: learning_rate: %e" % (epoch, learning_rate))
      epoch_time, step_time, train_loss, summary = m.train(batches)
      self.logger.info("Epoch %d (train): epoch-time %.2f, loss %s" % (epoch, epoch_time, " ".join(["%.3f" % l for l in train_loss])))

      batches = self.get_batch('valid')
      epoch_time, step_time, valid_loss, summary = m.valid(batches)
      self.summary_writer.add_summary(summary, m.epoch.eval())
      self.logger.info("Epoch %d (valid): epoch-time %.2f, loss %s" % (epoch, epoch_time, " ".join(["%.3f" % l for l in valid_loss])))

      save_as_best = self.test_for_valid(m)
      self.save_model(m, save_as_best=save_as_best)
      m.add_epoch()

  def train_one_by_one(self, model):
    '''
    '''
    m = model
    def _run_task(m, task):
      epoch = m.epoch.eval()
      batches = self.get_batch('train')
      self.logger.info("Epoch %d: Start" % m.epoch.eval())
      epoch_time, step_time, train_loss, summary = m.train(task, batches)
      self.logger.info("Epoch %d (train): epoch-time %.2f, loss %.3f" % (epoch, epoch_time, train_loss))

      batches = self.get_batch('valid')
      epoch_time, step_time, valid_loss, summary = m.valid(task, batches)
      self.summary_writer.add_summary(summary, m.epoch.eval())
      self.logger.info("Epoch %d (valid): epoch-time %.2f, loss %.3f" % (epoch, epoch_time, valid_loss))
      m.add_epoch()

    # Train the model in a reverse order of important tasks.
    task = m.tasks.keys()[1]
    for i in range(m.epoch.eval(), int(self.config.trainer.max_epoch/2)):
      _run_task(m, task)
      save_as_best = self.test_for_valid(model=m, target_tasks=[task])
      self.save_model(m, save_as_best=save_as_best)

    # Load a model with the best score of WikiP2D task. 
    best_ckpt = os.path.join(self.checkpoints_path, BEST_CHECKPOINT_NAME)
    m = self.create_model(self.config, load_best=True)

    task = m.tasks.keys()[0]
    for epoch in range(m.epoch.eval(), self.config.trainer.max_epoch):
      _run_task(m, task)
      save_as_best = self.test_for_valid(model=m, target_tasks=[task])
      self.save_model(m, save_as_best=save_as_best)

  def test_for_valid(self, model, target_tasks=None):
    '''
    This is a function to show the performance of the model in each epoch.
    If you'd like to run testing again in a different setting from terminal, execute test().
    <Args>
    - model:
    - target_tasks:
    <Return>
    - A boolean, which shows whether the score of the first task in this epoch becomes higher or not.
    '''
    m = model

    tasks = OrderedDict([(k, v) for k, v in m.tasks.items() if not target_tasks or k in target_tasks])
    epoch = m.epoch.eval()
    save_as_best = [False for t in tasks]

    mode = 'valid'
    valid_batches = self.get_batch(mode)
    for i, (task_name, task_model) in enumerate(tasks.items()):
      if task_name not in valid_batches:
        continue
      
      batches = valid_batches[task_name]
      output_path = self.tests_path + '/%s_%s.%02d' % (task_name, mode, epoch)
      valid_score, valid_summary = task_model.test(batches, mode, 
                                                   self.logger, output_path)
      self.summary_writer.add_summary(valid_summary, m.epoch.eval())
      
      if valid_score > task_model.max_score.eval():
        save_as_best[i] = True
        self.logger.info("Epoch %d (valid): %s max score update (%.3f->%.3f): " % (m.epoch.eval(), task_name, task_model.max_score.eval(), valid_score))
        task_model.update_max_score(valid_score)

    # mode = 'test'
    # test_batches = self.get_batch(mode)
    # for i, (task_name, task_model) in enumerate(tasks.items()):
    #   if not task_name in test_batches:
    #     continue
    #   batches = test_batches[task_name]
    #   output_path = self.tests_path + '/%s_%s.%02d' % (task_name, mode, epoch)
    #   test_score, test_summary = task_model.test(batches, mode, 
    #                                              self.logger, output_path)
    #   self.summary_writer.add_summary(test_summary, m.epoch.eval())

    # Currently select the best epoch by the score on the first task.
    return save_as_best[0] 

  def test(self):
    target_tasks = []
    m = self.create_model(self.config, load_best=True)
    tasks = OrderedDict([(k, v) for k, v in m.tasks.items() if not target_tasks or k in target_tasks])

    mode = 'valid'
    valid_batches = self.get_batch(mode)
    for i, (task_name, task_model) in enumerate(tasks.items()):
      if task_name not in valid_batches:
        continue
      batches = valid_batches[task_name]
      output_path = self.tests_path + '/%s_%s.best' % (task_name, mode) if not args.debug else None
      test_score, _ = task_model.test(batches, mode, self.logger, output_path)

    mode = 'test'
    test_batches = self.get_batch(mode)
    for i, (task_name, task_model) in enumerate(tasks.items()):
      if task_name not in test_batches:
        continue
      batches = test_batches[task_name]
      output_path = self.tests_path + '/%s_%s.best' % (task_name, mode) if not args.debug else None

      test_score, _ = task_model.test(batches, mode, self.logger, output_path)
    return m

  def demo(self, utterances):
    m = self.create_model(self.config, load_best=True)
    m.tasks.dialogue_en.demo(utterances)
    
  def debug(self):
    task_name = list(self.config.tasks.keys())[0]
    t = time.time()
    mode = 'train'
    batches = self.get_batch(mode)[task_name]
    #model = self.create_model(self.config)

    for i, batch in enumerate(batches):
      for k, v in flatten_recdict(batch).items():
        if isinstance(v, np.ndarray):
          print(k, v.shape)
        else:
          print(k, type(v))
          
      for j, ex in enumerate(flatten_batch(batch)):
        print('<%03d-%03d>' % (i,j))
        print_example(ex, self.vocab)
        #self.dataset[task_name].print_example(b, self.vocab)
        print('')
      exit(1)
    exit(1)

def main(args):
  tf_config = tf.ConfigProto(
    log_device_placement=False,
    allow_soft_placement=True, # GPU上で実行できない演算を自動でCPUに
    gpu_options=tf.GPUOptions(
      allow_growth=True, # True->必要になったら確保, False->全部
    )
  )

  with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:
    random.seed(0)
    tf.set_random_seed(0)
    manager = ExperimentManager(args, sess)
    if args.mode == 'demo':
      while True:
        utterance = input('> ')
        manager.demo([utterance])
    else:
      getattr(manager, args.mode)()




def get_parser():
  desc = ""
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('model_root_path', type=str, help ='')
  parser.add_argument('mode', type=str, help ='')
  parser.add_argument('-ct','--config_type', default='tmp', 
                      type=str, help ='')
  parser.add_argument('-cp','--config_path', default='configs/lol/main.conf',
                      type=str, help ='')
  parser.add_argument('-cl', '--cleanup', default=False,
                      type=str2bool, help ='')
  parser.add_argument('--debug', default=False,
                      type=str2bool, help ='')
  return parser

if __name__ == "__main__":
  # Common arguments are defined in base.py
  parser = get_parser()
  args = parser.parse_args()
  main(args)


# <memo>
# AutoEncoderでノイズを入れる（シャッフルやマスクをする）

# Pairwise ranking lossで応答選択
#  
