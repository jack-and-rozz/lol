# coding: utf-8
import pandas as pd
from pprint import pprint
import os, re, sys, random, copy, time
import itertools
import numpy as np
from collections import OrderedDict, defaultdict, Counter

from core.utils.constants import TEAMID, QUEUEID
from core.utils.common import dotDict, recDotDefaultDict, recDotDict, flatten, flatten_recdict, batching_dicts, dbgprint, read_jsonlines, timewatch
from core.dataset.base import DatasetBase, PartitionedDatasetBase
from core.dataset.base import padding as _padding

def reformat_role(role, lane):
  if lane == 'BOTTOM':
    if role == 'DUO_CARRY':
      return 'ADC'
    elif role == 'DUO_SUPPORT':
      return 'SUPPORT'
    else:
      return '<unk>'
  else:
    return lane
  pass

# https://developer.riotgames.com/api-methods/#match-v4
def create_example(jline, vocab):
  # assert type(context) == list # The context must be separated into tokens.
  #example = recDotDefaultDict()
  # example.context.raw = context
  # example.context.word = vocab.encoder.word.tokens2ids(context)
  # if vocab.encoder.char:
  #   example.context.char = vocab.encoder.char.tokens2ids(context)
  # if response:
  #   example.response.raw = response
  #   example.response.word = vocab.decoder.word.tokens2ids(response)
  example = recDotDefaultDict()

  example.gameVersion = '.'.join(jline.gameVersion.split('.')[:2])
  #example.gameMode = jline.gameMode
  example.queueID = int(jline.queueId)

  # 100 for blue side. 200 for red side.
  blue = 0 if jline.teams[0].teamId == 100 else 1 
  red = 1 - blue

  example.win = 0 if jline.teams[blue].win == 'Win' else 1
  #example.bans.order = [x.pickTurn for x in jline.teams[blue].bans] +[x.pickTurn for x in jline.teams[red].bans] 

  example.bans.ids = [vocab.champion.key2id(int(x.championId)) for x in jline.teams[blue].bans] + [vocab.champion.key2id(int(x.championId)) for x in jline.teams[red].bans]
  example.bans.raw = [vocab.champion.key2token(int(x.championId)) for x in jline.teams[blue].bans] + [vocab.champion.key2token(int(x.championId)) for x in jline.teams[red].bans]

  # participantsは必ずblueからになってる？
  example.teams = [0 if x.teamId == TEAMID.blue else 1 for x in jline.participants]
  
  example.picks.ids = [vocab.champion.key2id(int(x.championId)) for x in jline.participants]
  example.picks.raw = [vocab.champion.key2token(int(x.championId)) for x in jline.participants]
  example.roles.raw = [reformat_role(x.timeline.role, x.timeline.lane) for x in jline.participants]
  example.roles.ids = [vocab.role.token2id(x) for x in example.roles.raw]
  return example

def padding(batch, minlen=None, maxlen=None):
  batch.roles.ids = _padding(
    batch.roles.ids, minlen=[None], maxlen=[None])
  batch.picks.ids = _padding(
    batch.picks.ids, minlen=[None], maxlen=[None])
  batch.bans.ids = _padding(
    batch.bans.ids, minlen=[None], maxlen=[None])
  # for k, v in flatten_recdict(batch).items():
  #   if type(v) == np.ndarray:
  #     print(k ,v.shape)
  return batch

def create_test_batch(contexts, vocab):
  raise NotImplementedError

def print_example(example):
  #pprint(example)
  
  def _format(team, role, pick, ban):
    return [team, role, pick, ban]

  lines = []
  for i in range(0, 5):
    win = True if example.win == 0 else False
    role = example.roles.raw[i]
    pick = example.picks.raw[i]
    ban = example.bans.raw[i]
    lines.append(['BLUE', role, pick, ban, win])

  for i in range(5, 10):
    win = True if example.win == 1 else False
    role = example.roles.raw[i]
    pick = example.picks.raw[i]
    ban = example.bans.raw[i]
    lines.append(['RED', role, pick, ban, win])

  header = ['Team', 'Role', 'Pick', 'Ban', 'Win']
  df = pd.DataFrame(lines)
  df.columns = header
  df = df.set_index('Team')
  print(df)
  print('Patch:\t', example.gameVersion)

class _LoLmatchTrainDataset(PartitionedDatasetBase):
  def preprocess(self, line):
    return line
  
  def filter_example(self, example):
    # Return True if the example is acceptable.
    if example.queueId != QUEUEID.solo_ranked:
      return False
    return True

  @timewatch()
  def load_data(self, source_path, max_rows):
    sys.stdout.write("Loading dataset from \'%s\'... \n" % source_path)
    data = read_jsonlines(source_path, max_rows=max_rows)
    data = [self.preprocess(d) for d in data]
    self.data = self.create_examples(data)

  def create_examples(self, data):
    examples = [create_example(d, self.vocab) for d in data if self.filter_example(d)]
    #examples = [create_example(d, self.vocab) for d in data]
    #examples = [ex for ex in examples if self.filter_example(ex)]
    return examples

  def padding(self, batch, minlen=None, maxlen=None):
    return padding(batch, minlen, maxlen)

class _LoLmatchTestDataset(_LoLmatchTrainDataset):
  pass


class LoLmatchDataset(DatasetBase):
  train_class = _LoLmatchTrainDataset
  test_class = _LoLmatchTestDataset
