#coding: utf-8
from pprint import pprint
from collections import defaultdict, OrderedDict, Counter

from core.utils.common import read_json
from core.vocabulary.base import VocabularyBase

class JsonFileVocabulary(VocabularyBase):
  def __init__(self, config):
    '''
    All pretrained embeddings must be under the source_dir.'
    This class can merge two or more pretrained embeddings by concatenating both.
    For OOV word, this returns zero vector.

    '''
    super(JsonFileVocabulary, self).__init__(pad_token=config.pad_token,
                                             unk_token=config.unk_token)
    self.trainable = config.trainable
    self.tokenizer = lambda x: x
    self.vocab, self.rev_vocab, keys = self.init_vocab(config.vocab_path, 
                                                       config.vocab_size)
    self._key2id = {k:i+len(self.start_vocab) for i,k in enumerate(keys)}

  def key2id(self, key):
    return self._key2id.get(key, self.UNK_ID)

  def key2token(self, key):
    return self.id2token(self.key2id(key))

  def id2properties(self, _id):
    raise NotImplementedError

  def init_vocab(self, vocab_path, vocab_size):
    raise NotImplementedError

class ChampionVocabulary(JsonFileVocabulary):
  def init_vocab(self, vocab_path, vocab_size):
    data = read_json(vocab_path)
    keys = []
    rev_vocab = []
    for name, d in data.data.items():
      keys.append(int(d.key))
      rev_vocab.append(name)

    rev_vocab = self.start_vocab + rev_vocab
    vocab = OrderedDict()
    cnt = 0
    for t in rev_vocab:
      if not t in vocab:
        vocab[t] = cnt
        cnt += 1
    return vocab, rev_vocab, keys

