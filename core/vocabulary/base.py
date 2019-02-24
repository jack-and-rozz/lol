#coding: utf-8
import os, time, re, sys, math
from collections import defaultdict, OrderedDict, Counter
from orderedset import OrderedSet
import core.utils.common as common
import numpy as np
#from sklearn.preprocessing import normalize
from nltk import word_tokenize

PAD_ID = 0  # PAD_ID must be 0 for sequence length counting.
_DIGIT_RE = re.compile(r"\d")

def word_tokenizer(lowercase=False, normalize_digits=False, split_quotation=False,
                   use_nltk_tokenizer=False):
  '''
  Args:
     - flatten: Not to be used (used only in char_tokenizer)
  '''
  def _tokenizer(sent, flatten=None): # Arg 'flatten' is not used in this func. 
    if type(sent) != str:
      sent = ' '.join(sent) 
    if split_quotation:
      sent = sent.replace("'", " ' ")
    if normalize_digits:
      sent = re.sub(_DIGIT_RE, "0", sent) 
    if lowercase:
      sent = sent.lower()
    sent = sent.replace('\n', '')
    sent = sent.split() if not use_nltk_tokenizer else word_tokenize(sent)
    return [w for w in sent if w]
  return _tokenizer

def char_tokenizer(lowercase=False, normalize_digits=False, split_quotation=False,
                   use_nltk_tokenizer=False):
  def _tokenizer(sent, flatten=False):
    if split_quotation:
      sent = sent.replace("'", " ' ")
    if normalize_digits:
      sent = re.sub(_DIGIT_RE, "0", sent) 
    if lowercase:
      sent = sent.lower()
    def word2chars(word):
      return [c for c in word]
    words = sent.replace('\n', '')
    words = words.split() if not use_nltk_tokenizer else word_tokenize(words)
    chars = [word2chars(w) for w in words]
    if flatten:
      chars = common.flatten(chars)
    return chars
  return _tokenizer

def random_embedding_generator(embedding_size):
  return lambda: np.random.uniform(-math.sqrt(3), math.sqrt(3), 
                                   size=embedding_size)
def zero_embedding_generator(embedding_size):
  return lambda: np.array([0.0 for _ in range(embedding_size)])

init_generator = random_embedding_generator

class VocabularyBase(object):
  def __init__(self, pad_token='<pad>', unk_token='<unk>'):
    self.vocab = None
    self.rev_vocab = None
    self.start_vocab = [pad_token, unk_token]
    self.PAD = pad_token
    self.UNK = unk_token

  @property
  def size(self):
    return len(self.vocab)

  @property
  def PAD_ID(self):
    return self.token2id(self.PAD)

  @property
  def UNK_ID(self):
    return self.token2id(self.UNK)

  def id2token(self, _id):
    if type(_id) not in [int, np.int32, np.int64]:
      sys.stderr.write(str(type(_id)))
      raise ValueError('Token ID must be an integer.')
    elif _id < 0 or _id > len(self.rev_vocab):
      return self.UNK
    elif _id == self.PAD_ID:
      return ''
    else:
      return self.rev_vocab[_id]

  def token2id(self, token):
    return self.vocab.get(token, self.vocab.get(self.UNK, None))

class WordVocabularyBase(VocabularyBase):
  def __init__(self, pad_token='<pad>', unk_token='<unk>', bos_token='<bos>'):
    super(WordVocabularyBase, self).__init__(pad_token=pad_token, 
                                             unk_token=unk_token)
    self.BOS = bos_token
    self.start_vocab = [pad_token, unk_token, bos_token]

  @property
  def BOS_ID(self):
    return self.token2id(self.BOS)

  def ids2tokens(self, ids, link_span=None):
    '''
    <Args>
    - ids: a list of word-ids.
    - link_span : a tuple of the indices between the start and the end of a link.
    <Return>
    - a string.
    '''
    def _ids2tokens(ids, link_span):
      sent_tokens = [self.id2token(word_id) for word_id in ids]
      if link_span:
        for i in range(link_span[0], link_span[1]+1):
          sent_tokens[i] = common.colored(sent_tokens[i], 'link')
      sent_tokens = [w for w in sent_tokens if w]
      return " ".join(sent_tokens)
    return _ids2tokens(ids, link_span)

  def get(self, token):
    return self.token2id(token)
  
  def tokens2ids(self, sentence, word_dropout=0.0):
    if type(sentence) == list:
      # If sentence is a list, each element must be a word.
      assert type(sentence[0]) == str
      sentence = " ".join(sentence)
    tokens = self.tokenizer(sentence) 
    if word_dropout:
      res = [self.token2id(word) if np.random.rand() <= word_dropout else self.vocab.get(self.UNK) for word in tokens]
    else:
      res = [self.token2id(word) for word in tokens]

    return res


class CharVocabularyBase(VocabularyBase):
  def tokens2ids(self, tokens):
    if type(tokens) == list:
      tokens = " ".join(tokens)
    tokens = self.tokenizer(tokens) 
    res = [[self.token2id(char) for char in word] for word in tokens]
    return res

  def ids2tokens(self, ids, link_span=None):
    sent_tokens = ["".join([self.id2token(char_id) for char_id in word]) 
                   for word in ids]
    if link_span:
      for i in range(link_span[0], link_span[1]+1):
        sent_tokens[i] = common.colored(sent_tokens[i], 'link')
      sent_tokens = [w for w in sent_tokens if w]
    return " ".join(sent_tokens)

class VocabularyWithEmbedding(WordVocabularyBase):
  def __init__(self, config):
    '''
    All pretrained embeddings must be under the source_dir.'
    This class can merge two or more pretrained embeddings by concatenating both.
    For OOV word, this returns zero vector.

    '''
    super(VocabularyWithEmbedding, self).__init__(pad_token=config.pad_token,
                                                  unk_token=config.unk_token,
                                                  bos_token=config.bos_token)
    
    self.trainable = config.trainable
    self.tokenizer = word_tokenizer(lowercase=config.lowercase,
                                    normalize_digits=config.normalize_digits,
                                    split_quotation=config.split_quotation)
    self.centralize_embedding = config.centralize_embedding
    self.normalize_embedding = config.normalize_embedding
    self.vocab, self.rev_vocab, self.embeddings = self.init_vocab(
      config.emb_config, config.vocab_size)

  def merge(self, pretrained, vocab_size, 
            vocab_merge_type='union', embedding_merge_type='first_found'):
    '''
    <Args>
    - pretrained: A list of dictionary, {word: vector}.
    - vocab_merge_type: ['union', 'intersection']
    - embedding_merge_type: ['first_found', 'average', 'concat']
    '''
    if vocab_merge_type != 'union':
      raise NotImplementedError

    if embedding_merge_type != 'first_found':
      raise NotImplementedError

    rev_vocab = list(OrderedSet(common.flatten([list(v.keys()) for v in pretrained]))) # union.
    embedding_size = len(pretrained[0][rev_vocab[0]])
    embedding_dict = defaultdict(zero_embedding_generator(embedding_size))
    for w in rev_vocab:
      embedding_dict[w] = [vecs[w] for vecs in pretrained if w in vecs][0]
    return embedding_dict

  @common.timewatch()
  def init_vocab(self, emb_configs, vocab_size):
    if type(emb_configs) != list:
      emb_configs = [emb_configs]


    # Load several pretrained embeddings and concatenate them.
    pretrained = [self.load_vocab(c.path, vocab_size, c.size, c.skip_first) for c in emb_configs]
    pretrained = self.merge(pretrained, vocab_size)
    rev_vocab = self.start_vocab + list(pretrained.keys())
    rev_vocab = list(OrderedSet(rev_vocab))

    vocab = OrderedDict()
    
    cnt = 0
    for t in rev_vocab:
      if not t in vocab:
        vocab[t] = cnt
        cnt += 1

    # Merge pretrained embeddings.
    embeddings = np.array([pretrained[w] for w in vocab], dtype=np.float32)


    if self.centralize_embedding:
      center = np.mean(embeddings, axis=0)
      embeddings -= center

    if self.normalize_embedding:
      # Normalize the pretrained embeddings for each of the embedding types.
      embeddings = np.array([common.normalize_vector(v) for v in embeddings])


    #embeddings = np.array(embeddings)
    sys.stderr.write("Done loading word embeddings.\n")
    return vocab, rev_vocab, embeddings

  def load_vocab(self, embedding_path, vocab_size, embedding_size, skip_first):
    """
    Load pretrained vocabularies.
    Args:
    - embedding_path: a string.
    - vocab_size: an integer.
    - embedding_size: an integer.
    - skip_first: a boolean.
    - token_list: list of tokens (word, char, label, etc.).
    """
    sys.stderr.write("Loading word embeddings from {}...\n".format(embedding_path))
    embedding_dict = defaultdict(init_generator(embedding_size))

    with open(embedding_path) as f:
      for i, line in enumerate(f.readlines()):
        if skip_first and i == 0:
          continue
        if len(embedding_dict) >= vocab_size:
          break

        word_and_emb = line.split()
        word = self.tokenizer(word_and_emb[0])
        if len(word) != 1:
          sys.stderr.write('A token must have no space:\n' + line + '\n')
          sys.stderr.write('Current token: %s\n' % common.colored(' '.join(word), 'underline'))

          continue
        word = word[0]
        vector = [float(s) for s in word_and_emb[1:]]
        if len(vector) != embedding_size:
          #sys.stderr.write('Embedding parse error:\n' + line + '\n')
          continue

        # If a capitalized (or digit-normalized) word is changed to its lowercased form, which is used as an alternative only when the exact one is not registered. 
        # e.g. Texas -> texas, 1999->0000, etc.
        if word == word_and_emb[0]:
          embedding_dict[word] = vector
        elif word not in embedding_dict: 
          embedding_dict[word] = vector
    return embedding_dict

class PredefinedCharVocab(CharVocabularyBase):
  def __init__(self, config):
    super(PredefinedCharVocab, self).__init__(pad_token=config.pad_token,
                                              unk_token=config.unk_token)
    # Lowercasing and normalizing digits shouldn't be done as for char-base inputs.
    self.tokenizer = char_tokenizer(lowercase=False,
                                    normalize_digits=False,
                                    split_quotation=config.split_quotation)
    self.vocab, self.rev_vocab = self.init_vocab(config.vocab_path, 
                                                 config.vocab_size)

  def init_vocab(self, vocab_paths, vocab_size):
    rev_vocab = []
    if type(vocab_paths) != list:
      vocab_paths = [vocab_paths]

    for vocab_path in vocab_paths:
      with open(vocab_path) as f:
        rev_vocab += [l.split()[0] for i, l in enumerate(f) if i < vocab_size]
    sys.stderr.write("Done loading the vocabulary.\n")
    rev_vocab = self.start_vocab + rev_vocab
    rev_vocab = list(OrderedSet(rev_vocab))
    vocab = OrderedDict({t:i for i,t in enumerate(rev_vocab)})
    return vocab, rev_vocab

class FeatureVocabulary(VocabularyBase):
  '''
  A class to manage transformation between feature tokens and their ids.
  '''
  def __init__(self, all_tokens, pad_token='<pad>', unk_token='<unk>'):
    super(FeatureVocabulary, self).__init__(pad_token=pad_token, 
                                            unk_token=unk_token)

    counter = Counter(all_tokens)
    self.freq = counter.values
    self.rev_vocab = self.start_vocab + list(counter.keys())
    self.vocab = OrderedDict([(t, i) for i,t in enumerate(self.rev_vocab)])

  def __str__(self):
    return str(self.vocab)

  def tokens2ids(self, tokens):
    return [self.token2id(t) for t in tokens]

  def ids2tokens(self, ids):
    return [self.id2token(_id) for _id in ids]
 

WordVocabulary = FeatureVocabulary
