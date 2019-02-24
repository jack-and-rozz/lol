# coding: utf-8
import json
from core.utils.common import flatten
#############################################
#        CoreNLP
#############################################

def get_parser():
  '''
  **deprecated**
  Run stanford CoreNLP. This requires less setups but slow. Use CoreNLP Server instead.
  '''
  import corenlp, json
  corenlp_dir = os.environ['CORENLP']
  properties_file = os.path.join(corenlp_dir, 'user.properties')
  parser = corenlp.StanfordCoreNLP(corenlp_path=corenlp_dir, properties=properties_file)
  def _parse(text):
    result = json.loads(parser.parse(text))
    sentences = [sent for sent in result['sentences'] if sent['words']]
    paragraph = [" ".join([w[0] for w in sent['words']]).encode('utf-8') for sent in sentences]
    return paragraph
  return _parse

def connect_to_corenlp_server(host='http://localhost', port=9000):
  from stanfordcorenlp import StanfordCoreNLP
  return StanfordCoreNLP(host, port)


def fix_brackets(sent):
  LRB = '-LRB-' # (
  RRB = '-RRB-' # )
  LSB = '-LSB-' # [
  RSB = '-RSB-' # ]
  LCB = '-LCB-' # {
  RCB = '-RCB-' # }
  return sent.replace(LRB, '(').replace(RRB, ')').replace(LSB, '[').replace(RSB, ']').replace(LCB, '{').replace(RCB, '}')

def run_corenlp(text, _corenlp, props={'annotators': 'tokenize,ssplit', 'tokenize.options': 'untokenizable=allDelete'}):
  '''
  How to pass options to CoreNLP: 
  https://stanfordnlp.github.io/CoreNLP/tokenize.html
  '''
  #text = text.replace('%', '% ') # to espace it from percent-encoding.
  result = json.loads(_corenlp.annotate(text, props))
  sentences = [' '.join([tokens['word'] for tokens in sent['tokens'] if tokens['word']]) for sent in result['sentences']]
  sentences = [fix_brackets(sent).split() for sent in sentences]  
  return sentences

def setup_tokenizer(tokenizer_type=None):
  assert tokenizer_type is None or tokenizer_type in ['corenlp', 'nltk']
  if tokenizer_type == 'corenlp':
    #from core.utils.tokenizer import connect_to_corenlp_server, run_corenlp
    corenlp = connect_to_corenlp_server(host='http://localhost', port=9000)
    tokenizer = lambda uttr: flatten(run_corenlp(uttr, corenlp))
  elif tokenizer_type == 'nltk':
    from nltk import word_tokenize
    tokenizer = word_tokenize
  else:
    tokenizer = lambda uttr: uttr.split()

  # tokenizer must return a list of words.
  return tokenizer


