import torch.nn as nn
class Model(nn.Module):

  def __init__(self, vocab_size, embedding_dim = 32, hidden_dim = 1024, context_size = 2):
    super(Model, self).__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.hidden_layer1 = nn.Linear(embedding_dim * context_size, hidden_dim)
    self.relu1 = nn.ReLU()
    self.hidden_layer2 = nn.Linear(hidden_dim, hidden_dim)
    self.relu2 = nn.ReLU()
    self.classifier = nn.Linear(hidden_dim, vocab_size)
    self.softmax = nn.Softmax()

  def forward(self, x):
    x = self.embedding(x)
    x = x.view(x.size(0), -1)
    x = self.relu1(self.hidden_layer1(x))
    x = self.relu2(self.hidden_layer2(x))
    x = self.classifier(x) ## logits

    # x = self.softmax(self.classifier(x))

    return x

import re

def clean_text(text, pattern = r'[^a-zA-Z0-9 \.\n]'):
  text = text.lower()
  text = text.replace("<bos>", "BOSTOKEN")
  text = re.sub(pattern, '', text)
  text = text.replace("BOSTOKEN", "<bos>")
  return text

class Tokenizer:

  def __init__(self, vocab_size = 1e6, pattern = r'[^a-zA-Z0-9 \.\n]') -> None:
    self.vocab_size = vocab_size
    self.pattern = pattern
    self.word_to_idx = {}
    self.idx_to_word = {}
    self.word_freq = {}

  def build_vocab(self, text):
    text = clean_text(text, self.pattern)
    text = text.lower()
    words = re.findall(r'\w+|\n|\.', text)
    unique_words = list(set(words))

    for idx, word in enumerate(unique_words):
      self.word_freq[word] = words.count(word)

    sorted_words = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)
    top_words = sorted_words[:int(self.vocab_size)]

    for idx, (word, _) in enumerate(top_words):

      self.word_to_idx[word] = idx
      self.idx_to_word[idx] = word

    self.word_to_idx['<UNK>'] = len(self.word_to_idx)
    self.idx_to_word[len(self.idx_to_word)] = '<UNK>'

  def encode(self, text):
    text = clean_text(text, self.pattern)
    words = re.findall(r'<bos>|\w+|\n|\.', text)
    encoded_text = []
    for word in words:
      if word in self.word_to_idx:
        encoded_text.append(self.word_to_idx[word])
      else:
        encoded_text.append(self.word_to_idx['<UNK>'])
    return encoded_text

  def decode(self, idxs):
    words = []
    for idx in idxs:
      if idx in self.idx_to_word:
        words.append(self.idx_to_word[idx])
      else:
        words.append('<UNK>')

    return ' '.join(words)

context_size = 10
batch_size = 128
import re

def clean_text(text, pattern = r'[^a-zA-Z0-9 \.\n]'):
  text = text.lower()
  text = text.replace("<bos>", "BOSTOKEN")
  text = re.sub(pattern, '', text)
  text = text.replace("BOSTOKEN", "<bos>")
  return text

pattern = r"""
    <bos>|<eos>|<unk>|         
    //.*|                      
    /\*[\s\S]*?\*/|           
    \#.*|         
    "(?:\\.|[^"\\])*"|         
    '(?:\\.|[^'\\])*'|
    \$\{?[A-Za-z_][A-Za-z0-9_]*\}?|  
    [A-Za-z_][A-Za-z0-9_]*|
    [0-9]+(?:\.[0-9]+)?|       
    [|&;><!~^%*/=+\-]=?|      
    [(){}\[\],.]|             
    [A-Za-z0-9_\-./]+|         
    \n                        
"""

class Tokenizer_linux:

  def __init__(self, vocab_size = 2 ** 16) -> None:
    self.vocab_size = vocab_size
    self.word_to_idx = {}
    self.idx_to_word = {}
    self.word_freq = {}

  def build_vocab(self, text):
    # text = clean_text(text, self.pattern)
    text = text.lower()

    words = re.findall(pattern, text, flags=re.VERBOSE)
    unique_words = list(set(words))
    # print(pattern)
    print(len(text))
    print(len(words))
    for idx, word in enumerate(unique_words):
      self.word_freq[word] = words.count(word)

    sorted_words = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)
    top_words = sorted_words[:int(self.vocab_size)]

    for idx, (word, _) in enumerate(top_words):

      self.word_to_idx[word] = idx
      self.idx_to_word[idx] = word

    self.word_to_idx['<UNK>'] = len(self.word_to_idx)
    self.idx_to_word[len(self.idx_to_word)] = '<UNK>'

  def encode(self, text):
    # text = clean_text(text, self.pattern)
    words = re.findall(pattern, text, flags=re.VERBOSE)
    encoded_text = []
    for word in words:
      if word in self.word_to_idx:
        encoded_text.append(self.word_to_idx[word])
      else:
        encoded_text.append(self.word_to_idx['<UNK>'])
    return encoded_text

  def decode(self, idxs):
    words = []
    for idx in idxs:
      if idx in self.idx_to_word:
        words.append(self.idx_to_word[idx])
      else:
        words.append('<UNK>')

    return ' '.join(words)