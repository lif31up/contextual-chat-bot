import torch
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
def tokenize(input): return [stemmer.stem(token) for token in input.split(' ')]
def yml_get_dictionary(yml_data, transform=tokenize):
  dictionary = []
  for intent in yml_data:
    for word in intent['patterns']:
      for token in transform(word): dictionary.append(token)
  # for for
  return list(set(dictionary))
# get_dictionary()

def yml_get_dataset(yml_data, transform):
  xy = list()
  for index, intent in enumerate(yml_data):
    for pattern in intent['patterns']:
      y = torch.zeros(len(yml_data))
      y[index] = 1.
      xy.append((transform(pattern), y))
  # for for
  return xy
# yml_to_item()

def yml_get_tag(yml_data): return [intent['tag'] for intent in yml_data]

class Transformer:
  def __init__(self, yml_data):
    self.dictionary = yml_get_dictionary(yml_data, self.tokenize)
  # __init__()

  def bag(self, input):
    if not self.dictionary: print("error on transform.py: dictionary is None.")
    bag, token = torch.zeros(len(self.dictionary)), self.tokenize(input)
    for index, word in enumerate(self.dictionary):
      if word in token: bag[index] = 1.
    # for if
    return bag
  # bag
# class
