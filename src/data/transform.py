import torch
from typing import Callable

def tokenize(string: str, stemmer) -> list: return [stemmer.stem(token) for token in string.split(' ')]

def tokens_to_bag(string: str, dictionary: list, tokenize: Callable) -> torch.tensor:
  if not dictionary:
    print("dictionary is not valid.")
    return None
  bag = torch.zeros(len(dictionary))
  for index, word in enumerate(dictionary):
    if word in tokenize(string): bag[index] = 1.
  # for if
  return bag
# token_to_bag():

def yml_to_dict(data: list, tokenize: Callable) -> list:
    dictionary = []
    for intent in data:
      for word in intent['patterns']:
        for token in tokenize(word):
          dictionary.append(token)
    # for for
    return list(set(dictionary))
# yml_to_dict():

def yml_to_xy(data: list, transform: Callable)->list:
  xy = list()
  for index, intent in enumerate(data):
    for pattern in intent['patterns']:
      y = torch.zeros(len(data))
      y[index] = 1.
      xy.append((transform(pattern), y))
  # for for
  return xy
# yml_to_item()

def yml_to_tag(yml_data: list) -> list: return [intent['tag'] for intent in yml_data]
