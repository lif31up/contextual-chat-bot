import torch
from typing import Callable

def tokenize(raw_string: str, stemmer) -> list: return [stemmer.stem(token) for token in raw_string.split(' ')]

def tokens_to_bag(raw_string: str, dictionary: list, tokenizer: Callable) -> torch.tensor:
  if not dictionary:
    print("dictionary is not valid.")
    return None
  bag = torch.zeros(len(dictionary))
  for index, word in enumerate(dictionary):
    if word in tokenizer(raw_string): bag[index] = 1.
  # for if
  return bag
# token_to_bag():

def yml_to_dict(yml_data: list, tokenizer: Callable) -> list:
    dictionary = []
    for intent in yml_data:
      for word in intent['patterns']:
        for token in tokenizer(word):
          dictionary.append(token)
    # for for
    return list(set(dictionary))
# yml_to_dict():

def yml_to_xy(yml_data: list, transform: Callable, dictionary: list)->list:
  xy = list()
  for index, intent in enumerate(yml_data):
    for pattern in intent['patterns']:
      y = torch.zeros(len(yml_data))
      y[index] = 1.
      xy.append((transform(pattern, dictionary), y))
  # for for
  return xy
# yml_to_item()

def yml_to_tag(yml_data: list) -> list: return [intent['tag'] for intent in yml_data]
