import torch

class Transformer:
  def __init__(self): self.transforms = []
  def Compose(self, transforms):
    self.transforms = transforms
    return self
  # Compose

  def __call__(self, x):
    for transform in self.transforms:
      x = transform(x)
    return x
  # __call__()
# Transformer

def _stem(string: str, stemmer): return [stemmer.stem(token) for token in string.split(' ')]
def get_stem(stemmer):
  def stem(string): return _stem(string, stemmer=stemmer)
  return stem
# get_stem

def _bagging(tokens: list, dictionary: list):
  bag = torch.zeros(len(dictionary))
  for index, word in enumerate(dictionary):
    if word in tokens: bag[index] = 1.
  return bag
# bagging()
def get_bagging(dictionary):
  def bagging(string):
    return _bagging(string, dictionary)
  return bagging
# get_bagging