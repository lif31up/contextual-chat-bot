from nltk.stem.porter import PorterStemmer
import yaml
import numpy as np

stemmer: PorterStemmer = PorterStemmer()

def tokenize(raw) -> list: return [stemmer.stem(raw) for element in raw.split(' ')]
def bag(pattern, dictionary) -> np.ndarray:
  bag_of_word = np.zeros(len(dictionary), dtype=np.float32)
  for index, element in enumerate(dictionary):
    if element in pattern: bag_of_word[index] = 1
  # for
  return bag_of_word
# bag()
def transform(raw, dictionary) -> np.ndarray:
  bag_of_word = np.zeros(len(dictionary), dtype=np.float32)
  for index, element in enumerate(dictionary):
    if element in tokenize(raw): bag_of_word[index] = 1
  # for
  return bag_of_word
# transform()

class Dictionary:
  dictionary, tags, data = list(), list(), None
  def __init__(self, path):
    if path:
      self.get_data(path)

  def get_data(self, path):
    with open(path) as file:
      self.data = yaml.load(file, Loader=yaml.FullLoader)
    # with

    for intent in self.data:
      tag = intent["tag"]
      self.tags.append(tag)
      for raw in intent["patterns"]:
        pattern = tokenize(raw)
        self.dictionary.extend(pattern)
    # for
  # get_data()
  def transform:
# Dictionary