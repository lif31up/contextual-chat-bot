import yaml
import torch
from torch.utils.data import Dataset
from typing import Callable
from nltk.stem.porter import PorterStemmer

from src.data.transform import yml_to_dict, yml_to_xy, yml_to_tag


class BOWDataSet(Dataset):
  def __init__(self, yml_file: str, tokenizer: Callable, transform: Callable):
    with open(yml_file) as file: self.proto_data = yaml.safe_load(file)
    self.stemmer: PorterStemmer = PorterStemmer()
    self.tokenize = lambda string: tokenizer(string, self.stemmer)
    self.dictionary = yml_to_dict(self.proto_data, self.tokenize)
    self.transform = lambda string: transform(string, self.dictionary, self.tokenize)
    self.data = yml_to_xy(self.proto_data, self.transform)
    self.labels = yml_to_tag(self.proto_data)
  # __init__()

  def __len__(self): return len(self.data)
  def __getitem__(self, index) -> tuple | None:
    bag, label = self.data[index]
    if not torch.is_tensor(bag) or len(label) is not len(self.labels):
      print(f"__getitem__: error at idx {index}.")
      return None
    return bag, label
# BOWDataSet

def visualize_dataset(dataset: Dataset, data_range: tuple):
  start, end = data_range
  for idx in range(start, end):
    x, y = dataset.data[idx]
    print(f"x: {x} \n y: {y}")
# visualize_dataset