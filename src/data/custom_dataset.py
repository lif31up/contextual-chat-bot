import yaml
import torch
from torch.utils.data import Dataset
from typing import Callable
from nltk.stem.porter import PorterStemmer

from src.data.transform import tokenize, yml_to_dict, yml_to_xy, yml_to_tag


class BOWDataSet(Dataset):
  def __init__(self, yml_file: str, tokenizer: Callable, transform: Callable):
    with open(yml_file) as file: self.proto_data = yaml.safe_load(file)
    self.stemmer: PorterStemmer = PorterStemmer()
    self.transform, self.tokenizer = transform, lambda pattern: tokenizer(pattern, self.stemmer)
    self.dictionary = yml_to_dict(self.proto_data, tokenize)
    self.data = yml_to_xy(self.proto_data, transform, self.dictionary)
    self.tags = yml_to_tag(self.proto_data)
  # __init__()

  def __len__(self): return len(self.data)
  def __getitem__(self, index) -> tuple | None:
    bag, label = self.data[index]
    if not torch.is_tensor(bag) or len(label) is not len(self.tags):
      print(f"__getitem__: error at idx {index}.")
      return None
    return bag, label
# BOWDataSet