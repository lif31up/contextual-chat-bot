import torch
from torch.utils.data import Dataset
import yaml
from src.transform import Transformer

def PatternYml(path: str = "./data/raw/trainset.yml"):
  with open(path, "r") as file: intents = yaml.safe_load(file)
  dataset, classes, n_classes = list(), list(), len(intents)
  for intent in intents:
    tag, patterns = intent['tag'], intent['patterns']
    classes.append(tag)
    for pattern in patterns:
      label = torch.zeros(n_classes)
      label[classes.index(tag)] = 1.
      dataset.append((pattern, label))
  return BOWDataSet(dataset, classes)
# yml_decoder()

class BOWDataSet(Dataset):
  def __init__(self, dataset, classes):
    self.dataset = dataset
    self.classes = classes
    self.transform = Transformer()
  # __init__()
  def __len__(self): return len(self.dataset)
  def __getitem__(self, index):
    feature, label = self.dataset[index]
    if len(self.transform.transforms) > 0: feature = self.transform(feature)
    return feature, label
# BOWDataSet