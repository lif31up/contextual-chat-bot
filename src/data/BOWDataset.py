from torch.utils.data import Dataset
from src.data.transform import yml_get_dataset, yml_get_tag, Transformer

class BOWDataSet(Dataset):
  def __init__(self, yml_data):
    self.transformer = Transformer(yml_data)
    self.dataset = yml_get_dataset(yml_data, self.transformer.tokenize)
    self.labels = yml_get_tag(yml_data)
  # __init__()

  def __len__(self): return len(self.dataset)

  def __getitem__(self, index):
    bag, label = self.dataset[index]
    if self.transformer: bag = self.transformer.bag(bag)
    return bag, label
# BOWDataSet