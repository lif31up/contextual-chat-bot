from torch import Dataset

def visualize_dataset(dataset: Dataset, data_range: tuple):
  start, end = data_range
  for idx in range(start, end):
    x, y = dataset.data[idx]
    print(f"x: {x} \n y: {y}")
# visualize_dataset