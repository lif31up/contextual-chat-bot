from nltk import PorterStemmer
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from src.model.NeuralNet import NeuralNet
from src.BOWDataset import PatternYml
from src.transform import get_stem, get_bagging

def main(path: str, save_to: str, iters=500):
  dataset = PatternYml(path)

  porter_stemmer = PorterStemmer()
  stem = get_stem(porter_stemmer) # create stemmer

  dictionary = list()
  for feature, label in dataset:
    token = stem(feature)
    for word in token: dictionary.append(word)
  dictionary = list(set(dictionary))
  bagging = get_bagging(dictionary) # create bagger
  dataset.transform.Compose([stem, bagging])

  # read to train
  n_hidn = 10
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = NeuralNet(len(dictionary), n_hidn, len(dataset.classes)).to(device=device)
  criterion = torch.nn.CrossEntropyLoss()
  optim = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

  # tra!n
  total_loss = 0.
  for _ in tqdm(range(iters)):
    for feature, label in DataLoader(dataset, batch_size=1, shuffle=True):
      loss = criterion(model.forward(feature), label)
      optim.zero_grad()
      loss.backward()
      optim.step()
      total_loss += loss.item()
  print(f"loss: {total_loss / len(dataset):.4f}")

  # saving the weights
  feature = {
    "state": model.state_dict(),
    "labels": dataset.classes,
    "n_hidn": n_hidn,
    "dictionary": dictionary
  } # feature
  torch.save(feature, save_to)
# __main__

if __name__ == "__main__": main("../data/raw/trainset.yml", "./model/model.pth")