from nltk import PorterStemmer
from src.transform import *
from src.model.NeuralNet import NeuralNet

def main(path: str):
  data = torch.load(path)
  state = data["state"]
  labels, dictionary, n_hidn = data["labels"], data["dictionary"], data["n_hidn"]

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = NeuralNet(len(dictionary), n_hidn, len(labels)).to(device)
  model.load_state_dict(state)
  model.eval()

  porter_stemmer = PorterStemmer()
  stem = get_stem(porter_stemmer)  # create stemmer

  bagging = get_bagging(dictionary) # create bagger
  transformer = Transformer().Compose([stem, bagging])

  while 1:
    pattern = input("input> ").strip()
    pattern = transformer(pattern)
    probs = model.forward(pattern)
    probs = torch.softmax(probs, dim=0)
    pred = torch.argmax(probs, dim=0)
    if probs[pred] >= 0.65: print(f"pred: {labels[pred.item()]}")
    else: print(f"idk")
  # while
# __main__

if __name__ == "__main__": main("./model/model.pth")