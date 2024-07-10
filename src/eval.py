from nltk.stem.porter import PorterStemmer

from src.data.transform import *
from src.model.NeuralNet import NeuralNet

def main(path: str):
  data = torch.load(path)
  state = data["state"]
  n_inpt, n_hidn, n_oupt = data["inpt"], data["hidn"], data["oupt"]
  dictionary, labels = data["dictionary"], data["labels"]

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = NeuralNet(n_inpt, n_hidn, n_oupt).to(device)
  model.load_state_dict(state)
  model.eval()

  stemmer, stemmer_tokenize = PorterStemmer(), lambda string: tokenize(string, stemmer)
  while 1:
    pattern = input("input> ")
    if pattern.strip() == "exit": break
    bag = tokens_to_bag(pattern, dictionary, stemmer_tokenize)
    output = model(bag)
    predict = torch.argmax(torch.softmax(output, dim=0))
    print(f"context: {labels[predict]}")
  # while
# __main__

if __name__ == "__main__": main("model/context_model.pth")