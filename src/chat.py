from nltk.stem.porter import PorterStemmer
import yaml

from src.data.transform import *
from src.models.context_model import NeuralNet

def main(model_path: str, responses_path: str):
  with open(responses_path) as file: responses = yaml.safe_load(file)

  data = torch.load(model_path)
  state = data["state"]
  n_inpt, n_hidn, n_oupt = data["inpt"], data["hidn"], data["oupt"]
  dictionary, labels = data["dictionary"], data["labels"]

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = NeuralNet(n_inpt, n_hidn, n_oupt).to(device)
  model.load_state_dict(state)
  model.eval()

  stemmer = PorterStemmer()
  while 1:
    pattern = input("input> ")
    if pattern.strip() == "exit": break;
    bag = tokens_to_bag(pattern, dictionary, lambda string: tokenize(string, stemmer))
    output = model(bag)
    predict = torch.argmax(torch.softmax(output, dim=0))
    print(f"bot: {responses[predict]['patterns'][0]}")
  # while

  return 0
# __main__
if __name__ == "__main__": main("../src/models/context_model.pth", "../src/data/raw/responses.yml")