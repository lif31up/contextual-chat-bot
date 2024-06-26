import torch

from src.data.transform import tokens_to_bag
from src.models.context_model import NeuralNet

data = torch.load("../src/models/context_model.pth")
state = data["state"]
n_inpt, n_hidn, n_oupt = data["inpt"], data["hidn"], data["oupt"]
dictionary, tags = data["dict"], data["tags"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet(n_inpt, n_hidn, n_oupt).to(device)
model.load_state_dict(state)
model.eval()

"""
while 1:
  pattern = input("input: ")
  bag = tokens_to_bag(pattern, tokenize)
  output = model(bag)
  probs = torch.softmax(output, dim=-1)
  print(output)
  print(probs)
# while
"""