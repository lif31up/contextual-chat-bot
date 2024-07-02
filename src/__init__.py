"""

import torch

from src.data.transform import tokens_to_bag
from src.models.context_model import NeuralNet

data = torch.load("../src/models/context_model.pth")
state = data["state"]
n_inpt_parms, n_hidn_parms, n_oupt_parms = data["inpt"], data["hidn"], data["oupt"]
dictionary, tags = data["dict"], data["tags"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet(n_inpt_parms, n_hidn_parms, n_oupt_parms).to(device)
model.load_state_dict(state)
model.eval()

def get_input():
  pattern = input("input: ")
  return pattern if pattern != "exit" else exit(0)
# get_input():

def __init__():
  while 1:
    pattern = get_input()
    bag = tokens_to_bag(pattern, dictionary)
    output = model(bag)
    probs = torch.softmax(output, dim=-1)
    print(output)
    print(probs)
  # while
# __init__

"""