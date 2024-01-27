import sys
import torch
from src.models.context_model import NeuralNet
from data.tokenize import *

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

while 1:
  input = get_input()
  sentence = tokenize(input)
  bag = bag_of_words(input, dictionary)
  bag = torch.from_numpy(bag.reshape(1, bag.shape[0]))
  output = model(bag)
  _, predicted = torch.max(output, dim=1)
  tag = tags[predicted.item()]
  probs = torch.softmax(output, dim=1)
  prob = probs[0][predicted.item()]
  if prob.item() > 0.45:
    print(f"output: {tag}", end="\n")
  else:
    print(f"output: idk", end="\n")
# while
