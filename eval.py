import sys
import yaml
import torch
from model import NeuralNet
from mode import bag_of_words, tokenize

try: filename = sys.argv[-1]
except: exit()

data = torch.load(filename)
state = data["state"]
n_inpt_parms, n_hidn_parms, n_oupt_parms = data["inpt"], data["hidn"], data["oupt"]
dictionary, tags = data["dict"],  data["tags"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(n_inpt_parms, n_hidn_parms, n_oupt_parms).to(device)
model.load_state_dict(state)
model.eval()

while 1:
    cmd = input("me> ")
    if cmd == "exit": exit()
    sentence = tokenize(cmd)
    bag = bag_of_words(cmd, dictionary)
    bag = torch.from_numpy( bag.reshape(1, bag.shape[0]) )
    output = model(bag)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75: print(f"hiana> {tag}", end="\n\n")
    else: print(f"hiana: idk", end="\n\n")
# while
