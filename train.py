import sys
import numpy as np
import alive_progress
import torch
from torch.utils.data import DataLoader

from model import NeuralNet
from mod import tokenize, bag_of_words
from loader import yaml_loader

try: filename = sys.argv[-1]
except: exit()

trainset = yaml_loader(filename=filename)
loader = DataLoader(dataset=trainset, batch_size=1, shuffle=True, num_workers=0)

# Hyper Prameter
iterations = 1000
n_inpt_parms = trainset.data_size
n_hidn_parms = int(len(trainset.dictionary) * 1.2)
n_oupt_parms = trainset.n_tags

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet(n_inpt_parms, n_hidn_parms, n_oupt_parms).to(device=device)
criterion = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)

progress_bar = alive_progress.alive_it(range(iterations))
for epoch in progress_bar:
    for x, y in loader:
        loss = criterion(model.forward(x), y)
        optim.zero_grad()
        loss.backward()
        optim.step()
# for for
print(f"loss: {loss.item():.4f}")

features = {
    "state" : model.state_dict(),
    "inpt" : n_inpt_parms,
    "hidn" : n_hidn_parms,
    "oupt" : n_oupt_parms,
    "dict" : trainset.dictionary,
    "tags" : trainset.tags
}  # features

save_path = "./model.pth"
torch.save(features, save_path)
