import sys
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from src.models.context_model import NeuralNet
from src.data.yaml import YamlLoader

trainset = YamlLoader(filename="../data/raw/trainset.yaml")
loader = DataLoader(dataset=trainset, batch_size=1, shuffle=True, num_workers=0)

iterations = 10000
n_inpt_parms = trainset.data_size
n_hidn_parms = int(len(trainset.dictionary) * 1.2)
n_oupt_parms = trainset.n_tags

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet(n_inpt_parms, n_hidn_parms, n_oupt_parms).to(device=device)
criterion = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)

for x, y in tqdm(loader):
    loss = criterion(model.forward(x), y)
    optim.zero_grad()
    loss.backward()
    optim.step()

print(f"loss: {loss.item():.4f}")

features = {
    "state": model.state_dict(),
    "inpt": n_inpt_parms,
    "hidn": n_hidn_parms,
    "oupt": n_oupt_parms,
    "dict": trainset.dictionary,
    "tags": trainset.tags,
}  # features

torch.save(features, "../data/processed/model.pth")
