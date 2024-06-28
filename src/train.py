from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

# custom importing
from src.models.context_model import NeuralNet
from src.data.transform import BOWDataSet

train_set = BOWDataSet("../src/data/raw/trainset.yml")
loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True, num_workers=0)

iterations = 1000
n_inpt_params = len(train_set.dictionary)
n_hidn_params = 1000
n_oupt_params = len(train_set.tags)
print(f'input parameters: {n_inpt_params}\nhidden parameters: {n_hidn_params}\noutput parameters: {n_oupt_params}\niterations: {iterations}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet(n_inpt_params, n_hidn_params, n_oupt_params).to(device=device)
criterion = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

for x, y in tqdm(loader):
    loss = criterion(model.forward(x), y)
    optim.zero_grad()
    loss.backward()
    optim.step()

print(f"loss: {loss.item():.4f}")

features = {
    "state": model.state_dict(),
    "inpt": n_inpt_params,
    "hidn": n_hidn_params,
    "oupt": n_oupt_params,
    "dict": train_set.dictionary,
    "tags": train_set.tags,
}  # features

torch.save(features, "../src/models/context_model.pth")
