from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

# custom importing
from src.data.transform import tokenize, tokens_to_bag
from src.models.context_model import NeuralNet
from src.data.custom_dataset import BOWDataSet

trainset = BOWDataSet("../src/data/raw/trainset.yml", tokenizer=tokenize, transform=tokens_to_bag)
loader = DataLoader(dataset=trainset, batch_size=32, shuffle=True, num_workers=0)

iterations = 1000
n_inpt = len(trainset.dictionary)
n_hidn = int(n_inpt * 1.2)
n_oupt = len(trainset.tags)
print(f'input parameters: {n_inpt}\nhidden parameters: {n_hidn}\noutput parameters: {n_oupt}\niterations: {iterations}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet(n_inpt, n_hidn, n_oupt).to(device=device)
criterion = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)

for _ in tqdm(range(iterations)):
    for x, y in loader:
        loss = criterion(model.forward(x), y)
        optim.zero_grad()
        loss.backward()
        optim.step()
# for for

print(f"loss: {loss.item():.4f}")

features = {
    "state": model.state_dict(),
    "inpt": n_inpt,
    "hidn": n_hidn,
    "oupt": n_oupt,
    "dict": trainset.dictionary,
    "tags": trainset.tags,
}  # features

torch.save(features, "../src/models/context_model.pth")
