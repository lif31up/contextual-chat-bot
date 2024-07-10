from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from src.data.transform import tokenize, tokens_to_bag
from src.model.NeuralNet import NeuralNet
from src.data.BOWDataset import BOWDataSet

def main(path: str, save_to: str, iters=1000):
    trainset = BOWDataSet(path, tokenizer=tokenize, transform=tokens_to_bag)
    loader = DataLoader(dataset=trainset, batch_size=32, shuffle=True, num_workers=0)

    # hyper parameter
    n_inpt = len(trainset.dictionary)
    n_hidn = int(n_inpt * 1.2)
    n_oupt = len(trainset.labels)
    print(f'hidden nodes\' weight: {n_hidn}\niterations: {iters}')

    # read to train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNet(n_inpt, n_hidn, n_oupt).to(device=device)
    criterion = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)

    # tra!n
    for _ in tqdm(range(iters)):
        for x, y in loader:
            loss = criterion(model.forward(x), y)
            optim.zero_grad()
            loss.backward()
            optim.step()
    # for for
    print(f"loss: {loss.item():.4f}")

    # saving the model's parameters and the other data
    features = {
        "state": model.state_dict(),
        "inpt": n_inpt,
        "hidn": n_hidn,
        "oupt": n_oupt,
        "dictionary": trainset.dictionary,
        "labels": trainset.labels,
    }  # features
    torch.save(features, save_to)
# __main__

if __name__ == "__main__": main("../src/data/raw/trainset.yml", "../src/model/context_model.pth", iters=1000)