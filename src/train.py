from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from src.data.transform import tokenize, tokens_to_bag
from src.models.context_model import NeuralNet
from src.data.custom_dataset import BOWDataSet

def init_dataloader(file_path: str) -> tuple:
    trainset = BOWDataSet(file_path, tokenizer=tokenize, transform=tokens_to_bag)
    return DataLoader(dataset=trainset, batch_size=32, shuffle=True, num_workers=0), trainset
# init_dataloader()

def main(file_path: str):
    try: loader, trainset = init_dataloader(file_path)
    except Exception as e: return 1

    # hyper parameter
    iterations = 1000
    n_inpt = len(trainset.dictionary)
    n_hidn = int(n_inpt * 1.2)
    n_oupt = len(trainset.tags)
    print(f'hidden nodes\' weight: {n_hidn}\niterations: {iterations}')

    # read to train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNet(n_inpt, n_hidn, n_oupt).to(device=device)
    criterion = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)

    # tra!n
    for _ in tqdm(range(iterations)):
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
        "dict": trainset.dictionary,
        "tags": trainset.tags,
    }  # features
    torch.save(features, "./src/models/context_model.pth")

    return 0
# __main__
if __name__ == "__main__": main()