from collections import defaultdict
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import os
import pathlib
import sys
import utils


def distance(x,y):
    return torch.pow(torch.nn.functional.pairwise_distance(x, y), 2)


class tfidfData(Dataset):
    def __init__(self, data_list, labels):
        self.data = data_list
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def get(self, idx):
        data = []
        for id in idx:
            sample = self.data[id]
            data.append(torch.sparse.FloatTensor(torch.from_numpy(sample.indices).unsqueeze(0).long(),
                                     torch.from_numpy(sample.data).float(), [self.data.shape[1]]).to_dense())
        return torch.stack(data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]
        return torch.sparse.FloatTensor(torch.from_numpy(sample.indices).unsqueeze(0).long(),
                                 torch.from_numpy(sample.data).float(), [self.data.shape[1]]).to_dense(), self.labels[idx]

class Reshaper(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return torch.reshape(x, self.dim)


class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.Dropout(0.001),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.fc(x)


def getRandomSimSample(dictionary, cuisine):
    cui = []
    for c in cuisine:
        cui.append(random.sample(dictionary[c], 1)[0])
    return cui

def getRandomDisimSample(dictionary, cuisine):
    cui = []
    for c in cuisine:
        b = random.randint(0, 28000)
        while b in dictionary[c]:
            b = random.randint(0,28000)
        cui.append(b)
    return cui

if __name__ == '__main__':
    cwd = pathlib.Path(os.getcwd())

    torch.multiprocessing.freeze_support()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    hyperparams = {"batch_size": 20,
                   "num_epochs": 200,
                   "lr": 0.001,
                   "wd": 0.0001,
                   "fold": 1,
                   "ckpt": int(sys.argv[1]) if len(sys.argv) > 1 else None}

    # Load Training and Validation Data
    train, train_label, val, val_label = utils.loadFoldData(hyperparams["fold"])

    train_label_dict = {}
    for uni in np.unique(train_label):
        train_label_dict[uni] = np.nonzero(train_label == uni)[0].tolist()

    val_label_dict = {}
    for uni in np.unique(val_label):
        val_label_dict[uni] = np.nonzero(val_label == uni)[0].tolist()

    train = [" ".join(t).lower() for t in train]
    tfidf = TfidfVectorizer(binary=True)
    train_ft = tfidf.fit_transform(train).astype('float16')
    train_data = tfidfData(train_ft, train_label)
    train_loader = DataLoader(train_data, batch_size=hyperparams["batch_size"],
                                shuffle=True, num_workers=hyperparams["batch_size"])

    val = [" ".join(t).lower() for t in val]
    val_ft = tfidf.transform(val).astype('float16')
    val_data = tfidfData(val_ft, val_label)
    val_loader = DataLoader(val_data, batch_size=hyperparams["batch_size"],
                                shuffle=True, num_workers=hyperparams["batch_size"])

    # Define model, loss and optimizer
    model = MLP(train_ft.shape[1])
    criterion = nn.TripletMarginLoss(margin=1.0)
    # criterion = 0.5*(nn.MSELoss() + nn.CrossEntropyLoss())
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["wd"])


    # Load from previous checkpoint
    ckpt_path = cwd / "ae_ckpt"
    ckpt_path.mkdir(exist_ok=True)
    if hyperparams["ckpt"]:
        checkpoint = torch.load(str(ckpt_path / str(hyperparams["ckpt"])))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Loading Checkpoint", hyperparams["ckpt"])

    # Move model and optimiser to CUDA if available
    model.to(device)
    if torch.cuda.is_available():
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

    for epoch in range(hyperparams["num_epochs"]):
        modes = {"Train": train_loader, "Validation": val_loader}
        label_dicts = {"Train": train_label_dict, "Validation": val_label_dict}
        losses = {"Train": 0.0, "Validation": 0.0}
        for mode in modes.keys():
            if mode == "Train":
                model.train()
                optimizer.zero_grad()
            else:
                model.eval()
            for data, label in modes[mode]:
                data = data.to(device)
                output = model(data)
                sim = getRandomSimSample(label_dicts[mode], label)
                sim = train_data.get(sim).to(device)
                sim = model(sim)
                dis = getRandomDisimSample(label_dicts[mode], label)
                dis = train_data.get(dis).to(device)
                dis = model(dis)
                loss = criterion(output, sim, dis)

                if mode == "Train":
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                losses[mode] += loss.item()

            losses[mode] = losses[mode] / len(modes[mode])
            print("{0} Loss for epoch {1}: {2:.10f}".format(
                mode,
                epoch + (hyperparams["ckpt"] if hyperparams["ckpt"] else 0)+1,
                losses[mode])
            )

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_loss': losses["Train"],
            'validation_loss': losses["Validation"],
        }, str(ckpt_path / str(epoch + 1 + (hyperparams["ckpt"] if hyperparams["ckpt"] else 0))))