import argparse
import os
from collections import defaultdict
from pathlib import Path

import gdown
import h5py
import numpy as np
import pytorch_lightning as pl
import timm
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--epochs", default=1, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--pretrained", action="store_true")
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

seed = args.seed
lr = args.lr
epochs = args.epochs
pretrained = args.pretrained
debug = args.debug

name = "resnet18.a1_in1k"
savefile = f"{name}_yearbook_lr{lr}_ep{epochs}_seed{seed}"
savefile = f"{name}_pretrained_yearbook_lr{lr}_ep{epochs}_seed{seed}" if pretrained else savefile
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def run():
    pl.seed_everything(seed)
    wandb.init(
        project="yearbook",
        config={
            "lr": lr,
            "epochs": epochs,
            "name": name,
            "pretrained": pretrained,
            "seed": seed,
            "savefile": savefile,
        },
        mode="disabled" if debug else "online",
    )

    # Dataset
    data_dir = Path.home() / "opt/data/yearbook"
    transform = T.Compose([T.ToTensor(), T.Normalize([0.5709], [0.2742])])
    train_ds = YearbookDataset(split="train", data_dir=data_dir, transform=transform)
    val_ds = YearbookDataset(split="val", data_dir=data_dir, transform=transform)
    test_ds = YearbookDataset(split="test", data_dir=data_dir, transform=transform)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=False)

    print(f"Train length: \t{len(train_dl.dataset)}")
    print(f"Val length: \t{len(val_dl.dataset)}")
    print(f"Test length: \t{len(test_dl.dataset)}")

    # Train
    m = timm.create_model("resnet18.a1_in1k", pretrained=False, num_classes=2)
    m.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = torch.nn.Identity()
    m = m.to(DEVICE)

    optim = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs, eta_min=1e-3)

    for i in range(epochs):
        loss = train(m, optim, sched, train_dl)

        if i % 2 == 0:
            m.eval()
            acc_train = get_acc(m, train_dl)
            acc_val = get_acc(m, val_dl)

            wandb.log({"train/loss": loss})
            wandb.log({"train/accuracy": acc_train})
            wandb.log({"val/accuracy": acc_val})
            print(f"[epoch {i}] acc_train: {acc_train:.2f}, acc_val: {acc_val:.2f}")

    # Evaluate
    torch.save(m.state_dict(), Path.home() / f"opt/weights/yearbook/{savefile}.pt")
    acc_weight, accs = evaluate_groups(m, test_dl)
    wandb.log({"test/acc_weight": acc_weight})
    wandb.log({"test/acc_worst": min(accs.values())})
    print(f"[test] acc_weight: \t{acc_weight:.2f}")
    print(f"[test] acc_worst: \t{min(accs.values()):.2f}, group_worst: \t{min(accs, key=accs.get)}")
    print(accs)


class YearbookDataset(Dataset):
    def __init__(self, split, data_dir, transform=None, target_transform=None):
        self.transform = transform if transform else lambda x: x
        self.target_transform = target_transform if target_transform else lambda x: x
        self._maybe_download(data_dir)

        self.X = []
        self.y = []
        self.g = []
        with h5py.File(os.path.join(data_dir, "yearbook.hdf5"), "r") as f:
            if split in ["train", "val"]:
                split = "test" if split == "val" else split
                for year in range(1930, 1971):
                    self.X.append(np.array(f[str(year)][split]["images"]))
                    self.y.append(np.array(f[str(year)][split]["labels"]))
                    year_size = len(f[str(year)][split]["labels"])
                    self.g.append(np.array([year] * year_size))
            else:
                for year in range(1971, 2014):
                    self.X.append(np.array(f[str(year)]["train"]["images"]))
                    self.y.append(np.array(f[str(year)]["train"]["labels"]))
                    self.X.append(np.array(f[str(year)]["test"]["images"]))
                    self.y.append(np.array(f[str(year)]["test"]["labels"]))
                    year_size = len(f[str(year)]["train"]["labels"]) + len(f[str(year)]["test"]["labels"])
                    self.g.append(np.array([year] * year_size))
        self.X = np.vstack(self.X)
        self.y = np.hstack(self.y)
        self.g = np.hstack(self.g)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        return self.transform(x), self.target_transform(y), self.g[idx]

    def __len__(self):
        return len(self.y)

    def _maybe_download(self, destination_dir):
        destination_dir = Path(destination_dir)
        destination = destination_dir / "yearbook.hdf5"
        drive_id = "16lPT5DS3tz0XWnBuP8C-8zBnK0bwK7eX"
        if destination.exists():
            return
        destination_dir.mkdir(parents=True, exist_ok=True)
        gdown.download(
            url=f"https://drive.google.com/u/0/uc?id={drive_id}&export=download&confirm=pbef",
            output=str(destination),
            quiet=False,
        )


@torch.no_grad()
def get_acc(model, dl):
    model.eval()
    acc = []
    for x, y, _ in dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        acc.append(torch.argmax(model(x), dim=1) == y)
    acc = torch.cat(acc)
    acc = torch.sum(acc) / len(acc)
    return acc.item()


@torch.no_grad()
def evaluate_groups(model, dl):
    model.eval()
    preds, ys, gs = [], [], []

    pbar = tqdm(dl)
    for x, y, g in pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)
        prob = F.softmax(model(x), dim=1)
        _, pred = prob.data.max(1)
        preds.append(pred)
        ys.append(y)
        gs.append(g)
    preds = torch.cat(preds)
    ys = torch.cat(ys)
    gs = torch.cat(gs)

    gs_dict = defaultdict(lambda: [])
    for i in range(len(gs)):
        gs_dict[gs[i].item()].append(ys[i].item() == preds[i].item())

    accs_dict = defaultdict(lambda: float)
    weighted_acc = 0
    for g_id, g_preds in gs_dict.items():
        accs_dict[g_id] = sum(g_preds) / len(g_preds)
        weighted_acc += accs_dict[g_id] * len(g_preds)
    weighted_acc /= len(preds)

    return weighted_acc, accs_dict


def train(model, optim, sched, dl):
    model.train()
    loss_total = 0
    pbar = tqdm(dl)
    for itr, (x, y, g) in enumerate(pbar):
        x, y = x.to(DEVICE), y.to(DEVICE)

        loss = F.cross_entropy(model(x), y)
        loss_total += loss

        pbar.set_postfix_str(f"loss: {loss:.2f}")
        optim.zero_grad()
        loss.backward()
        optim.step()
    sched.step()

    return loss_total / len(dl)


if __name__ == "__main__":
    run()
