import argparse
from collections import defaultdict
from pathlib import Path

import pytorch_lightning as pl
import timm
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from wilds import get_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--epochs", default=20, type=int)
parser.add_argument("--seed", default=42, type=int)
args = parser.parse_args()

name = "resnet50.a1_in1k"
savefile = f"{name}_pretrained_water_lr{args.lr}_ep{args.epochs}_seed{args.seed}"


def run():
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pl.seed_everything(args.seed)

    # Dataset
    scale = 256.0 / 224.0
    target_resolution = (224, 224)
    transform_train = T.Compose(
        [
            T.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2,
            ),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    transform_test = T.Compose(
        [
            T.Resize(
                (int(target_resolution[0] * scale), int(target_resolution[1] * scale))
            ),
            T.CenterCrop(target_resolution),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    data_dir = Path.home() / "opt/data/waterbirds_v1.0"
    data = get_dataset("waterbirds", root_dir=data_dir, download=True)
    train_ds = data.get_subset("train", transform=transform_train)
    val_ds = data.get_subset("val", transform=transform_test)
    test_ds = data.get_subset("test", transform=transform_test)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

    # Train
    m = timm.create_model(
        name,
        pretrained=True,
        num_classes=data._n_classes,
        drop_rate=0.1,
        img_size=224,
    )
    m = m.to(DEVICE)

    optim = SGD(m.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
    sched = CosineAnnealingLR(optim, T_max=args.epochs, eta_min=1e-4)
    best_ep, best_acc = 0, 0.0
    for i in range(args.epochs):
        m.train()
        loss_total = 0
        pbar = tqdm(train_dl)
        for itr, (x, _, y) in enumerate(pbar):
            x, y = x.to(DEVICE), y.to(DEVICE)

            loss = F.cross_entropy(m(x), y)
            loss_total += loss

            pbar.set_postfix_str(f"loss: {loss:.2f}")
            optim.zero_grad()
            loss.backward()
            optim.step()
            sched.step()

        m.eval()
        acc_train = evaluate_groups(m, train_dl, DEVICE)
        acc_val = evaluate_groups(m, val_dl, DEVICE)
        print(f"[epoch {i}] acc_train: {acc_train:.2f}, acc_val: {acc_val:.2f}")

        if acc_val[1] > best_acc:
            best_ep = i
            best_acc = acc_val[1]
            torch.save(m.state_dict(), f"{savefile}.pt")

    # Evaluate
    m.load_state_dict(torch.load(f"{savefile}.pt"))
    torch.save(m.state_dict(), f"{savefile}.pt")
    acc_weight, acc_worst = evaluate_groups(m, test_dl, DEVICE)
    print(f"[test] best epoch checkpoint: {best_ep}")
    print(f"[test] acc_weight: {acc_weight:.2f}, acc_worst: {acc_worst:.2f}")


@torch.no_grad()
def evaluate_groups(model, dl, device):
    model.eval()
    preds, ys, gs = [], [], [], []

    for x, y, g in dl:
        x, y, g = x.to(device), y.to(device), g.to(device)
        g = g[:, 0]  # for wilds, first metadata index is group metadata

        pred = model(x)
        _, pred = pred.data.max(1)

        preds.append(pred)
        ys.append(y)
        gs.append(g)
    preds = torch.cat(preds)
    ys = torch.cat(ys)
    gs = torch.cat(gs)
    gs = 2 * ys + gs

    groups = defaultdict(list)
    for g, y, pred in zip(gs, ys, preds):
        groups[g.item()].append(y.item() == pred.item())

    weighted_acc = 0
    accuracies = []
    for _, group_preds in groups.items():
        accuracy = sum(group_preds) / len(group_preds)
        accuracies.append(accuracy)
        weighted_acc += accuracy * len(group_preds)
    weighted_acc /= len(preds)

    return weighted_acc, min(accuracies)


if __name__ == "__main__":
    run()
