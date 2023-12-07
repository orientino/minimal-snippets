import argparse
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import CIFAR100
import pytorch_lightning as pl
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--epochs", default=160, type=int)
parser.add_argument("--seed", default=42, type=int)
args = parser.parse_args()

lr = args.lr
epochs = args.epochs
seed = args.seed
pl.seed_everything(seed)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def run():
    # Dataset
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])
    ])
    train_set = CIFAR100(root=".", train=True, download=True, transform=train_transform)
    train_set, val_set = random_split(train_set, [0.8, 0.2])
    test_set = CIFAR100(root=".", train=False, download=True, transform=test_transform)

    train_dl = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=2)
    test_dl = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)

    # Model
    m = models.resnet18(weights=None, num_classes=100)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m = m.to(DEVICE)

    optim = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs, eta_min=5e-4)
    
    # Train
    best_acc = 0.0
    for i in range(int(epochs)):
        m.train()
        loss_total = 0
        pbar = tqdm(train_dl)
        for itr, (x, y) in enumerate(pbar):
            x, y = x.to(DEVICE), y.to(DEVICE)

            loss = F.cross_entropy(m(x), y)
            loss_total += loss
            
            pbar.set_postfix_str(f"loss: {loss:.2f}")
            optim.zero_grad()
            loss.backward()
            optim.step()
        sched.step()

        m.eval()
        acc_val = get_acc(m, val_dl)
        if acc_val > best_acc:
            torch.save(m.state_dict(), "model.pt")
        print(f"[epoch {i}] acc_val: {acc_val:.4f}")

    m.load_state_dict(torch.load("model.pt"))
    print(f"[test] acc_test: {get_acc(m, test_dl):.4f}")


@torch.no_grad()
def get_acc(model, dl):
    acc = []
    for x, y in dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        acc.append(torch.argmax(model(x), dim=1) == y)
    acc = torch.cat(acc)
    acc = torch.sum(acc) / len(acc)

    return acc.item()


if __name__ == "__main__":
    run()

