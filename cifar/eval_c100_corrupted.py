import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

data_dir = Path.home() / "opt/data/cifar/cifar-100-c"
severity = 1
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def run():
    m = models.resnet18(weights=None, num_classes=100)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m = m.to(DEVICE)
    m.load_state_dict(torch.load("resnet18_c100.pt"))

    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])]
    )
    corruptions = [
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "snow",
        "frost",
        "fog",
        "brightness",
        "contrast",
        "elastic_transform",
        "pixelate",
        "jpeg_compression",
    ]
    accs = {}
    for c in corruptions:
        dt = CIFAR100C(data_dir, c, severity, test_transform)
        dl = DataLoader(dt, batch_size=1000, shuffle=False, num_workers=2)
        accs[c] = get_acc(m, dl)
    print(accs)


class CIFAR100C(Dataset):
    def __init__(self, dir, corruption, severity, transform=None, label_transform=None):
        self.transform = transform
        self.label_transform = label_transform

        data_path = os.path.join(dir, corruption + ".npy")
        label_path = os.path.join(dir, "labels.npy")
        self.data = np.load(data_path)
        self.labels = np.load(label_path)
        self.offset = (severity - 1) * 10000

        assert severity > 0 and severity < 6

    def __getitem__(self, index):
        imgs, lbls = self.data[index + self.offset], self.labels[index + self.offset]
        if self.transform is not None:
            imgs = self.transform(imgs)
        if self.label_transform is not None:
            lbls = self.target_transform(lbls)

        return imgs, lbls

    def __len__(self):
        return len(self.data) // 5


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
