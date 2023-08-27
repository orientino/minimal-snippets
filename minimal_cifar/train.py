import argparse
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import CIFAR10
from torchmetrics import Accuracy
import pytorch_lightning as pl


parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--epochs", default=200, type=int)
args = parser.parse_args()


def main():
    pl.seed_everything(args.seed)
    model = LitResNet18()
    dm = CIFAR10DataModule(batch_size=128)
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator="gpu")
    dm.setup("fit")
    trainer.fit(model, dm)
    dm.setup("test")
    trainer.test(model, dm)


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128):
        super().__init__()
        self.batch_size = batch_size
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
        ])
                    
    def setup(self, stage=None):
        if stage == "fit" or stage == None:
            train_set = CIFAR10(root=".", train=True, download=True, transform=self.train_transform)
            self.train_set, self.val_set = random_split(train_set, [0.8, 0.2])
        if stage == "test":
            self.test_set = CIFAR10(root=".", train=False, download=True, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=2)


class LitResNet18(pl.LightningModule):
    def __init__(self, n_classes=10):
        super().__init__()
        self.model = models.resnet18(pretrained=False, num_classes=n_classes)
        self.model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = torch.nn.Identity()
        self.accuracy = Accuracy(task="multiclass", num_classes=n_classes)
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=5e-4)
        return [optimizer], [scheduler]
    
    def forward(self, batch):
        X, y = batch
        logits = self.model(X)
        y_hat = torch.argmax(logits, dim=1)
        return F.cross_entropy(logits, y), self.accuracy(y_hat, y)

    def training_step(self, batch, _):
        loss, acc = self(batch)
        self.log("train/acc", acc, prog_bar=True)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss, acc = self(batch)
        self.log("val/acc", acc, prog_bar=True)
        self.log("val/loss", loss, prog_bar=True)
        
    def test_step(self, batch, _):
        loss, acc = self(batch)
        self.log("test/acc", acc, prog_bar=True)
        self.log("test/loss", loss, prog_bar=True)


if __name__ == "__main__":
    main()
