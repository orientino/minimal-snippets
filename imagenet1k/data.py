"""
Data loading and augmentation for ImageNet-1k training.
Follows big_vision's vit_s16_i1k.py configuration.
"""

import numpy as np
import torch
import torch.distributed as dist
from datasets import load_dataset
from timm.data.auto_augment import rand_augment_transform
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms


class ImageNetDataset(Dataset):
    """Wrapper for HuggingFace ImageNet dataset with transforms."""

    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]
        label = sample["label"]

        # Convert grayscale to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_tr_transforms():
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                224,
                scale=(0.05, 1.0),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            rand_augment_transform(
                config_str="rand-m2-n10",
                hparams={"translate_const": 100, "img_mean": (128, 128, 128)},
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def get_vl_transforms():
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def mixup_data(x, y, alpha=0.5):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def create_dataloaders(
    batch_size_per_gpu=256,
    num_workers=8,
    pin_memory=True,
    distributed=True,
):
    tr_dataset = ImageNetDataset(
        load_dataset("ILSVRC/imagenet-1k", split="train[:99%]"), get_tr_transforms()
    )
    vl_dataset = ImageNetDataset(
        load_dataset("ILSVRC/imagenet-1k", split="validation"), get_vl_transforms()
    )

    if distributed:
        tr_sampler = DistributedSampler(
            tr_dataset,
            num_replicas=dist.get_world_size() if dist.is_initialized() else 1,
            rank=dist.get_rank() if dist.is_initialized() else 0,
            shuffle=True,
            seed=42,
        )
        vl_sampler = DistributedSampler(
            vl_dataset,
            num_replicas=dist.get_world_size() if dist.is_initialized() else 1,
            rank=dist.get_rank() if dist.is_initialized() else 0,
            shuffle=False,
        )
    else:
        tr_sampler = None
        vl_sampler = None

    tr_loader = DataLoader(
        tr_dataset,
        batch_size=batch_size_per_gpu,
        sampler=tr_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    vl_loader = DataLoader(
        vl_dataset,
        batch_size=batch_size_per_gpu,
        sampler=vl_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )

    return tr_loader, vl_loader, tr_sampler, vl_sampler
