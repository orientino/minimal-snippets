"""
Data loading for ImageNet-1k training.
"""

import os

import torch.distributed as dist
import torchvision.transforms as T
import webdataset as wds
from timm.data.auto_augment import rand_augment_transform
from torch.utils.data import DataLoader

IMAGENET_TRAIN_SAMPLES = 1281167


def get_dataloaders(
    dir_data,
    batch_size_per_gpu=256,
    num_workers=8,
    pin_memory=True,
    distributed=True,
):
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    tr_transform = T.Compose(
        [
            T.RandomResizedCrop(224, scale=(0.05, 1.0)),
            T.RandomHorizontalFlip(),
            rand_augment_transform(
                config_str="rand-m10-n2",
                hparams={"translate_const": 100, "img_mean": (128, 128, 128)},
            ),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    vl_transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    tr_dataset = (
        wds.WebDataset(
            os.path.join(dir_data, "imagenet1k-train-{0000..1023}.tar"),
            shardshuffle=True,
            nodesplitter=wds.split_by_node,
        )
        .shuffle(5000)
        .decode("pil")
        .map(lambda x: (tr_transform(x["jpg"]), int(x["cls"])))
    )
    vl_dataset = (
        wds.WebDataset(
            os.path.join(dir_data, "imagenet1k-validation-{00..63}.tar"),
            shardshuffle=False,
            nodesplitter=wds.split_by_node,
        )
        .decode("pil")
        .map(lambda x: (vl_transform(x["jpg"]), int(x["cls"])))
    )
    tr_loader = DataLoader(
        tr_dataset,
        batch_size=batch_size_per_gpu,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
        drop_last=True,
    )
    vl_loader = DataLoader(
        vl_dataset,
        batch_size=batch_size_per_gpu,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
    )

    steps_per_epoch = IMAGENET_TRAIN_SAMPLES // (batch_size_per_gpu * world_size)
    return tr_loader, vl_loader, steps_per_epoch
