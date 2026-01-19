"""
Data loading for ImageNet-1k training.
"""

import os

import torch.distributed as dist
import torchvision.transforms as T
import webdataset as wds
from PIL import Image
from timm.data.auto_augment import rand_augment_transform
from torch.utils.data import DataLoader

IMAGENET_TRAIN_SAMPLES = 1_281_167


def get_dataloaders(
    dir_data,
    batch_size_per_gpu=256,
    num_workers=8,
    pin_memory=True,
    distributed=True,
):
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    steps_per_epoch = IMAGENET_TRAIN_SAMPLES // (batch_size_per_gpu * world_size)

    tr_transform = T.Compose(
        [
            T.RandomResizedCrop(224, scale=(0.05, 1.0)),
            T.RandomHorizontalFlip(),
            rand_augment_transform(
                config_str="rand-m10-n2",
                hparams={
                    "img_mean": (128, 128, 128),
                    "interpolation": Image.BILINEAR,
                },
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
            resampled=True,
            shardshuffle=True,
            nodesplitter=wds.split_by_node,
        )
        .shuffle(250_000)
        .decode("pil")
        .map(lambda x: (tr_transform(x["jpg"]), int(x["cls"])))
    )
    tr_loader = (
        wds.WebLoader(  # wraps DataLoader
            tr_dataset,
            batch_size=None,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True,
        )
        .unbatched()  # unbatch-shuffle-rebatch for cross-worker mixing
        .shuffle(5000)
        .batched(batch_size_per_gpu)
        .with_epoch(steps_per_epoch)
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
    vl_loader = DataLoader(
        vl_dataset,
        batch_size=batch_size_per_gpu,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
    )

    return tr_loader, vl_loader, steps_per_epoch
