"""
Training script for ViT-S/16 on ImageNet-1k with DDP.
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from data import create_dataloaders, mixup_criterion, mixup_data
from model import vit_small_patch16_224
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )

    return rank, world_size, local_rank


def cleanup_distributed():
    dist.destroy_process_group()


def cosine_scheduler(base_lr, final_lr, epochs, steps_per_epoch, warmup_epochs=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * steps_per_epoch
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(0, base_lr, warmup_iters)

    iters = np.arange(epochs * steps_per_epoch - warmup_iters)
    schedule = final_lr + 0.5 * (base_lr - final_lr) * (
        1 + np.cos(np.pi * iters / len(iters))
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * steps_per_epoch
    return schedule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--mixup_prob", type=float, default=0.2)
    parser.add_argument("--mixup_alpha", type=float, default=0.5)
    parser.add_argument("--dir_output", type=str, required=True)
    parser.add_argument("--dir_data", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=50)
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()

    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed(42 + rank)
    torch.backends.cudnn.benchmark = True

    tr_loader, vl_loader, steps_per_epoch = create_dataloaders(
        dir_data=args.dir_data,
        batch_size_per_gpu=args.bs,
        num_workers=args.num_workers,
        pin_memory=True,
        distributed=True,
    )

    model = vit_small_patch16_224(num_classes=1000).cuda()
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
        betas=(0.9, 0.999),
    )
    scheduler = cosine_scheduler(
        base_lr=args.lr,
        final_lr=0,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
    )

    if rank == 0:
        os.makedirs(args.dir_output, exist_ok=True)
        print(f"Starting training on {world_size} GPUs")
        print(f"Batch size per GPU: {args.bs}")
        print(f"Total batch size: {args.bs * world_size}")
        print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        print(f"Steps per epoch: {steps_per_epoch}")

    best_acc = 0.0
    for epoch in range(args.epochs):
        t0 = time.time()

        # Train
        model.train()
        tr_loss, tr_n = 0, 0
        for step, (x, y) in enumerate(tr_loader):
            if step >= steps_per_epoch:
                break

            lr = scheduler[epoch * steps_per_epoch + step]
            for p in optimizer.param_groups:
                p["lr"] = lr

            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            if torch.rand(1).item() < args.mixup_prob:
                x, y_a, y_b, lam = mixup_data(x, y, alpha=args.mixup_alpha)
                with autocast("cuda", dtype=torch.bfloat16):
                    loss = mixup_criterion(criterion, model(x), y_a, y_b, lam)
            else:
                with autocast("cuda", dtype=torch.bfloat16):
                    loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            tr_loss += loss.item() * x.size(0)
            tr_n += x.size(0)
            if rank == 0 and step % args.log_interval == 0:
                print(f"ep {epoch} step {step} loss {loss.item():.4f}")
        metrics = torch.tensor([tr_loss, tr_n], device="cuda")
        dist.all_reduce(metrics)
        tr_loss = metrics[0].item() / metrics[1].item()

        # Validate
        model.eval()
        vl_loss, vl_correct1, vl_correct5, vl_n = 0, 0, 0, 0
        with torch.no_grad():
            for x, y in vl_loader:
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                with autocast("cuda", dtype=torch.bfloat16):
                    out = model(x)
                vl_loss += criterion(out, y).item() * x.size(0)
                top5 = out.topk(5, dim=1)[1]
                vl_correct1 += top5[:, 0].eq(y).sum().item()
                vl_correct5 += top5.eq(y.view(-1, 1)).sum().item()
                vl_n += x.size(0)
        metrics = torch.tensor([vl_loss, vl_correct1, vl_correct5, vl_n], device="cuda")
        dist.all_reduce(metrics)
        vl_loss, vl_acc1, vl_acc5 = (
            metrics[0].item() / metrics[3].item(),
            metrics[1].item() / metrics[3].item() * 100,
            metrics[2].item() / metrics[3].item() * 100,
        )

        # Logging and checkpointing
        if rank == 0:
            print(
                f"ep {epoch} "
                f"vl_loss {vl_loss:.4f} vl_acc@1 {vl_acc1:.2f} vl_acc@5 {vl_acc5:.2f} "
                f"time {(time.time() - t0) / 60:.1f}m"
            )
            ckpt = {
                "epoch": epoch,
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": args,
            }
            torch.save(ckpt, os.path.join(args.dir_output, "last.pth"))
            if vl_acc1 > best_acc:
                best_acc = vl_acc1
                torch.save(ckpt, os.path.join(args.dir_output, "best.pth"))

    cleanup_distributed()


if __name__ == "__main__":
    main()
