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


def cosine_scheduler(
    base_lr, final_lr, epochs, niter_per_ep, warmup_epochs=0, start_warmup_lr=0
):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_lr, base_lr, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_lr + 0.5 * (base_lr - final_lr) * (
        1 + np.cos(np.pi * iters / len(iters))
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def validate(model, vl_loader, criterion, rank):
    model.eval()

    total_loss = 0
    total_correct_top1 = 0
    total_correct_top5 = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in vl_loader:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            with autocast("cuda", dtype=torch.bfloat16):
                outputs = model(images)
                loss = criterion(outputs, labels)

            _, pred_top5 = outputs.topk(5, dim=1, largest=True, sorted=True)
            pred_top1 = pred_top5[:, 0]

            correct_top1 = pred_top1.eq(labels).sum().item()
            correct_top5 = (
                pred_top5.eq(labels.view(-1, 1).expand_as(pred_top5)).sum().item()
            )

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_correct_top1 += correct_top1
            total_correct_top5 += correct_top5
            total_samples += batch_size

    metrics = torch.tensor(
        [total_loss, total_correct_top1, total_correct_top5, total_samples],
        dtype=torch.float32,
        device="cuda",
    )
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

    avg_loss = metrics[0].item() / metrics[3].item()
    top1_acc = metrics[1].item() / metrics[3].item() * 100
    top5_acc = metrics[2].item() / metrics[3].item() * 100

    return avg_loss, top1_acc, top5_acc


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
    parser.add_argument("--dir_output", type=str, default="./checkpoints")
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
        niter_per_ep=steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
    )

    if rank == 0:
        print(f"Starting training on {world_size} GPUs")
        print(f"Batch size per GPU: {args.bs}")
        print(f"Total batch size: {args.bs * world_size}")
        print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"\n{'=' * 80}")
        print("Starting training")
        print(f"{'=' * 80}\n")

    # Training loop
    # -------------------------------------------------------

    best_acc = 0.0
    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        model.train()
        total_loss = 0
        total_samples = 0
        for step, (images, labels) in enumerate(tr_loader):
            if step >= steps_per_epoch:
                break

            lr = scheduler[epoch * steps_per_epoch + step]
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            if torch.rand(1).item() < args.mixup_prob:
                images, labels_a, labels_b, lam = mixup_data(
                    images, labels, alpha=args.mixup_alpha
                )
                with autocast("cuda", dtype=torch.bfloat16):
                    loss = mixup_criterion(
                        criterion, model(images), labels_a, labels_b, lam
                    )
            else:
                with autocast("cuda", dtype=torch.bfloat16):
                    loss = criterion(model(images), labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            if rank == 0 and step % args.log_interval == 0:
                print(
                    f"Epoch [{epoch}] Step [{step}/{steps_per_epoch}] Loss: {loss.item():.4f} LR: {lr:.6f}"
                )

        tr_loss = total_loss / total_samples
        vl_loss, top1_acc, top5_acc = validate(model, vl_loader, criterion, rank)

        epoch_time = time.time() - epoch_start_time

        if rank == 0:
            print(f"\n{'=' * 80}")
            print(f"Epoch [{epoch}/{args.epochs}] Summary:")
            print(f"  Train Loss: {tr_loss:.4f}")
            print(f"  Val Loss:   {vl_loss:.4f}")
            print(f"  Top-1 Acc:  {top1_acc:.2f}%")
            print(f"  Top-5 Acc:  {top5_acc:.2f}%")
            print(f"  Time: {epoch_time / 60:.1f} min")
            print(f"{'=' * 80}\n")

            os.makedirs(args.dir_output, exist_ok=True)
            checkpoint_state = {
                "epoch": epoch,
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": args,
            }

            torch.save(
                checkpoint_state, os.path.join(args.dir_output, "checkpoint_last.pth")
            )
            if top1_acc > best_acc:
                best_acc = top1_acc
                torch.save(
                    checkpoint_state,
                    os.path.join(args.dir_output, "checkpoint_best.pth"),
                )

    cleanup_distributed()


if __name__ == "__main__":
    main()
