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
import wandb
from data import get_dataloaders
from model import vit_small_patch16_224
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size(), local_rank


def wsc_scheduler(base_lr, final_lr, total_steps, warm_steps=0, cool_steps=0):
    warm_schedule = np.array([])
    cool_schedule = np.array([])
    if warm_steps > 0:
        warm_schedule = np.linspace(0, base_lr, warm_steps)
    if cool_steps > 0:
        cool_schedule = np.linspace(base_lr, final_lr, cool_steps)
    stable_schedule = np.array([base_lr] * (total_steps - warm_steps - cool_steps))
    schedule = np.concatenate((warm_schedule, stable_schedule, cool_schedule))
    assert len(schedule) == total_steps
    return schedule


def mixup(x, y, num_classes, p=0.2):
    """https://github.com/google-research/big_vision/blob/main/big_vision/utils.py#L1146"""
    a = np.random.beta(p, p)
    a = max(a, 1 - a)  # ensure a >= 0.5 so that `unrolled x` is dominant
    mixed_x = a * x + (1 - a) * x.roll(1, dims=0)
    y_onehot = torch.zeros(y.size(0), num_classes, device=y.device)
    y_onehot.scatter_(1, y.unsqueeze(1), 1)  # one-hot encoding
    mixed_y = a * y_onehot + (1 - a) * y_onehot.roll(1, dims=0)
    return mixed_x, mixed_y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--warm_ratio", type=float, default=0.1)
    parser.add_argument("--cool_ratio", type=float, default=0.4)
    parser.add_argument("--mixup_p", type=float, default=0.2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dir_output", type=str, required=True)
    parser.add_argument("--dir_data", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.backends.cudnn.benchmark = True
    if rank == 0:
        os.makedirs(args.dir_output, exist_ok=True)
        wandb.init(project="imagenet1k")
        wandb.config.update(args)

    tr_loader, vl_loader, steps_per_epoch = get_dataloaders(
        dir_data=args.dir_data,
        batch_size_per_gpu=args.bs,
        num_workers=args.num_workers,
        pin_memory=True,
        distributed=True,
    )

    model = vit_small_patch16_224(num_classes=1000).cuda()
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    model = torch.compile(model) if args.compile else model
    criterion = nn.CrossEntropyLoss()

    total_steps = args.epochs * steps_per_epoch
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
    )
    scheduler = wsc_scheduler(
        base_lr=args.lr,
        final_lr=0,
        total_steps=total_steps,
        warm_steps=int(args.warm_ratio * total_steps),
        cool_steps=int(args.cool_ratio * total_steps),
    )

    best_acc = 0.0
    for epoch in range(args.epochs):
        t0 = time.time()

        # Train
        model.train()
        for step, (x, y) in enumerate(tr_loader):
            if step >= steps_per_epoch:
                break

            lr = scheduler[epoch * steps_per_epoch + step]
            for p in optimizer.param_groups:
                p["lr"] = lr

            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            x, y_soft = mixup(x, y, num_classes=1000, p=args.mixup_p)
            with autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = -torch.sum(
                    y_soft * torch.log_softmax(logits, dim=1), dim=1
                ).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            # decouple weight decay from learning rate
            # pytorch adamw: p -= lr * wd * p
            # true decouple: p -= wd * p
            with torch.no_grad():
                mult = lr / args.lr
                for n, p in model.named_parameters():
                    if ".bias" not in n and ".norm" not in n:
                        p.data.mul_(1 - args.wd * mult)

            if rank == 0 and step % args.log_interval == 0:
                print(f"ep {epoch} tr_loss {loss.item():.4f} lr {lr:.6f}")
                wandb.log({"train/loss": loss.item(), "train/lr": lr})

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
            wandb.log(
                {
                    "val/loss": vl_loss,
                    "val/acc1": vl_acc1,
                    "val/acc5": vl_acc5,
                    "epoch_time_min": (time.time() - t0) / 60,
                    "norm_l2": torch.sqrt(
                        sum(torch.sum(p**2) for p in model.parameters())
                    ).item(),
                }
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

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
