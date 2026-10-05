"""
Training script for GPT on FineWeb-Edu. Features:
- DDP training
- adam or muon
- gradient accumulation
- independent weight decay
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import wandb
from torch import nn
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, Muon

from .data import data_generator, get_files
from .model import gpt_small


def print0(*args, **kwargs):
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_scheduler(total_steps, warm_steps=0, final_value=0.0):
    cool_steps = np.arange(total_steps - warm_steps)
    warm_schedule = np.linspace(0, 1, warm_steps + 1)[1:]
    cool_schedule = final_value + 0.5 * (1 - final_value) * (
        1 + np.cos(np.pi * cool_steps / len(cool_steps))
    )
    schedule = np.concatenate((warm_schedule, cool_schedule))
    assert len(schedule) == total_steps
    return schedule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mbs", type=int, default=256)  # local batch size per GPU
    parser.add_argument("--tbs", type=int, default=524288)  # 0.5M tr tokens per bs
    parser.add_argument("--tpp", type=float, default=20.0)
    parser.add_argument("--vl_tokens", type=int, default=10485760)  # 1.0M vl tokens
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--mom", type=float, default=0.9)
    parser.add_argument("--opt", type=str, default="adam")
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--data", type=str, default="finewebedu10B")
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--warm_ratio", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--decay_1d", action="store_true")  # decay 1D params
    parser.add_argument("--decay_2d", action="store_true")  # decay 2D params
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--d_embds", type=int, default=384)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--dir_output", type=str, required=True)
    parser.add_argument("--dir_data", type=str, required=True)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=100)
    args = parser.parse_args()
    args.gpu = torch.cuda.get_device_name(0)

    dist.init_process_group("nccl")  # multi-gpu: setup
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    is_main = rank == 0

    dir_output = os.path.join(args.dir_output, args.run_name)
    os.makedirs(dir_output, exist_ok=True)
    seed_everything(args.seed)

    bs = args.tbs // args.seq_len  # global batch size
    assert bs % (world_size * args.mbs) == 0
    acc_steps = bs // (world_size * args.mbs)
    args.bs, args.acc_steps = bs, acc_steps
    print0("\n".join(f"{k}: {v}" for k, v in vars(args).items()))

    wandb.init(project="norm", mode="disabled" if not is_main else "online")
    wandb.config.update(args)

    tr_dl = data_generator(
        get_files(args.dir_data, args.data, "train"),
        args.tbs,
        args.seq_len,
        world_size,
        rank,
        local_rank,
    )
    vl_dl = data_generator(
        get_files(args.dir_data, args.data, "val"),
        args.vl_tokens,
        args.seq_len,
        world_size,
        rank,
        local_rank,
    )
    x_vl, y_vl = next(vl_dl)
    assert len(x_vl) % args.mbs == 0

    if args.model == "gpt":
        m = gpt_small(
            seq_len=args.seq_len,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_embds=args.d_embds,
        ).to(local_rank)
    else:
        raise ValueError(f"unknown model {args.model}")

    m = DDP(m, device_ids=[local_rank])
    m.compile(dynamic=False)
    criterion = nn.CrossEntropyLoss()
    n_params = sum(p.numel() for p in m.parameters())
    print0(f"model params: {n_params / 1e6:.2f}M")

    if args.opt == "muon":
        hidden_w_2d = [p for p in m.module.blocks.parameters() if p.ndim >= 2]
        hidden_w_1d = [p for p in m.parameters() if p.ndim < 2]
        adam = Adam(
            [
                {"params": [m.module.embd.weight], "lr": args.lr},
                {"params": [m.module.head.weight], "lr": args.lr},
                {"params": hidden_w_1d, "lr": args.lr},
            ],
            betas=(args.mom, args.mom),
        )
        muon = Muon(
            [{"params": hidden_w_2d}],
            args.lr,
            momentum=args.mom,
            weight_decay=0,
            adjust_lr_fn="match_rms_adamw",
        )
        optims = [adam, muon]
    elif args.opt == "adam":
        adam = Adam(
            m.parameters(),
            lr=args.lr,
            betas=(args.mom, args.mom),
        )
        optims = [adam]
    seen = {p for o in optims for gp in o.param_groups for p in gp["params"]}
    assert seen == set(m.parameters()), "param groups must tile all params"
    tot_tokens = int(args.tpp * n_params)  # chinchilla-style token budget
    tot_steps = tot_tokens // args.tbs
    scheduler = cosine_scheduler(
        total_steps=tot_steps,
        warm_steps=int(args.warm_ratio * tot_steps),
    )
    print0(f"tot tokens: {tot_tokens:,}")
    print0(f"tot steps: {tot_steps:,}")

    # Training loop
    m.train()
    for step in range(tot_steps):
        sc = scheduler[step]
        lr = args.lr * sc
        wd = args.wd * sc  # true decouple wd from lr
        for o in optims:
            for group in o.param_groups:
                group["lr"] = lr

        m.zero_grad(set_to_none=True)
        tr_loss = torch.zeros((), device=local_rank)
        x, y = next(tr_dl)
        for i in range(acc_steps):  # accumulate gradients over local-batches
            xmb = x[i * args.mbs : (i + 1) * args.mbs]
            ymb = y[i * args.mbs : (i + 1) * args.mbs]
            with autocast("cuda", dtype=torch.bfloat16):
                logits = m(xmb)
                loss = criterion(logits.view(-1, logits.size(-1)), ymb.view(-1)) / acc_steps  # fmt: skip
            loss.backward()
            tr_loss += loss.detach()
            del logits
        dist.all_reduce(tr_loss, op=dist.ReduceOp.AVG)
        if not torch.isfinite(tr_loss).item():  # kill on NaN/inf loss
            print0("non-finite loss, stopping")
            wandb.finish()
            dist.destroy_process_group()
            return

        gnorm = nn.utils.clip_grad_norm_(  # return grad norm before clip
            m.parameters(),
            args.grad_clip if args.grad_clip > 0 else float("inf"),
        )
        for o in optims:  # optimizer step
            o.step()
        with torch.no_grad():  # true decouple wd from lr
            for p in m.parameters():
                if (args.decay_1d and p.dim() == 1) or (args.decay_2d and p.dim() == 2):
                    p.data.mul_(1 - wd)

        # Log train metrics
        if step % args.log_interval == 0 and is_main:
            print0(f"step {step} tr_loss {tr_loss.item():.4f} lr {lr:.6f}")
            metrics = {
                "tr/loss": tr_loss.item(),
                "tr/lr": lr,
                "tr/wd": wd,
            }
            wandb.log(metrics, step)

        # Log validation metrics
        if step % args.eval_interval == 0 or step == tot_steps - 1:
            m.eval()
            vl_loss, vl_n = 0.0, 0
            with torch.no_grad():
                for i in range(len(x_vl) // args.mbs):
                    xmb = x_vl[i * args.mbs : (i + 1) * args.mbs]
                    ymb = y_vl[i * args.mbs : (i + 1) * args.mbs]
                    with autocast("cuda", dtype=torch.bfloat16):
                        logits = m(xmb)
                        loss = criterion(logits.view(-1, logits.size(-1)), ymb.view(-1))
                    vl_loss += loss.item() * ymb.numel()
                    vl_n += ymb.numel()
            stats = torch.tensor([vl_loss, vl_n], device=local_rank)
            dist.all_reduce(stats)
            vl_loss = stats[0].item() / stats[1].item()
            if is_main:
                print0(f"step {step} vl_loss {vl_loss:.4f}")
                metrics = {
                    "vl/loss": vl_loss,
                    "vl/g_norm": gnorm.item(),
                    "vl/w_norm": torch.sqrt(sum(p.data.norm() ** 2 for p in m.parameters())),
                }  # fmt: skip
                for n, p in m.named_parameters():
                    metrics[f"vl/w_norm/{n}"] = p.data.norm().item()
                wandb.log(metrics, step)
            m.train()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
