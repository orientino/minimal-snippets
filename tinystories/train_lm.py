import argparse
import os

import numpy as np
import pytorch_lightning as pl
import tiktoken
import torch
import torch.nn as nn
import wandb
from datasets import load_dataset
from torch.nn import functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--wd", type=float, default=0)
parser.add_argument("--train_iters", type=int, default=10000)
parser.add_argument("--eval_iters", type=int, default=100)
parser.add_argument("--eval_interval", type=int, default=100)

parser.add_argument("--n_embd", type=int, default=256)
parser.add_argument("--n_layer", type=int, default=2)
parser.add_argument("--n_head", type=int, default=4)

parser.add_argument("--debug", action="store_true")
args = parser.parse_args()


tokenizer = tiktoken.get_encoding("gpt2")
BATCH_SIZE = 128
BLOCK_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run():
    pl.seed_everything(args.seed)
    wandb.init(project="tinystories", mode="disabled" if args.debug else "online")
    wandb.config.update(args)

    # Create data
    vocab_size = tokenizer.n_vocab
    root = "/project/home/p200535/data/tinystories"
    root = "/scratch/users/czhang/cooldown/tinystories"
    train_ds = TinyStories(root, train=True)
    val_ds = TinyStories(root, train=False)

    print(f"vocab size {vocab_size}")
    print(f"train tokens {len(train_ds) // 1e6:.0f}M")
    print(f"val tokens {len(val_ds) // 1e6:.0f}M")

    # Create model
    m = GPTLanguageModel(
        vocab_size=vocab_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
    )
    m = m.to(DEVICE)
    optim = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=args.wd)
    print(f"model size {sum(p.numel() for p in m.parameters()) // 1e6:.0f}M")
    print()

    # Training
    for i in tqdm(range(args.train_iters)):
        x, y = get_batch(train_ds)
        x, y = x.to(DEVICE), y.to(DEVICE)

        logits, loss = m(x, y)
        optim.zero_grad()
        loss.backward()
        optim.step()

        if i % args.eval_interval == 0 or i == args.train_iters - 1:
            loss_train = get_loss(m, train_ds, DEVICE)
            loss_val = get_loss(m, val_ds, DEVICE)
            print(f"[step {i}] train loss {loss_train:.4f}, val loss {loss_val:.4f}")
            wandb.log({"train/loss": loss_train})
            wandb.log({"val/loss": loss_val})

    print("\nSaving the model weights...")
    torch.save(m.state_dict(), "model.pt")
    print("\nGenerating text from the model...")
    context = torch.tensor(tokenizer.encode("\n"), dtype=torch.long, device=DEVICE)
    context = context.unsqueeze(0)
    y = tokenizer.decode(m.generate(context, max_new_tokens=500)[0].tolist())
    print(y)


@torch.no_grad()
def get_loss(m, data, DEVICE):
    m.eval()
    losses = torch.zeros(args.eval_iters)
    for k in range(args.eval_iters):
        x, y = get_batch(data)
        x, y = x.to(DEVICE), y.to(DEVICE)
        _, loss = m(x, y)
        losses[k] = loss.item()
    m.train()
    return losses.mean()


def get_batch(data):
    """Sample a random batch from the dataset."""

    idx = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    b = [data[i] for i in idx]
    x = torch.stack([_b[0] for _b in b])
    y = torch.stack([_b[1] for _b in b])
    return x, y


class TinyStories(Dataset):
    def __init__(self, root, train):
        if not os.path.exists(root):
            os.makedirs(root)

        data = load_dataset("roneneldan/TinyStories")
        if train:
            self._pretokenize(data["train"]["text"], os.path.join(root, "train.npy"))
            self.data = np.load(os.path.join(root, "train.npy"))
        else:
            self._pretokenize(data["validation"]["text"], os.path.join(root, "val.npy"))
            self.data = np.load(os.path.join(root, "val.npy"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x = self.data[i : i + BLOCK_SIZE]
        y = self.data[i + 1 : i + BLOCK_SIZE + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    def _pretokenize(self, data, path):
        """Pretokenize the data to speed up the training."""

        if os.path.exists(path):
            print("Data already tokenized.")
            return

        arrays = []
        for line in tqdm(data):
            tokens = tokenizer.encode(line)
            arrays.append(np.array(tokens, dtype=np.int32))
        np.save(path, np.concatenate(arrays))


class Head(nn.Module):
    def __init__(self, n_embd, head_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        # compute attention scores
        # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T, hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, dropout)
        self.ff = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_embd,
        n_layer,
        n_head,
        dropout=0.1,
    ):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, dropout) for _ in range(n_layer)]
        )
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln(x)  # (B,T,C)
        logits = self.head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last BLOCK_SIZE tokens
            idx_cond = idx[:, -BLOCK_SIZE:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


if __name__ == "__main__":
    run()
