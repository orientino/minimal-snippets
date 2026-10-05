"""
Streaming dataloader for the modded-nanogpt FineWeb-EDU token shards.
"""

import os
from pathlib import Path

import torch


def get_files(dir_data, dataset, split):
    pattern = os.path.join(dir_data, dataset)
    files = sorted(Path(pattern).glob(f"*_{split}_*.bin"))
    assert files, f"no {split} shards found in {pattern}"
    return files


def _load_data_shard(path):
    header = torch.from_file(str(path), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])
    with path.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "token count does not match header"
    return tokens


def data_generator(files, batch_tokens, seq_len, world_size, rank, device):
    assert batch_tokens % world_size == 0
    local_tokens = batch_tokens // world_size
    while True:  # cycle through shards forever
        for path in files:
            tokens = _load_data_shard(path)
            pos = 0
            while pos + batch_tokens + 1 <= len(tokens):
                buf = tokens[pos + rank * local_tokens :][: local_tokens + 1]
                x = buf[:-1].to(device, dtype=torch.int32, non_blocking=True)
                y = buf[1:].to(device, dtype=torch.int64, non_blocking=True)
                pos += batch_tokens
                yield x.view(-1, seq_len), y.view(-1, seq_len)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_data", type=str, required=True)
    parser.add_argument("--data", type=str, default="finewebedu10B")
    args = parser.parse_args()

    tr = get_files(args.dir_data, args.data, "train")
    vl = get_files(args.dir_data, args.data, "val")
    print(f"train shards: {len(tr)}, val shards: {len(vl)}")
