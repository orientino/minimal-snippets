import numpy as np


class CustomScheduler:
    def __init__(self, optim, schedule):
        self.optim = optim
        self.schedule = schedule
        self.it = 0

    def step(self):
        if self.it < len(self.schedule) - 1:
            lr = self.schedule[self.it]
            for param_group in self.optim.param_groups:
                param_group["lr"] = lr
            self.it += 1

    def get_last_lr(self):
        return self.schedule[self.it]


def cosine_scheduler(lr, lr_end, niter_per_epoch, epochs, warmup_epochs=5):
    # https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/utils.py

    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_epoch
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(0, lr, warmup_iters)

    iters = np.arange(epochs * niter_per_epoch - warmup_iters)
    schedule = lr_end + 0.5 * (lr - lr_end) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_epoch
    return schedule
