import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from visualize import (
    LossSurface,
    PCACoordinates,
    get_weights,
)


class QuadraticDataset(Dataset):
    def __init__(self, n_samples=256):
        self.x = torch.randn((n_samples, 1))
        self.y = self.x**2 + torch.randn_like(self.x) * 0.25

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.model(x)


ds = QuadraticDataset()
dl = DataLoader(ds, batch_size=32, shuffle=True)
model = Model()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

training_path = [get_weights(model)]
for epoch in tqdm(range(50)):
    for x, y in ds:
        loss = criterion(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Callback to store the weights
    training_path.append(get_weights(model))

# Plot the loss surface
coords = PCACoordinates(training_path)
loss_surface = LossSurface(model, ds.x, ds.y)
loss_surface.compile(points=30, coords=coords, criterion=criterion, scale=0.5)

ax = loss_surface.plot_surface(levels=30, dpi=100)
ax = loss_surface.plot_path(coords, training_path, ax)
plt.show()
