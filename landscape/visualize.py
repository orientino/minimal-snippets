import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

"""
P_list: parameters of the model per layer, e.g. [64, 64, ..., 10]
P: total number of parameters of the model
N: number of weights, i.e. len(training_path)
"""


class PCACoordinates(object):
    def __init__(self, training_path):
        self.pca_, self.pcs = get_principal_components(training_path)
        self.origin_ = [w.clone() for w in training_path[-1]]  # P_list
        self.v0_ = normalize_weights(self.pcs[0], self.origin_)  # P_list
        self.v1_ = normalize_weights(self.pcs[1], self.origin_)  # P_list

    def __call__(self, a, b):
        return [
            a * w0 + b * w1 + wc for w0, w1, wc in zip(self.v0_, self.v1_, self.origin_)
        ]


class RandomCoordinates(object):
    def __init__(self, origin):
        self.origin_ = [w.clone() for w in origin]
        self.v0_ = normalize_weights([torch.randn_like(w) for w in origin], origin)
        self.v1_ = normalize_weights([torch.randn_like(w) for w in origin], origin)

    def __call__(self, a, b):
        return [
            a * w0 + b * w1 + wc for w0, w1, wc in zip(self.v0_, self.v1_, self.origin_)
        ]


class LossSurface(object):
    def __init__(self, model, x, y):
        self.model = model
        self.x = x
        self.y = y

    def compile(self, points, coords, criterion, scale=1.0):
        a_grid = torch.linspace(-1.0, 1.0, steps=points) * scale
        b_grid = torch.linspace(-1.0, 1.0, steps=points) * scale
        loss_grid = np.empty([len(a_grid), len(b_grid)])
        for i, a in enumerate(a_grid):
            for j, b in enumerate(b_grid):
                load_weights(self.model, coords(a, b))
                loss = criterion(self.model(self.x), self.y)
                loss_grid[j, i] = loss.item()
        load_weights(self.model, coords.origin_)
        loss = criterion(self.model(self.x), self.y)
        load_weights(self.model, coords.origin_)

        self.a_grid = a_grid
        self.b_grid = b_grid
        self.loss_grid = loss_grid

    def plot(self, coords, training_path, levels=30, **kwargs):
        xs, ys, zs = self.a_grid, self.b_grid, self.loss_grid

        # Plot the loss surface
        _, ax = plt.subplots(**kwargs)
        ax.set_aspect("equal")
        min_loss = zs.min()
        max_loss = zs.max()
        levels = np.exp(np.linspace(np.log(min_loss), np.log(max_loss), num=levels))
        CS = ax.contour(
            xs,
            ys,
            zs,
            levels=levels,
            cmap="coolwarm",
            linewidths=0.5,
            norm=mpl.colors.LogNorm(vmin=min_loss, vmax=max_loss * 2.0),
        )
        ax.clabel(CS, inline=True, fontsize=5, fmt="%1.2f")

        # Plot the training path
        path2d = weights_to_coordinates(coords, training_path)
        ax.scatter(
            path2d[:, 0],
            path2d[:, 1],
            s=20,
            c=range(path2d.shape[0]),
            cmap="viridis",
            norm=plt.Normalize(0, path2d.shape[0]),
        )
        plt.show()


def get_principal_components(training_path, n_components=2):
    """
    Compute the first two principal components of the training path
    """

    weights_vectorized = vectorize_weights(training_path)  # [P, N] vectorize
    pca = PCA(n_components=2, whiten=True)
    pcs = pca.fit_transform(weights_vectorized)  # [P, 2]
    weights = unvectorize_weights(pcs, training_path[0])  # [P_list, 2] un-vectorize
    return pca, weights


def weights_to_coordinates(coords, path):
    """
    Project the training path onto the first two principal components
    using the pseudoinverse.
    """
    w_c = vectorize_weights([coords.origin_])  # [P, 1]
    pcs = vectorize_weights([coords.v0_, coords.v1_])  # [P, 2]
    pcs_i = np.linalg.pinv(pcs)  # [2, P] pseudo-inverse

    # center the weights and project into principla components
    coord_path = [pcs_i @ (vectorize_weights([w]) - w_c) for w in path]
    coord_path = np.array([c.flatten() for c in coord_path])  # [N, 2]

    return coord_path


# Helper functions -----------------------------------------------------------------------------------------------


def vectorize_weights(weights_list):
    vec_list = []
    for weights in weights_list:
        vec = [w.detach().numpy().flatten() for w in weights]
        vec = np.concatenate(vec)
        vec_list.append(vec)
    weights_matrix = np.column_stack(vec_list)
    return weights_matrix


def unvectorize_weights(weights_matrix, example):
    weight_vecs = np.hsplit(weights_matrix, weights_matrix.shape[1])
    sizes = [v.numel() for v in example]
    shapes = [v.shape for v in example]
    weight_list = []
    for net_weights in weight_vecs:
        vs = np.split(net_weights, np.cumsum(sizes))[:-1]
        vs = [v.reshape(s) for v, s in zip(vs, shapes)]
        vs = [torch.tensor(v, dtype=torch.float32) for v in vs]
        weight_list.append(vs)
    return weight_list


def normalize_weights(weights, origin):
    return [w * torch.norm(wc) / torch.norm(w) for w, wc in zip(weights, origin)]


def load_weights(model, weight_list):
    param_list = list(model.parameters())
    assert len(param_list) == len(weight_list), "Weights mismatch"

    for param, weight in zip(param_list, weight_list):
        param.data = weight


def get_weights(model):
    return [w.clone() for w in model.parameters()]
