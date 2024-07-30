import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA


class PCACoordinates(object):
    def __init__(self, training_path):
        self.pca_, self.components = get_path_components_(training_path)
        self.origin_ = [w.clone() for w in training_path[-1]]
        self.v0_ = normalize_weights(self.components[0], self.origin_)
        self.v1_ = normalize_weights(self.components[1], self.origin_)

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

    def plot(self, scale=1.0, points=24, levels=20, ax=None, **kwargs):
        xs = self.a_grid
        ys = self.b_grid
        zs = self.loss_grid
        if ax is None:
            _, ax = plt.subplots(**kwargs)
            ax.set_aspect("equal")
        # Set Levels
        min_loss = zs.min()
        max_loss = zs.max()
        levels = np.exp(np.linspace(np.log(min_loss), np.log(max_loss), num=levels))
        CS = ax.contour(
            xs,
            ys,
            zs,
            levels=levels,
            cmap="magma",
            linewidths=0.75,
            norm=mpl.colors.LogNorm(vmin=min_loss, vmax=max_loss * 2.0),
        )
        ax.clabel(CS, inline=True, fontsize=8, fmt="%1.2f")
        return ax


def weights_to_coordinates(coords, path):
    """
    Project the training path onto the first two principal components
    using the pseudoinverse.
    """
    pcs = vectorize_weights([coords.v0_, coords.v1_])
    pcs_i = np.linalg.pinv(pcs)
    # use origin vector as center
    w_c = np.squeeze(vectorize_weights([path[-1]]))
    # center all the weights on the path and project onto components
    coord_path = np.array(
        [pcs_i @ (np.squeeze(vectorize_weights([w])) - w_c) for w in path]
    )
    return coord_path


def plot_path(coords, training_path, ax=None, end=None, **kwargs):
    path = weights_to_coordinates(coords, training_path)
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
    colors = range(path.shape[0])
    end = path.shape[0] if end is None else end
    norm = plt.Normalize(0, end)
    ax.scatter(
        path[:, 0],
        path[:, 1],
        s=4,
        c=colors,
        cmap="cividis",
        norm=norm,
    )
    return ax


def get_path_components_(training_path, n_components=2):
    weight_vectorized = vectorize_weights(training_path)  # vectorize weights
    pca = PCA(n_components=2, whiten=True)
    components = pca.fit_transform(weight_vectorized)
    weight = unvectorize_weights(components, training_path[0])  # un-vectorize
    return pca, weight


def vectorize_weights(weight_list):
    vec_list = []
    for weights in weight_list:
        vec = [w.detach().numpy().flatten() for w in weights]
        vec = np.concatenate(vec)
        vec_list.append(vec)
    weight_matrix = np.column_stack(vec_list)
    return weight_matrix


def unvectorize_weights(weight_matrix, example):
    weight_vecs = np.hsplit(weight_matrix, weight_matrix.shape[1])
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
