import numpy as np
from sklearn.datasets import make_circles, make_moons


def load_moons(n_samples=100, noise=0.1, random_state=None):
    X, y = make_moons(
        n_samples=n_samples, noise=noise, random_state=random_state
    )
    return X, y, None

def load_circles(n_samples=100, noise=0.05, factor=0.5, random_state=None):
    X, y = make_circles(
        n_samples=n_samples,
        noise=noise,
        factor=factor,
        random_state=random_state,
    )
    return X, y, None

def load_xor(n_samples=100, noise=0.1, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    X = np.random.uniform(-1, 1, size=(n_samples, 2))
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
    if noise > 0:
        X += np.random.normal(0, noise, size=X.shape)
    return X, y, None

def load_spirals(n_samples=100, noise=0.1, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    n = n_samples // 2
    theta = np.linspace(0, 2 * np.pi, n)
    r = np.linspace(0.1, 1, n)

    x1 = r * np.cos(theta) + np.random.normal(0, noise, n)
    y1 = r * np.sin(theta) + np.random.normal(0, noise, n)
    x2 = r * np.cos(theta + np.pi) + np.random.normal(0, noise, n)
    y2 = r * np.sin(theta + np.pi) + np.random.normal(0, noise, n)

    X = np.vstack([
        np.column_stack([x1, y1]),
        np.column_stack([x2, y2])
    ])
    y = np.array([0] * n + [1] * (n_samples - n))
    return X, y, None

def load_checkerboard(n_samples=100, noise=0.1, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    X = np.random.uniform(-2, 2, size=(n_samples, 2))
    y = (np.floor(X[:, 0]) + np.floor(X[:, 1])) % 2
    y = y.astype(int)
    if noise > 0:
        X += np.random.normal(0, noise, size=X.shape)
    return X, y, None