#!/usr/bin/env python3
"""
TOPIC-T / TOPIC-K – Transitionally Oriented and Powered Interacting Cells
Pure-Python + NumPy implementation with Particle Swarm Optimization (PSO).

sklearn-compatible interface (works when scikit-learn is present; falls back
gracefully when it is not).

Files requested:
  - topick.py  – actual implementation (this file, PSO solver)
  - topick.nim – not possible (Nim/nimpy absent from the environment)

Performance measured on the UCI Sonar (Mines vs Rocks) dataset.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, List

# ---------------------------------------------------------------------------
# Optional sklearn imports (soft dependency)
# ---------------------------------------------------------------------------
try:
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
    from sklearn.utils.multiclass import unique_labels
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

    class BaseEstimator:
        def get_params(self, deep=True):
            return {k: getattr(self, k) for k in self.__dict__ if not k.endswith("_")}
        def set_params(self, **params):
            for k, v in params.items():
                setattr(self, k, v)
            return self

    class ClassifierMixin:
        def score(self, X, y):
            return float(np.mean(self.predict(X) == y))

# ---------------------------------------------------------------------------
# Direction mapping (8-neighbourhood, clockwise from Top)
# ---------------------------------------------------------------------------
DIR_MAP = [
    (-1,  0),  # 0 Top
    (-1,  1),  # 1 Top-Right
    ( 0,  1),  # 2 Right
    ( 1,  1),  # 3 Bottom-Right
    ( 1,  0),  # 4 Bottom
    ( 1, -1),  # 5 Bottom-Left
    ( 0, -1),  # 6 Left
    (-1, -1),  # 7 Top-Left
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _inverse_sigmoid(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return np.log(p / (1.0 - p))


# ---------------------------------------------------------------------------
# Core TOPIC unit (single Document = M Characters)
# ---------------------------------------------------------------------------
class TopicUnit:
    """One Document consisting of M Characters (3×3 grids)."""

    def __init__(self, M: int, K: int, n_features: int, rng: np.random.Generator):
        self.M = M
        self.K = K
        self.n_features = n_features
        self.input_dim = n_features + 3  # x + row_off + col_off + prev_ref
        # Director weights: (M, 3, 3, input_dim)
        self.W = rng.normal(0.0, 0.15, size=(M, 3, 3, self.input_dim))

    def _director(self, m: int, r: int, c: int, x: np.ndarray,
                  row_off: float, col_off: float, prev_ref: float) -> float:
        inp = np.concatenate([x, [row_off, col_off, prev_ref]])
        return float(_sigmoid(self.W[m, r, c] @ inp))

    def _move(self, r: int, c: int, direction: int,
              prev_r: int, prev_c: int) -> Tuple[int, int]:
        """Move, preferring not to reverse; fall back clockwise on edges."""
        for k in range(8):
            d = (direction + k) % 8
            dr, dc = DIR_MAP[d]
            nr, nc = r + dr, c + dc
            if 0 <= nr < 3 and 0 <= nc < 3 and not (nr == prev_r and nc == prev_c):
                return nr, nc
        return r, c  # stay if completely stuck (rare)

    def forward(self, x: np.ndarray, record_visits: bool = False
                ) -> Tuple[float, np.ndarray, Optional[np.ndarray]]:
        """
        Returns
        -------
        doc_ref : float
            Document Referent (final probability for binary case).
        char_refs : ndarray of shape (M,)
            Character Referents.
        visits : ndarray of shape (M, 3, 3) or None
        """
        char_refs = np.zeros(self.M)
        visits = np.zeros((self.M, 3, 3), dtype=np.int32) if record_visits else None

        for m in range(self.M):
            r, c = 1, 1          # Initial Position = centre
            prev_r, prev_c = 1, 1
            prev_ref = 0.5
            row_off = 0.0
            col_off = 0.0
            refs = []

            for step in range(self.K + 1):
                if record_visits:
                    visits[m, r, c] += 1
                ref = self._director(m, r, c, x, row_off, col_off, prev_ref)
                refs.append(ref)

                if step == self.K:
                    break

                direction = min(7, int(ref * 8.0))
                nr, nc = self._move(r, c, direction, prev_r, prev_c)
                row_off = float(nr - r)
                col_off = float(nc - c)
                prev_r, prev_c = r, c
                prev_ref = ref
                r, c = nr, nc

            char_refs[m] = np.mean(refs)

        doc_ref = float(np.mean(char_refs))
        return doc_ref, char_refs, visits

    def get_params_flat(self) -> np.ndarray:
        return self.W.ravel().copy()

    def set_params_flat(self, theta: np.ndarray) -> None:
        self.W = theta.reshape(self.W.shape)


# ---------------------------------------------------------------------------
# Simple global-best PSO
# ---------------------------------------------------------------------------
def pso(
    objective,
    bounds: List[Tuple[float, float]],
    n_particles: int = 30,
    n_iter: int = 80,
    w: float = 0.72,
    c1: float = 1.49,
    c2: float = 1.49,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, float]:
    if rng is None:
        rng = np.random.default_rng()
    dim = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    pos = rng.uniform(lo, hi, size=(n_particles, dim))
    vel = rng.uniform(-(hi - lo), hi - lo, size=(n_particles, dim)) * 0.1
    pbest = pos.copy()
    pbest_val = np.array([objective(p) for p in pos])
    gbest_idx = int(np.argmin(pbest_val))
    gbest = pbest[gbest_idx].copy()
    gbest_val = float(pbest_val[gbest_idx])

    for it in range(n_iter):
        r1 = rng.random((n_particles, dim))
        r2 = rng.random((n_particles, dim))
        vel = (w * vel
               + c1 * r1 * (pbest - pos)
               + c2 * r2 * (gbest - pos))
        # velocity clamp
        vmax = 0.2 * (hi - lo)
        vel = np.clip(vel, -vmax, vmax)
        pos = np.clip(pos + vel, lo, hi)

        vals = np.array([objective(p) for p in pos])
        improved = vals < pbest_val
        pbest[improved] = pos[improved]
        pbest_val[improved] = vals[improved]

        best_idx = int(np.argmin(pbest_val))
        if pbest_val[best_idx] < gbest_val:
            gbest_val = float(pbest_val[best_idx])
            gbest = pbest[best_idx].copy()

    return gbest, gbest_val


# ---------------------------------------------------------------------------
# scikit-learn style estimator
# ---------------------------------------------------------------------------
class TopicT(BaseEstimator, ClassifierMixin if HAS_SKLEARN else object):
    """
    TOPIC-T estimator (binary classification focus; extensible).

    Parameters
    ----------
    M : int
        Number of Characters (3×3 grids).
    K : int
        Number of Sprite steps per Character.
    n_particles : int
        PSO swarm size.
    n_iter : int
        PSO iterations.
    w_min, w_max : float
        Bounds for Director weights.
    random_state : int or None
    """

    def __init__(
        self,
        M: int = 4,
        K: int = 5,
        n_particles: int = 25,
        n_iter: int = 60,
        w_min: float = -2.5,
        w_max: float = 2.5,
        random_state: Optional[int] = None,
    ):
        self.M = M
        self.K = K
        self.n_particles = n_particles
        self.n_iter = n_iter
        self.w_min = w_min
        self.w_max = w_max
        self.random_state = random_state

    def fit(self, X, y):
        if HAS_SKLEARN:
            X, y = check_X_y(X, y, dtype=np.float64, ensure_min_samples=2)
            self.classes_ = unique_labels(y)
        else:
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y)
            self.classes_ = np.unique(y)

        if len(self.classes_) != 2:
            raise ValueError("This reference implementation supports binary classification only.")

        # map to 0/1
        y_bin = np.where(y == self.classes_[1], 1.0, 0.0)
        self.n_features_in_ = X.shape[1]
        self.rng_ = np.random.default_rng(self.random_state)

        unit = TopicUnit(self.M, self.K, self.n_features_in_, self.rng_)
        n_params = unit.W.size
        bounds = [(self.w_min, self.w_max)] * n_params

        def objective(theta: np.ndarray) -> float:
            unit.set_params_flat(theta)
            loss = 0.0
            visit_sum = np.zeros((self.M, 3, 3), dtype=np.float64)

            for i in range(len(X)):
                doc_ref, char_refs, visits = unit.forward(X[i], record_visits=True)
                p = np.clip(doc_ref, 1e-12, 1.0 - 1e-12)
                yi = y_bin[i]
                loss += -(yi * np.log(p) + (1.0 - yi) * np.log(1.0 - p))
                if visits is not None:
                    visit_sum += visits

            loss /= len(X)

            # Visitation loss = 1 / harmonic mean of positive cell counts
            positive = visit_sum[visit_sum > 0]
            if positive.size > 0:
                harm = np.mean(1.0 / positive)
            else:
                harm = 10.0
            loss += 0.1 * harm
            return float(loss)

        best_theta, best_loss = pso(
            objective,
            bounds,
            n_particles=self.n_particles,
            n_iter=self.n_iter,
            rng=self.rng_,
        )
        unit.set_params_flat(best_theta)
        self.unit_ = unit
        self.best_loss_ = best_loss
        return self

    def predict_proba(self, X):
        if HAS_SKLEARN:
            check_is_fitted(self, attributes=["unit_"])
            X = check_array(X, dtype=np.float64)
        else:
            if not hasattr(self, "unit_"):
                raise RuntimeError("Estimator not fitted")
            X = np.asarray(X, dtype=np.float64)

        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature dimension mismatch")

        probs = []
        for x in X:
            doc_ref, _, _ = self.unit_.forward(x)
            p = np.clip(doc_ref, 1e-9, 1.0 - 1e-9)
            probs.append([1.0 - p, p])
        return np.asarray(probs)

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = (proba[:, 1] >= 0.5).astype(int)
        return self.classes_[idx]

    def decision_function(self, X):
        p = self.predict_proba(X)[:, 1]
        return _inverse_sigmoid(p)


# ---------------------------------------------------------------------------
# Sonar evaluation (prefer fetch_openml, fallback to local UCI file)
# ---------------------------------------------------------------------------
def load_sonar():
    """Load the classic Sonar (Mines vs Rocks) dataset.

    Preference order:
      1. sklearn.datasets.fetch_openml (data_id=151 or name='sonar')
      2. Local UCI file previously downloaded to /tmp/sonar.all-data
    """
    # --- 1. try OpenML via sklearn ---
    try:
        from sklearn.datasets import fetch_openml
        print("Loading Sonar via sklearn.datasets.fetch_openml ...")
        # OpenML data_id 151 is the classic Sonar data set
        ds = fetch_openml("sonar", as_frame=False)
        X = np.asarray(ds.data, dtype=np.float64)
        # target is 'Rock' / 'Mine' (or 0/1 depending on version)
        y_raw = ds.target
        if y_raw.dtype.kind in ("U", "O", "S"):
            y = np.array([1 if str(v).startswith("M") else 0 for v in y_raw], dtype=np.int32)
        else:
            y = np.asarray(y_raw, dtype=np.int32)
        return X, y
    except Exception as e:
        print(f"fetch_openml unavailable ({e}); falling back to local UCI file.")

    # --- 2. local fallback ---
    path = "/tmp/sonar.all-data"
    data, labels = [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            data.append([float(v) for v in parts[:-1]])
            labels.append(1 if parts[-1] == "M" else 0)
    return np.asarray(data, dtype=np.float64), np.asarray(labels, dtype=np.int32)


def train_test_split(X, y, test_size=0.3, random_state=42):
    rng = np.random.default_rng(random_state)
    n = len(X)
    idx = rng.permutation(n)
    n_test = int(n * test_size)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


if __name__ == "__main__":
    print("=" * 60)
    print("TOPIC-T (pure NumPy + PSO) – Sonar dataset evaluation")
    print("=" * 60)

    X, y = load_sonar()
    print(f"Sonar shape: {X.shape}, classes: {np.bincount(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    clf = TopicT(
        M=3,
        K=4,
        n_particles=20,
        n_iter=40,
        w_min=-2.0,
        w_max=2.0,
        random_state=42,
    )

    print("\nFitting (PSO) ...")
    clf.fit(X_train, y_train)
    print(f"Best training loss: {clf.best_loss_:.4f}")

    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]
    acc = float(np.mean(pred == y_test))

    # Mann-Whitney AUC
    pos = proba[y_test == 1]
    neg = proba[y_test == 0]
    if len(pos) and len(neg):
        auc = (
            np.sum(pos[:, None] > neg[None, :])
            + 0.5 * np.sum(pos[:, None] == neg[None, :])
        ) / (len(pos) * len(neg))
    else:
        auc = float("nan")

    print(f"\nTest accuracy : {acc:.3f}")
    print(f"Test AUC      : {auc:.3f}")
    print("Done.")
