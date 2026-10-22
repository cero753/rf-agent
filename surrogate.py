"""Learned surrogate: an MLP ensemble that predicts the reference model's correction to the fast model.

Residual learning on dimensionless features (h/lambda, W/h, er, tand). Trained only on thin, low-er
boards so out-of-distribution behaviour (thick, ceramic) is measurable. Ensemble spread = uncertainty.

  python surrogate.py          # build data, train, write evals/surrogate.png + models/surrogate.pkl
"""
from __future__ import annotations

import math
import pickle
import time
from pathlib import Path

import numpy as np

import rf_tools as rt

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "surrogate.pkl"
ID_MAX_H_LAMBDA = 0.03
ID_MAX_ER = 4.6


def features(w: float, l: float, sub: rt.Substrate) -> np.ndarray:
    fast = rt.fast_patch(w, l, 0.0, sub)
    lam = rt.C0 / fast.fr
    return np.array([sub.h / lam, math.log(w / sub.h), sub.er, sub.tand * 100, l / w])


def in_distribution(w: float, l: float, sub: rt.Substrate) -> bool:
    lam = rt.C0 / rt.fast_patch(w, l, 0.0, sub).fr
    return sub.h / lam < ID_MAX_H_LAMBDA and sub.er <= ID_MAX_ER


def _targets(w: float, l: float, sub: rt.Substrate) -> np.ndarray:
    fast, ref = rt.fast_patch(w, l, 0.0, sub), rt.reference_patch(w, l, 0.0, sub)
    return np.array([100 * (ref.fr / fast.fr - 1), math.log(ref.r_edge / fast.r_edge),
                     math.log(ref.q / fast.q)])


def sample(n: int, rng: np.random.Generator, ood: bool = False) -> tuple[np.ndarray, np.ndarray, list]:
    X, Y, meta = [], [], []
    while len(X) < n:
        f = rng.uniform(0.8, 14) * 1e9
        er = rng.uniform(9.0, 10.5) if ood and rng.random() < 0.5 else rng.uniform(2.0, 4.6 if not ood else 10.5)
        h_mm = rng.uniform(0.2, 3.5)
        sub = rt.Substrate(er, h_mm, rng.uniform(0, 0.025))
        w, l = rt.patch_dimensions(f, sub)
        w *= rng.uniform(0.8, 1.25)
        l *= rng.uniform(0.9, 1.1)
        if w <= 0 or l <= 0:
            continue
        idd = in_distribution(w, l, sub)
        if idd == ood:
            continue
        X.append(features(w, l, sub))
        Y.append(_targets(w, l, sub))
        meta.append((w, l, sub))
    return np.array(X), np.array(Y), meta


class Surrogate:
    def __init__(self, models, x_mu, x_sd, y_mu, y_sd, unc_threshold):
        self.models, self.x_mu, self.x_sd, self.y_mu, self.y_sd = models, x_mu, x_sd, y_mu, y_sd
        self.unc_threshold = unc_threshold

    def _raw(self, X: np.ndarray) -> np.ndarray:
        Xn = (X - self.x_mu) / self.x_sd
        return np.stack([m.predict(Xn) * self.y_sd + self.y_mu for m in self.models])  # (k, n, 3)

    def predict(self, w: float, l: float, y0: float, sub: rt.Substrate, f_target: float | None = None,
                z0: float = 50.0) -> dict:
        fast = rt.fast_patch(w, l, 0.0, sub)
        p = self._raw(features(w, l, sub)[None, :])[:, 0, :]
        mean, std = p.mean(0), p.std(0)
        fr = fast.fr * (1 + mean[0] / 100)
        r_edge = fast.r_edge * math.exp(mean[1])
        q = fast.q * math.exp(mean[2])
        r_in = r_edge * math.cos(math.pi * y0 / l) ** 2
        out = {"fr": fr, "r_edge": r_edge, "r_in": r_in, "q": q,
               "uncertainty_fr_pct": float(std[0]), "in_distribution": in_distribution(w, l, sub)}
        out["escalate"] = bool(out["uncertainty_fr_pct"] > self.unc_threshold or not out["in_distribution"])
        if f_target:
            z = rt._z_parallel_rlc(np.array([f_target]), r_in, fr, q)
            out["s11_at_target_db"] = float(20 * np.log10(abs(rt._s11(z, z0)[0]) + 1e-15))
        return out


def train(n_train: int = 5000, n_test: int = 600, seed: int = 0, k: int = 5, verbose: bool = True) -> Surrogate:
    from sklearn.neural_network import MLPRegressor

    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    X, Y, _ = sample(n_train, rng)
    if verbose:
        print(f"generated {n_train} reference samples in {time.perf_counter() - t0:.1f}s")
    x_mu, x_sd, y_mu, y_sd = X.mean(0), X.std(0), Y.mean(0), Y.std(0)
    models = []
    for i in range(k):
        idx = rng.integers(0, len(X), len(X))
        m = MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh", max_iter=600, early_stopping=True,
                         random_state=seed + i, learning_rate_init=3e-3)
        m.fit((X[idx] - x_mu) / x_sd, (Y[idx] - y_mu) / y_sd)
        models.append(m)
    s = Surrogate(models, x_mu, x_sd, y_mu, y_sd, unc_threshold=0.0)
    Xv, Yv, _ = sample(n_test, rng)
    sd_v = s._raw(Xv).std(0)[:, 0]
    s.unc_threshold = float(np.quantile(sd_v, 0.95))
    MODEL_PATH.parent.mkdir(exist_ok=True)
    MODEL_PATH.write_bytes(pickle.dumps(vars(s)))
    if verbose:
        print(f"trained {k}-model ensemble in {time.perf_counter() - t0:.1f}s; "
              f"escalation threshold (95th pct ID spread) = {s.unc_threshold:.3f}% fr")
    return s


_CACHE: Surrogate | None = None


def load() -> Surrogate:
    global _CACHE
    if _CACHE is None:
        _CACHE = (Surrogate(**pickle.loads(MODEL_PATH.read_bytes())) if MODEL_PATH.exists()
                  else train(verbose=False))
    return _CACHE


def surrogate_loop_design(f_hz: float, sub: rt.Substrate, z0: float = 50.0, tol_f: float = 1e-3,
                          tol_r: float = 1.0, max_iter: int = 8) -> dict:
    """Iterate on the surrogate for free; escalate to the reference only when the surrogate is
    unsure or out of distribution; always spend one reference call on final sign-off."""
    s = load()
    d = rt.open_loop_design(f_hz, sub, z0)
    w, l, y0 = d["w"], d["l"], d["y0"]
    surrogate_calls, escalated = 0, False
    for _ in range(max_iter):
        p = s.predict(w, l, y0, sub)
        surrogate_calls += 1
        if p["escalate"]:
            escalated = True
            break
        if abs(p["fr"] - f_hz) / f_hz < tol_f / 2 and abs(p["r_in"] - z0) < tol_r:
            break
        l = rt.correct_length(w, l, p["fr"], f_hz, sub)
        y0 = rt.inset_for(s.predict(w, l, 0.0, sub)["r_edge"], l, z0) or 0.0
        surrogate_calls += 1
    ref_calls = 0
    for _ in range(max_iter):
        ref = rt.reference_patch(w, l, y0, sub, z0)
        ref_calls += 1
        if abs(ref.fr - f_hz) / f_hz < tol_f and abs(ref.r_in - z0) < tol_r:
            break
        l = rt.correct_length(w, l, ref.fr, f_hz, sub)
        ref_calls += 1
        y0 = rt.inset_for(rt.reference_patch(w, l, 0.0, sub, z0).r_edge, l, z0) or 0.0
    return {"w": w, "l": l, "y0": y0, "surrogate_calls": surrogate_calls, "ref_calls": ref_calls,
            "escalated": escalated, "in_distribution": in_distribution(w, l, sub)}


def evaluate(s: Surrogate, n: int = 600, seed: int = 99) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    res = {}
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for name, ood in (("in-distribution", False), ("out-of-distribution", True)):
        X, Y, _ = sample(n, rng, ood=ood)
        p = s._raw(X)
        err = np.abs(p.mean(0)[:, 0] - Y[:, 0])            # |fr error| in %
        fast_err = np.abs(Y[:, 0])                          # fast model's own error vs reference
        unc = p.std(0)[:, 0]
        res[name] = {"surrogate_fr_mae_pct": float(err.mean()), "surrogate_fr_p95_pct": float(np.quantile(err, .95)),
                     "fast_model_fr_mae_pct": float(fast_err.mean()),
                     "mean_uncertainty_pct": float(unc.mean()),
                     "flagged_for_escalation": float(np.mean(unc > s.unc_threshold)) if not ood else 1.0,
                     "uncertainty_flags_alone": float(np.mean(unc > s.unc_threshold))}
        ax.scatter(unc, err, s=7, alpha=0.5, label=name)
    ax.axvline(s.unc_threshold, ls="--", c="grey", lw=0.8)
    ax.set(xscale="log", yscale="log", xlabel="Ensemble spread (% fr)", ylabel="|Surrogate fr error| (%)",
           title="Does the surrogate know when it's wrong?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "evals" / "surrogate.png", dpi=130)
    plt.close(fig)

    X, _, meta = sample(200, rng)
    t = time.perf_counter(); s._raw(X); batch = (time.perf_counter() - t) / len(X)
    w, l, sub = meta[0]
    t_ref = rt.timed(rt.reference_patch, w, l, 0.0, sub, repeat=20)
    res["per_design_batched_us"] = batch * 1e6
    res["reference_ms"] = t_ref * 1e3
    res["batched_speedup_x"] = t_ref / batch
    return res


if __name__ == "__main__":
    import json

    s = train()
    print(json.dumps(evaluate(s), indent=1))
