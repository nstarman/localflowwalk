"""Plots: phasecurvefit's JAX kNN backends against exact oracles (PR #160)."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import phasecurvefit as pcf

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
N = pcf.neighbors
K = 10

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
mpl.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False,
})


def oracle(p, k, queries=None):
    """Float64 brute force; self excluded; ties to the lower index (stable sort)."""
    p64 = p.astype(np.float64)
    q = p64 if queries is None else queries.astype(np.float64)
    idx, dist = [], []
    for s in range(0, len(q), 512):
        d2 = ((q[s : s + 512, None] - p64[None]) ** 2).sum(-1)
        if queries is None:
            d2[np.arange(d2.shape[0]), np.arange(s, s + d2.shape[0])] = np.inf
        o = np.argsort(d2, axis=1, kind="stable")[:, :k]
        idx.append(o)
        dist.append(np.sqrt(np.take_along_axis(d2, o, 1)))
    return np.concatenate(idx), np.concatenate(dist)


rng = np.random.default_rng(0)


def stream(n, interlopers=0.02):
    t = rng.uniform(0, 1, n)
    p = np.c_[10 * t, np.sin(3 * t), 0.3 * np.cos(2 * t)] + rng.normal(0, 0.02, (n, 3))
    m = int(interlopers * n)
    p[:m] = rng.uniform(p.min(0) - 3, p.max(0) + 3, (m, 3))
    return p.astype(np.float32)


def grid(n):
    side = int(round(n ** (1 / 3)))
    g = np.stack(np.meshgrid(*[np.arange(side)] * 3), -1).reshape(-1, 3)
    return g[rng.permutation(len(g))].astype(np.float32)


def clumps(n):
    p = rng.normal(size=(n, 3)).astype(np.float32)
    src = rng.integers(0, n // 10, n)
    keep = rng.random(n) < 0.5
    p[keep] = p[src[keep]]  # ~half the points are exact duplicates of others
    return p


DATA = {
    "Gaussian blob": rng.normal(size=(8000, 3)).astype(np.float32),
    "Stream + 2% interlopers": stream(8000),
    "Integer grid (heavy ties)": grid(8000),
    "Duplicate clumps": clumps(8000),
}


# ---------------------------------------------------------------- figure 1
rows = []
for name, p in DATA.items():
    oi, od = oracle(p, K)
    bi, bd = map(np.asarray, N.BucketKDTree().knn(jnp.asarray(p), K))
    ji, _ = map(np.asarray, jax.jit(lambda x: N.BucketKDTree().knn(x, K))(jnp.asarray(p)))
    fi, _ = map(np.asarray, N.BruteForce().knn(jnp.asarray(p), K))
    si, sd = map(np.asarray, N.SciPy().knn(jnp.asarray(p), K))
    rows.append({
        "name": name,
        "vs oracle": np.mean(bi == oi),
        "eager vs jit": np.mean(bi == ji),
        "vs BruteForce": np.mean(bi == fi),
        "vs SciPy": np.mean(np.isclose(bd, sd, rtol=1e-6, atol=1e-7)),
        "dist_err": np.max(np.abs(bd - od) / np.maximum(od, 1e-30)),
        "sd_err": np.max(np.abs(sd - od) / np.maximum(od, 1e-30)),
        "bd": bd, "od": od,
    })
    print(name, {k: (round(v * 100, 4) if k.startswith("v") or k.startswith("e") else v)
                 for k, v in rows[-1].items() if k not in ("bd", "od", "name")})

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.5]})
ax = axes[0]
r = rows[1]  # stream + interlopers
ax.plot([r["od"].min(), r["od"].max()], [r["od"].min(), r["od"].max()], color=INK2,
        lw=1, ls="--", zorder=1, label="y = x")
ax.scatter(r["od"].ravel(), r["bd"].ravel(), s=9, color=BLUE, alpha=0.5, lw=0, zorder=2,
           label="BucketKDTree")
ax.set_xscale("log"), ax.set_yscale("log")
ax.set_xlabel("float64 brute-force distance (oracle)")
ax.set_ylabel("BucketKDTree distance (float32)")
ax.set_title("Distances: stream + interlopers, n=8000, k=10")
ax.legend(loc="upper left")
ax.text(0.97, 0.04, f"max rel. error {r['dist_err']:.1e}\n(SciPy: {r['sd_err']:.1e})",
        transform=ax.transAxes, ha="right", va="bottom", color=INK2, fontsize=9)

ax = axes[1]
cmp = [("vs oracle", "float64 oracle", BLUE), ("eager vs jit", "itself under jit", ORANGE),
       ("vs SciPy", "SciPy distances*", AQUA)]
y = np.arange(len(rows))
h = 0.26
for j, (key, label, color) in enumerate(cmp):
    vals = np.array([r[key] for r in rows]) * 100
    pos = y + (j - 1) * (h + 0.02)
    ax.barh(pos, vals, height=h, color=color, label=label)
    for yy, v in zip(pos, vals):
        ax.text(v - 0.5 if v > 20 else v + 0.5, yy, f"{v:.2f}%" if v < 100 else "100%",
                va="center", ha="right" if v > 20 else "left", fontsize=8,
                color="white" if v > 20 else INK)
ax.set_yticks(y, [r["name"] for r in rows])
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("neighbour slots identical (%)")
ax.text(1.0, -0.42, "*SciPy breaks ties in its own order, so on tied data only its distances are comparable.",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)
ax.set_title("BucketKDTree neighbours identical to…")
ax.grid(axis="y", visible=False)
ax.legend(loc="lower left", bbox_to_anchor=(0, -0.32), ncol=3)
fig.tight_layout()
fig.savefig(OUT / "knn_vs_oracles.png", dpi=150)

# ---------------------------------------------------------------- figure 2
t = np.linspace(0, 1, 1500)
arc = np.c_[10 * np.cos(2.5 * t), 10 * np.sin(2.5 * t), 0.3 * t] + rng.normal(0, 0.05, (1500, 3))
perm = rng.permutation(1500)
arc, tt = arc[perm].astype(np.float32), t[perm]
vel = np.gradient(arc, axis=0)
pos_d = {c: jnp.asarray(arc[:, i]) for i, c in enumerate("xyz")}
vel_d = {c: jnp.ones(1500) for c in "xyz"}
orders = {}
for label, backend in (("SciPy", N.SciPy()), ("BucketKDTree", N.BucketKDTree()),
                       ("BruteForce", N.BruteForce())):
    o = pcf.orderers.MSTOrderer(k=10, jump_cap=2.0, neighbors=backend)
    orders[label] = np.asarray(o.order(pos_d, vel_d).indices)
oj = pcf.orderers.MSTOrderer(k=10, jump_cap=2.0)
orders["BucketKDTree (jit)"] = np.asarray(jax.jit(lambda p, v: oj.order(p, v).indices)(pos_d, vel_d))


def rank(idx):
    r = np.empty(len(idx), int)
    r[idx] = np.arange(len(idx))
    return r


ref = rank(orders["SciPy"])
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
ax.plot([0, 1500], [0, 1500], color=INK2, lw=1, ls="--", zorder=1, label="y = x")
for (label, color) in (("BucketKDTree", BLUE), ("BucketKDTree (jit)", ORANGE),
                       ("BruteForce", AQUA)):
    same = np.array_equal(orders[label], orders["SciPy"])
    ax.scatter(ref, rank(orders[label]), s=10 if color == BLUE else 4, color=color,
               lw=0, alpha=0.8, label=f"{label}{' (identical)' if same else ''}")
ax.set_xlabel("position in SciPy-backend ordering")
ax.set_ylabel("position in JAX-backend ordering")
ax.set_title("MSTOrderer: noisy 3-D arc, n=1500, k=10")
ax.legend(loc="upper left")

ax = axes[1]
gamma = rank(orders["BucketKDTree"])
ax.scatter(tt, gamma, s=5, color=BLUE, lw=0)
rho = np.corrcoef(tt, gamma)[0, 1]
ax.set_xlabel("true arc parameter t")
ax.set_ylabel("position in BucketKDTree ordering")
ax.set_title("Ordering recovers the arc")
ax.text(0.97, 0.04, f"Pearson r = {abs(rho):.5f}", transform=ax.transAxes, ha="right",
        va="bottom", color=INK2, fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "mst_vs_scipy.png", dpi=150)
print("mst identical:", {k: np.array_equal(v, orders["SciPy"]) for k, v in orders.items()},
      "r", rho)
