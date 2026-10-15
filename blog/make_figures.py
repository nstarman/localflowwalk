"""Make the figures for the pyOpenSci blog post.

Adapted from ``docs/tutorials/epitrochoid_mst.ipynb``. Run from the repo root:

    uv run python blog/make_figures.py
"""

from pathlib import Path

import jax
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

import quaxed.numpy as jnp
import unxt as u

import phasecurvefit as pcf

OUT = Path(__file__).parent / "images"
usys = u.unitsystems.si
meta = pcf.StateMetadata(usys=usys)
key = jr.key(201030)


def make_epitrochoid(key, n, noise_sigma, scale=120.0, R=5.0, r=1.0, d=4.5):
    """Open epitrochoid (5 inner loops) with noisy positions, exact velocities."""
    t = jnp.linspace(5.0, 355.0, n) * jnp.pi / 180.0
    ratio = (R + r) / r
    x0 = scale * ((R + r) * jnp.cos(t) - d * jnp.cos(ratio * t)) / 5.0
    y0 = scale * ((R + r) * jnp.sin(t) - d * jnp.sin(ratio * t)) / 5.0
    vx = scale * (-(R + r) * jnp.sin(t) + d * ratio * jnp.sin(ratio * t)) / 5.0
    vy = scale * ((R + r) * jnp.cos(t) - d * ratio * jnp.cos(ratio * t)) / 5.0
    kx, ky = jr.split(key)
    x = x0 + noise_sigma * jr.normal(kx, (n,))
    y = y0 + noise_sigma * jr.normal(ky, (n,))
    pos = {"x": u.Q(x, usys["length"]), "y": u.Q(y, usys["length"])}
    vel = {"x": u.Q(vx, usys["speed"]), "y": u.Q(vy, usys["speed"])}
    return t, pos, vel


key, k1, k2 = jr.split(key, 3)
t, pos, vel = make_epitrochoid(k1, n=2048, noise_sigma=6)
# Shuffle, so the input order carries no information.
perm = jr.permutation(k2, len(t))
qs = jax.tree.map(lambda a: a[perm], pos)
ps = jax.tree.map(lambda a: a[perm], vel)
t_shuffled = np.asarray(t[perm])
X = np.asarray(u.ustrip(usys, qs["x"]))
Y = np.asarray(u.ustrip(usys, qs["y"]))


def rho(result):
    """|Correlation| between rank in the ordering and the true curve parameter."""
    to = t_shuffled[np.asarray(result.ordering)]
    return abs(np.corrcoef(np.arange(to.size), to)[0, 1])


def draw_order(ax, result, title):
    o = np.asarray(result.ordering)
    ax.scatter(X, Y, s=2, c="lightgray")
    ax.scatter(X[o], Y[o], s=6, c=np.linspace(0, 1, o.size), cmap="viridis")
    ax.plot(X[o], Y[o], c="k", lw=0.4, alpha=0.6)
    ax.set(aspect="equal", title=f"{title}\n|ρ| = {rho(result):.2f}")
    ax.set_axis_off()


# --- Figure 1: the problem -----------------------------------------------
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(X, Y, s=4, c="gray", alpha=0.7)
every = 12
ax.quiver(
    X[::every],
    Y[::every],
    np.asarray(ps["x"].value[::every]),
    np.asarray(ps["y"].value[::every]),
    scale=10_000,
    width=0.004,
    alpha=0.6,
)
ax.set(aspect="equal", title="2048 shuffled samples (positions + velocities)")
ax.set_axis_off()
fig.savefig(OUT / "01-data.png", dpi=150, bbox_inches="tight")

# --- Figure 2: positions alone vs. phase space ---------------------------
P = np.stack([X, Y], axis=1)
med = float(np.median(cKDTree(P).query(P, k=2)[0][:, 1]))

spatial = pcf.orderers.MSTOrderer(k=16, jump_cap=6.0 * med, on_disconnected="largest")
velaware = pcf.orderers.MSTOrderer(
    k=16,
    jump_cap=8.0 * med,
    sever_cos_threshold=0.9,
    orient_by_velocity=True,
    on_disconnected="largest",
)
pipeline = velaware | pcf.orderers.SOMOrderer(n_prototypes=60)

res_spatial = pcf.order(qs, ps, spatial, metadata=meta)
res_vel = pcf.order(qs, ps, velaware, metadata=meta)
res_som = pcf.order(qs, ps, pipeline, metadata=meta)

fig, axs = plt.subplots(1, 3, figsize=(13, 4.6))
draw_order(axs[0], res_spatial, "MST, positions only")
draw_order(axs[1], res_vel, "MST, velocity-aware")
draw_order(axs[2], res_som, "velocity-aware MST | SOM")
fig.savefig(OUT / "02-ordering.png", dpi=150, bbox_inches="tight")

# --- Figure 3: a smooth track through the ordering -----------------------
key, enc_key, dec_key, train_key = jr.split(key, 4)
normalizer = pcf.nn.StandardScalerNormalizer(qs, ps)
decoder = pcf.nn.FourierTrackNet(
    out_size=2, n_frequencies=12, width_size=256, depth=4, key=dec_key
)
model = pcf.nn.PathAutoencoder.make(
    normalizer, gamma_range=res_som.gamma_range, decoder=decoder, key=enc_key
)
config = pcf.nn.TrainingConfig(
    n_epochs_encoder=800, n_epochs_decoder=1200, n_epochs_both=0
)
ae, _, _ = pcf.nn.train_autoencoder(model, res_som, key=train_key, config=config)

gamma, _ = ae.model.encode(qs, ps)
track = ae(jnp.linspace(*ae.gamma_range, 1_000))
fig, ax = plt.subplots(figsize=(5.5, 5))
im = ax.scatter(X, Y, s=4, c=np.asarray(gamma), cmap="viridis")
ax.plot(
    np.asarray(u.ustrip(usys, track["x"])),
    np.asarray(u.ustrip(usys, track["y"])),
    c="k",
    lw=1.5,
    label="learned track",
)
fig.colorbar(im, ax=ax, label="ordering parameter γ")
ax.set(aspect="equal", title="Learned γ and smooth track")
ax.set_axis_off()
ax.legend(loc="lower right")
fig.savefig(OUT / "03-track.png", dpi=150, bbox_inches="tight")

print(
    f"rho: spatial={rho(res_spatial):.2f} vel={rho(res_vel):.2f} som={rho(res_som):.2f}"
)
