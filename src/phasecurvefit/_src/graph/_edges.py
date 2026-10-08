"""The kNN edge list: lengths, velocity cosines, weights, keep mask; orientation."""

__all__: tuple[str, ...] = ("knn_edges", "orient_flip")

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

_COS_TINY = 1e-12  # |v_i||v_j| below this: cosine defined as 0


def knn_edges(
    p: Float[Array, "n d"],
    v: Float[Array, "n d"],
    nbr: Int[Array, "n k"],
    real: Bool[Array, " n"],
    /,
    *,
    jump_cap: float,
    velocity_weight: float,
    sever_cos_threshold: float | None,
) -> tuple[Array, Array, Array, Array, Array]:
    """Undirected edge list ``(lo, hi, d, w, valid)`` from kNN indices.

    ``nbr[i]`` lists ``i``'s neighbours; an index ``>= n`` is a missing
    neighbour. ``d`` is the spatial length, ``w`` the MST weight
    ``max(d + velocity_weight * (1 - cos), tiny)``; ``valid`` drops missing and
    padded neighbours, edges longer than ``jump_cap`` and (if set) edges with
    velocity cosine below ``sever_cos_threshold``.
    """
    n, k = nbr.shape
    rows = jnp.repeat(jnp.arange(n), k)
    raw = nbr.reshape(-1)
    cols = jnp.minimum(raw, n - 1)
    diff = p[rows] - p[cols]
    d = jnp.sqrt(jnp.sum(diff * diff, axis=-1))
    valid = (raw < n) & real[rows] & real[cols] & (d <= jump_cap)
    w = d
    if velocity_weight > 0.0 or sever_cos_threshold is not None:
        vi, vj = v[rows], v[cols]
        num = jnp.sum(vi * vj, axis=-1)
        den = jnp.linalg.norm(vi, axis=-1) * jnp.linalg.norm(vj, axis=-1)
        cos = jnp.where(den > _COS_TINY, num / jnp.maximum(den, _COS_TINY), 0.0)
        if velocity_weight > 0.0:
            w = d + velocity_weight * (1.0 - cos)
        if sever_cos_threshold is not None:
            valid = valid & (cos >= sever_cos_threshold)
    w = jnp.maximum(w, jnp.finfo(w.dtype).tiny)
    return jnp.minimum(rows, cols), jnp.maximum(rows, cols), d, w, valid


def orient_flip(
    p: Float[Array, "n d"],
    v: Float[Array, "n d"],
    full: Int[Array, " n"],
    blen: Int[Array, ""],
    /,
) -> Bool[Array, ""]:
    """Whether the path runs against the mean velocity (NaN velocities ignored)."""
    n = full.shape[0]
    seg = jnp.arange(n - 1) < blen - 1
    tang = p[full[1:]] - p[full[:-1]]
    vmid = 0.5 * (v[full[:-1]] + v[full[1:]])
    dot = jnp.where(seg[:, None], tang * vmid, 0.0)
    return jnp.nansum(dot) < 0.0
