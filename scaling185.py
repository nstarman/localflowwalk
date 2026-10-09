"""Walk scaling benchmark for PR #185; runs unchanged on main and on the PR branch.

usage: python scaling185.py OUT.json
  per_step: walk cost per step, strategy state precomputed outside the timed
            region (so this is pure stepping), STEPS steps, min of REPS runs.
  setup:    strategy.init alone (jitted), min of REPS runs.
  full:     a complete walk (all N steps, including setup), min of 2 runs.
"""
import json, sys, time
import equinox as eqx, jax, jax.numpy as jnp
import phasecurvefit as pcf

STEPS, REPS = 1000, 3
N_STEP = [1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000, 128_000]
N_FULL = [2_000, 4_000, 8_000, 16_000, 32_000]

def data(n):
    t = jnp.linspace(0, 20, n); k = jax.random.split(jax.random.key(0), 3)
    q = {"x": jnp.cos(t) * t + 0.01 * jax.random.normal(k[0], (n,)),
         "y": jnp.sin(t) * t + 0.01 * jax.random.normal(k[1], (n,)),
         "z": 0.01 * jax.random.normal(k[2], (n,))}
    p = {"x": jnp.cos(t) - t * jnp.sin(t), "y": jnp.sin(t) + t * jnp.cos(t), "z": jnp.zeros(n)}
    return q, p

def strategies():
    return {"BruteForce": pcf.strats.BruteForce(), "KDTree(k=50)": pcf.strats.KDTree(k=50)}

def cached(strategy, state):
    """Same strategy, but init() returns a precomputed state."""
    class Cached(type(strategy)):
        def __init__(self): self.__dict__.update(strategy.__dict__)
        def init(self, positions, /, *, metadata): return state
    return Cached()

@eqx.filter_jit
def run(q, p, o): return pcf.order(q, p, o).indices

def best(f, reps):
    jax.block_until_ready(f()); ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); jax.block_until_ready(f()); ts.append(time.perf_counter() - t0)
    return min(ts)

out = {"per_step_us": {}, "setup_s": {}, "full_s": {}}
for n in N_STEP:
    q, p = data(n)
    for name, s in strategies().items():
        # Eager init: main's KDTree state holds a Python int (n_points) that
        # its query() needs concrete, so it can't come out of a jit.
        state = s.init(q, metadata=None)
        if name != "BruteForce":
            out["setup_s"].setdefault(name, {})[n] = best(lambda: s.init(q, metadata=None), REPS)
        o = pcf.orderers.LocalFlowOrderer(config=pcf.WalkConfig(strategy=cached(s, state)),
                                          metric_scale=1.0, n_max=STEPS, start_idx=0)
        out["per_step_us"].setdefault(name, {})[n] = best(lambda: run(q, p, o), REPS) / STEPS * 1e6
        print(f"n={n:7d} {name:13s} step {out['per_step_us'][name][n]:9.2f} us", flush=True)
for n in N_FULL:
    q, p = data(n)
    for name, s in strategies().items():
        o = pcf.orderers.LocalFlowOrderer(config=pcf.WalkConfig(strategy=s), metric_scale=1.0, start_idx=0)
        out["full_s"].setdefault(name, {})[n] = best(lambda: run(q, p, o), 2)
        print(f"n={n:7d} {name:13s} full walk {out['full_s'][name][n]:7.2f} s", flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
