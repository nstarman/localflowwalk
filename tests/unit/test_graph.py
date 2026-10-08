"""Tests for the pure-JAX graph package (``phasecurvefit._src.graph``).

Each module is checked against its reference: scipy's csgraph, or the host
helpers in ``orderers/mst.py`` that the ``SciPy()`` path still uses.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phasecurvefit._src.graph._pointer import accumulate, find_roots, list_rank


class TestPointer:
    """Pointer jumping: roots, path sums and list ranking."""

    def test_find_roots_and_accumulate(self):
        """Roots and root-path sums of a random forest equal a direct walk."""
        rng = np.random.default_rng(0)
        n = 300
        parent = np.array(
            [rng.integers(0, i) if i and rng.random() < 0.9 else i for i in range(n)]
        )
        w = np.where(parent == np.arange(n), 0.0, rng.random(n)).astype(np.float32)
        roots = np.asarray(jax.jit(find_roots)(jnp.asarray(parent)))
        r2, dist = map(
            np.asarray, jax.jit(accumulate)(jnp.asarray(parent), jnp.asarray(w))
        )
        for i in range(n):
            j, total = i, 0.0
            while parent[j] != j:
                total += w[j]
                j = parent[j]
            assert roots[i] == r2[i] == j
            assert dist[i] == pytest.approx(total, rel=1e-5)

    def test_list_rank(self):
        """Steps to the end of a list; elements on a cycle are flagged."""
        order = np.random.default_rng(1).permutation(40)
        succ = np.full(50, 50)
        succ[order[:-1]] = order[1:]  # a 40-element list
        succ[40:50] = np.r_[41:50, 40]  # a 10-element cycle
        steps, reached = map(np.asarray, jax.jit(list_rank)(jnp.asarray(succ)))
        np.testing.assert_array_equal(steps[order], np.arange(39, -1, -1))
        np.testing.assert_array_equal(reached, np.arange(50) < 40)
