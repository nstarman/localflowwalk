"""Pure-JAX graph algorithms on fixed-shape edge lists.

Minimum spanning forests and components (Borůvka), tree diameter paths (Euler
tour + list ranking), robust MST edge-length clipping and component bridging.
Everything is jit/vmap-traceable with static shapes and no ``lax.cond``.

Imports only the standard library, numpy, jax, jaxtyping and equinox -- never
the parent package -- so it can be lifted into its own library.
"""

__all__: tuple[str, ...] = ()
