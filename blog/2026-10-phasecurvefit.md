---
title: "Ordering phase-space data along a curve with phasecurvefit"
date: "2026-10-XX"
type: blog
excerpt:
  phasecurvefit is a JAX-native package that orders noisy, shuffled phase-space
  samples along a curve and learns a smooth track through them, even when the
  curve crosses itself.
author: Nathaniel Starkman
blog_topic: software
comments: true
url: "/blog/phasecurvefit.html"
lastmod: "2026-10-XX"
image:
  src: images/blog/2026/phasecurvefit/02-ordering.png
  alt:
    "Three panels ordering the same self-crossing curve: positions only zig-zags
    between strands; adding velocities and a Self-Organizing Map recovers the
    true order."
---

<!--
OUTLINE / DRAFT. Lives on a branch of GalacticDynamics/phasecurvefit until ready.
To move to pyOpenSci/pyopensci.github.io:
  - this file   -> content/blog/2026-MM-DD-phasecurvefit.md (set date/lastmod)
  - blog/images -> static/images/blog/2026/phasecurvefit/
  - change the image paths below from `images/...` to `/images/blog/2026/phasecurvefit/...`
Figures: `uv run --group docs python blog/make_figures.py`
-->

<!-- Hook (2-3 sentences). -->

- Data often lie along a curve but arrive shuffled.
- Examples: stellar streams, orbits, trajectories, time series seen out of
  order.
- First question before any fit: where along the curve is each sample?
- `phasecurvefit` answers that, and has been
  [accepted into pyOpenSci](https://github.com/pyOpenSci/software-submission/issues/285).

## The problem: order is lost

- Input: positions (and usually velocities), shuffled, no labels.
- Need 1: an ordering coordinate γ for every sample.
- Need 2: a smooth track, γ → position.
- Science case: stellar streams = tidal debris from disrupted clusters/dwarfs.
- Stream tracks constrain the Milky Way's dark matter; link a review.
- Naive tools fail: nearest neighbour, clustering, sorting by angle.
- They fail where it matters: crossings, fold-backs, gaps, noise.

![Shuffled samples along a self-crossing curve, with velocity arrows](images/01-data.png)

_A toy curve with five self-crossings: 2048 noisy, shuffled samples. Arrows show
velocities._

## Positions are not enough

- At a crossing, the nearest point is often on the other strand.
- Key idea: use velocities too.
- Strands at a crossing move in different directions → cut those links.
- Left panel: positions-only MST zig-zags across strands (|ρ| = 0.26).
- Middle: velocity-aware MST stays on one strand (|ρ| = 0.90).
- Right: SOM refinement smooths the backbone (|ρ| = 1.00).
- Define |ρ|: rank correlation with the true order; 1 is perfect.
- No start point needed: the MST finds the two tips itself.
- `|` chains orderers; each stage seeds the next.

![Spatial MST vs velocity-aware MST vs MST | SOM](images/02-ordering.png)

_Left: a minimum spanning tree built on positions alone jumps between strands.
Middle: cutting links between samples moving in different directions keeps it on
one strand. Right: refining with a Self-Organizing Map recovers the true order._

```python
import phasecurvefit as pcf

orderer = pcf.orderers.MSTOrderer(
    k=16, sever_cos_threshold=0.9
) | pcf.orderers.SOMOrderer(n_prototypes=60)
result = pcf.order(positions, velocities, orderer)
result.ordering  # sample indices, tip to tip
```

## From an ordering to a track

- An ordering is discrete; a model needs a continuous track.
- Full pipeline: MST → SOM → autoencoder.
- The MST | SOM ordering is the autoencoder's training input.
- Encoder: phase-space point → γ, for every sample (even skipped ones).
- Decoder: γ → smooth track.
- Decoder is swappable; `FourierTrackNet` handles the sharp loops here.
- One call does it all: `pcf.fit_track(positions, velocities, key=...)`.
- Need speed (e.g. inside MCMC)? Training-free running-mean track.

![Samples coloured by learned γ with the smooth learned track](images/03-track.png)

_Trained on the MST | SOM ordering, the autoencoder assigns an ordering
parameter γ to every sample and learns a smooth track that follows each loop and
crossing._

## Other things it does

- Outlier rejection: stream + background mixture → membership probabilities.
- Cite Hogg, Bovy & Lang (2010) for the mixture model.
- Other orderers: local-flow walk (Nibauer et al. 2022).
- SOM ordering (Starkman et al. 2023).
- Pluggable distance metrics and neighbour strategies.
- Physical units via `unxt`.
- Built on JAX: `jit`, `vmap`, `grad`; CPU or GPU.
- Differentiable → can sit inside a larger model or likelihood.

## Getting started

```
pip install phasecurvefit[all]
```

- Minimal example: `pcf.order(positions, velocities)` with the default pipeline.
- Then `pcf.fit_track(positions, velocities, key=...)` for the track.
- Inputs are plain dicts, e.g. `{"x": ..., "y": ...}`; any dimension.
- Point to tutorials: stream MST, epitrochoid, outlier rejection.
- Link the docs quickstart.

## The pyOpenSci review

- Editor: @kysolvik. EiC: @crhea93.
- Reviewers: @SujataSaurabh, @crhea93.
- Thank each by name; one specific thing each caught.
- Concrete fixes from review: missing core dependency.
- `max_dist=0` walk bug.
- KDTree docs (a filter, not a speedup).
- Wrong default metric in docstrings.
- Link the PRs once merged.
- Why it was worth it: outside eyes found real bugs.
- Mention JOSS fast-track if you submit there.

## Try it out

- GitHub: GalacticDynamics/phasecurvefit.
- Docs: phasecurvefit.readthedocs.io.
- PyPI: `phasecurvefit`.
- Cite: Zenodo DOI + `CITATION.cff`.
- pyOpenSci review issue #285.
- Not just streams: any ordered phase-space data.
- Invite issues, use cases, contributions.
- Co-maintainer @jnibauer: co-author the post?
