phasecurvefit 0.3.3 is a bug-fix release on the 0.3.x line. There are no breaking API changes: no public names were removed or renamed, and the core dependencies and Python support (≥3.12) are the same as in 0.3.2. Several fixes change numerical results, though, so they are listed first.

Most of this release responds to the [pyOpenSci review](https://github.com/pyOpenSci/software-submission/issues/285); see #105 and #162.

## ⚠️ Behaviour changes

These are bug fixes, but you may see different results from 0.3.2:

- **`train_autoencoder` gives different results for the same `key`** (#135). Each training phase now gets its own random key. In 0.3.2, one sub-key was used twice. Trained models are no longer bit-identical to 0.3.2 for a given seed.
- **`RunningMeanDecoder` returns NaN for empty windows** (#104). If no member star falls within `window_size / 2` of a query $\gamma$, the decoder now returns NaN. In 0.3.2 it silently returned the centre of the data: a zero weighted sum divided by `1e-10` gives 0 in normalized coordinates, which is the data mean once un-normalized. To fall back to the member stars nearest in $\gamma$ instead, use `RunningMeanDecoder(..., empty_window="nearest")`.
- **`LocalFlowOrderer(max_dist=0)` stops at the start point** (#182). It used to revisit the start point repeatedly, returning `indices` such as `[0, 0, 0]`. The cause was `inf * 0 = NaN` in the visited-point mask. The same mask now also handles negative `max_dist` correctly.
- **Coincident (duplicate) points** (#118, #124). `KDTree` and `MSTOrderer` now exclude the current point from its neighbour list by index, not by dropping the first neighbour. They also keep zero-length edges between duplicate points in the MST graph. Orderings of data with exactly coincident points can change. Data without duplicates is unaffected.

## 🐛 Other fixes

- **`StateMetadata` defaults:** a single default `StateMetadata()` instance is no longer shared across calls (#188). `dict(metadata)` and `**metadata` now work correctly (#188).
- **Tutorials:** fixed an undefined `rng` in the stream tutorials (#93).

## ✨ Additions

These are small and additive:

- **`RunningMeanDecoder(empty_window="nan" | "nearest")`:** a new option, described above (#104).
- **A `tutorials` extra:** `pip install "phasecurvefit[tutorials]"` installs what the tutorial notebooks need (matplotlib, galax) (#94).

## 📝 Documentation

- **Tutorials and guides:**
  - Clearer explanations throughout, including the stream autoencoder and its loss curve, the running-mean offset, the outlier-rejection mixture likelihood, and the epitrochoid and MST tutorials (#101, #190).
  - Fixes and explanations of parameter choices in the quickstart, metrics, algorithm and `nn` guides (#113, #122, #127, #186).
  - Every tutorial notebook has been re-executed against this release (#108, #190).
- **`KDTree`:** now described accurately as a spatial *candidate filter*. In 0.3.x it is not faster than `BruteForce` (#181, #184).
- **Default metric:** docstrings now name the correct default, `AlignedMomentumDistanceMetric` (#183).
- **Installation and API docs:** GPU/CUDA installation instructions (#97), and API objects are now link targets in the docs (#174).

## ✅ Testing

- **Smoke tests:** a CI job now installs only the core dependencies and runs smoke tests, which catches undeclared imports (#187).
- **`tests/test_kdtree.py`:** it now runs in CI. A misnamed optional-dependency check had always skipped it (#194).
- **Test layout:** tests are organised into `unit/`, `smoke/` and `integration/`, as on `main` (#179).

---

<!-- Everything below is GitHub's auto-generated list (.github/release.yml); regenerate it with "Generate release notes" when publishing. -->

## What's Changed
### ✨ New Features
* Backport PR #91 on branch versions/v0.3.x (Add tutorials extra for matplotlib/galax) by @meeseeksmachine in https://github.com/GalacticDynamics/phasecurvefit/pull/94
### 🐛 Bug Fixes
* Backport PR #92 on branch versions/v0.3.x (fix: undefined rng variable in stream tutorials) by @meeseeksmachine in https://github.com/GalacticDynamics/phasecurvefit/pull/93
* Backport PR #103 on branch versions/v0.3.x (🐛 fix: RunningMeanDecoder returns NaN for empty windows) by @meeseeksmachine in https://github.com/GalacticDynamics/phasecurvefit/pull/104
* Backport PR #115 on branch versions/v0.3.x (🐛 fix: exclude self by index, not position, in kNN queries) by @meeseeksmachine in https://github.com/GalacticDynamics/phasecurvefit/pull/118
* Backport PR #121 on branch versions/v0.3.x (🐛 fix: keep zero-length (duplicate-point) edges in the MST graph) by @meeseeksmachine in https://github.com/GalacticDynamics/phasecurvefit/pull/124
* Backport PR #163: 📝 docs: make API objects link targets; clear the docs build's warnings by @nstarman in https://github.com/GalacticDynamics/phasecurvefit/pull/174
* Backport PR #175 on branch versions/v0.3.x (🐛 fix: stop the walk at max_dist=0 instead of revisiting the start) by @nstarman in https://github.com/GalacticDynamics/phasecurvefit/pull/182
* Backport PR #102 on branch versions/v0.3.x (🐛 fix: don't share one StateMetadata across calls) by @nstarman in https://github.com/GalacticDynamics/phasecurvefit/pull/188
* Backport PR #192 on branch versions/v0.3.x (✅ test: run test_kdtree.py in CI (OptDeps named the wrong distribution)) by @meeseeksmachine in https://github.com/GalacticDynamics/phasecurvefit/pull/194
### 📝 Documentation
* Backport PR #90 on branch versions/v0.3.x (docs: add GPU/CUDA installation instructions) by @meeseeksmachine in https://github.com/GalacticDynamics/phasecurvefit/pull/97
* Backport PR #96 on branch versions/v0.3.x (📝 docs: explain the stream autoencoder tutorial and its loss curve) by @meeseeksmachine in https://github.com/GalacticDynamics/phasecurvefit/pull/101
* docs: re-execute all tutorial notebooks (v0.3.x) by @nstarman in https://github.com/GalacticDynamics/phasecurvefit/pull/108
* Backport PR #110 on branch versions/v0.3.x (📝 docs: nn guide's encoder-epoch advice matches the default) by @meeseeksmachine in https://github.com/GalacticDynamics/phasecurvefit/pull/113
* Backport PR #116 on branch versions/v0.3.x (📝 docs: fix errors in the guides) by @nstarman in https://github.com/GalacticDynamics/phasecurvefit/pull/122
* Backport PR #117 on branch versions/v0.3.x (📝 docs: explain choices in the quickstart, metrics, algorithm and nn guides) by @nstarman in https://github.com/GalacticDynamics/phasecurvefit/pull/127
* Backport PR #130 on branch versions/v0.3.x (Potential fix for 1 code quality finding) by @meeseeksmachine in https://github.com/GalacticDynamics/phasecurvefit/pull/132
* Backport PR #177 on branch versions/v0.3.x (📝 docs: drop remaining KDTree speed claims) by @nstarman in https://github.com/GalacticDynamics/phasecurvefit/pull/184
* Backport PR #176 on branch versions/v0.3.x (📝 docs: describe KDTree as a candidate filter, not a speedup) by @meeseeksmachine in https://github.com/GalacticDynamics/phasecurvefit/pull/181
* Backport PR #178 on branch versions/v0.3.x (📝 docs: name the right default metric; drop stale ruff ignore) by @meeseeksmachine in https://github.com/GalacticDynamics/phasecurvefit/pull/183
* Backport PR #126 on branch versions/v0.3.x (📝 docs: fix quickstart wording flagged on PR #117) by @nstarman in https://github.com/GalacticDynamics/phasecurvefit/pull/186
* Backport PRs #95, #98, #99, #100, #111 on branch versions/v0.3.x (📝 tutorial explanations) by @nstarman in https://github.com/GalacticDynamics/phasecurvefit/pull/190
### ✅ Testing
* Backport PR #109 on branch versions/v0.3.x (🚚 refactor: organize tests/ into unit, smoke, and integration folders) by @nstarman in https://github.com/GalacticDynamics/phasecurvefit/pull/179
* Backport PR #172 on branch versions/v0.3.x (✅ test: core-only smoke tests in CI) by @meeseeksmachine in https://github.com/GalacticDynamics/phasecurvefit/pull/187
### Other Changes
* Backport PR #134 on branch versions/v0.3.x (Fix key indexing in autoencoder training) by @meeseeksmachine in https://github.com/GalacticDynamics/phasecurvefit/pull/135
* Backport PR #189 on branch versions/v0.3.x (🙈 chore: ignore coverage.xml too) by @nstarman in https://github.com/GalacticDynamics/phasecurvefit/pull/193

**Full Changelog**: https://github.com/GalacticDynamics/phasecurvefit/compare/v0.3.2...v0.3.3
