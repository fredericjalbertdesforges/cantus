# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **OLTW Dixon 2005** algorithm in `cantus.algorithms.oltw_dixon`
  with a clean-room implementation honouring the original paper
  (`MAX_RUN_COUNT = 3`, explicit `BOTH` direction, no warmup skip).
- **Feature extractor abstraction** (`cantus.features.base.FeatureExtractor`)
  designed for pluggable continuous-pitch and chroma-style features.
- **Piano-roll feature extractor** (`cantus.features.piano_roll`)
  for symbolic MIDI baseline benchmarking.
- **Alignment metrics** in `cantus.metrics`: alignment rate at
  multiple thresholds, mean/median/p95 absolute error, summary helper.
- **Initial test suite** with theoretical anchors (identity warp,
  2x stretch with frame-level ground truth) and a synthetic rubato
  curve regression test.
- Packaging with the `src` layout, Hatchling build backend, Apache 2.0
  license, `CITATION.cff`, and a GitHub Actions CI matrix covering
  ubuntu/macos/windows × Python 3.10–3.13 with `ruff`, `mypy --strict`,
  and `pytest`.

### Notes on design

Cantus does not aim for bit-exact parity with `pymatchmaker` (Park 2025).
Three documented design choices differ:

1. `MAX_RUN_COUNT = 3` (canonical Dixon) vs Matchmaker's `30`.
2. Cantus implements three directions (`R`, `C`, `B`); Matchmaker collapses to two.
3. Cantus does not skip a warmup window of reference frames at startup.

These choices follow the Dixon 2005 paper as written. See the parity
spike report at `_DEV/SPIKE_OUTCOME.md` for the validation that
established Cantus's algorithmic independence.
