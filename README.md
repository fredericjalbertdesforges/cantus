# cantus

Real-time score-following for voice and continuous-pitch instruments.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A focused, well-tested Python library for real-time alignment of monophonic continuous-pitch performance audio (voice, violin, flute, sax, oboe, clarinet) to a reference score.

**Documentation site**: [fredjalbertdesforges.com/papers/cantus](https://fredjalbertdesforges.com/papers/cantus/) (forthcoming)

## Why

The score-following landscape is dominated by tools designed for piano:
- [**pymatchmaker**](https://github.com/pymatchmaker/matchmaker) (Park 2025, ISMIR) explicitly limits itself to classical piano.
- [**ACCompanion**](https://github.com/CPJKU/accompanion) is a symbolic-only piano accompanist.
- [**Antescofo**](https://www.antescofo.com) is closed-source.

Voice, bowed strings, and wind instruments have **continuous-pitch** acoustics: chroma features fragment under vibrato, glissando, and portamento. A unified open-source library for real-time score-following on these sources does not currently exist.

`cantus` fills that gap.

## Status

**Pre-alpha (v0.0.1).** API is unstable. Under active development.

## The 9 ideas behind cantus

1. **Pluggable `FeatureExtractor`** — one ABC, many subclasses (chroma, piano-roll, CREPE/FCPE pitch trajectories, phonetic posteriorgrams).
2. **Algorithm-agnostic alignment** — OLTW Dixon and OLTW Arzt-Widmer share the same feature interface.
3. **Continuous-pitch first** — defaults tuned for monophonic non-piano sources, not piano-on-piano.
4. **Streaming and simulation share the same loop** — single source of truth.
5. **Canonical algorithms** — Dixon 2005 implemented to the paper, with explicit design choices documented (see `CHANGELOG.md`).
6. **Latency-aware metrics** — every feature extractor exposes its inherent latency in milliseconds.
7. **Datasets module** — adapters for Vocadito, MusicNet, Bach10, URMP, Violin Etudes.
8. **No live-audio dependency by default** — `cantus[audio]` is opt-in.
9. **JOSS-quality engineering** — type-checked, linted, tested across OS and Python versions.

## Installation

```bash
pip install cantus
```

For audio features and live streaming:

```bash
pip install "cantus[audio]"
```

## Quick start

```python
import numpy as np
from cantus.algorithms import oltw_align
from cantus.features import PianoRoll

# both reference and performance as 88-key piano rolls at 30 fps
ref  = ...  # shape [T_ref, 88]
perf = ...  # shape [T_perf, 88]

result = oltw_align(reference=ref, performance=perf, search_width=200)
print(result.alignment[:10])  # estimated reference frame for each perf frame
```

## Reproducibility

Each algorithm is unit-tested against (a) theoretical anchors on canonical inputs (identity warp, integer-stretch warp, synthetic rubato with known ground truth) and (b) parity against the Dixon 2005 paper as written. The test suite runs in under one second.

## Citation

If you use `cantus` in your research, please cite the archived release via its DOI (forthcoming). The `CITATION.cff` file at the repo root carries the same metadata in machine-readable form.

## License

Apache 2.0. See [LICENSE](LICENSE).

## Author

Fred Jalbert-Desforges, independent researcher in computational music analysis, Montreal, Quebec.

- Personal site: [fredjalbertdesforges.com](https://fredjalbertdesforges.com)
- ORCID: [0009-0002-4357-6942](https://orcid.org/0009-0002-4357-6942)
- E-mail: [fred@fredjalbertdesforges.com](mailto:fred@fredjalbertdesforges.com)

## Acknowledgements

Built within the [Cygnus Analysis](https://cygnusanalysis.com) research program. Companion library to [vega-mir](https://github.com/fredericjalbertdesforges/vega-mir) (information-theoretic analysis of symbolic music).
