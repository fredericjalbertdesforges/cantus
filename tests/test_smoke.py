"""Smoke tests: package imports cleanly and exposes the public API."""
from __future__ import annotations


def test_top_level_imports() -> None:
    import cantus

    assert cantus.__version__ == "0.0.1"
    # public surface
    for name in (
        "OLTWResult",
        "oltw_align",
        "FeatureExtractor",
        "PianoRollExtractor",
        "alignment_error_frames",
        "alignment_rate",
        "summarise",
    ):
        assert hasattr(cantus, name), f"missing public attr: {name}"


def test_subpackage_imports() -> None:
    from cantus.algorithms import oltw_align
    from cantus.features import FeatureExtractor, PianoRollExtractor

    assert callable(oltw_align)
    assert FeatureExtractor.__abstractmethods__
    assert PianoRollExtractor().spec.feature_dim == 88
