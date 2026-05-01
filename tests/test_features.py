"""Tests for cantus.features."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cantus.features import FeatureExtractor, PianoRollExtractor


def test_feature_extractor_is_abstract() -> None:
    with pytest.raises(TypeError):
        FeatureExtractor()  # type: ignore[abstract]


def test_piano_roll_spec() -> None:
    extractor = PianoRollExtractor(frame_rate=50)
    spec = extractor.spec
    assert spec.feature_dim == 88
    assert spec.frame_rate == 50
    assert spec.latency_ms == 0.0
    assert "piano_roll" in spec.name


def test_piano_roll_extract_from_pretty_midi(tmp_path: Path) -> None:
    """Build a tiny synthetic MIDI in memory and verify the roll shape."""
    pretty_midi = pytest.importorskip("pretty_midi")
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    # one note from t=0 to t=1
    inst.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=1.0))
    pm.instruments.append(inst)

    extractor = PianoRollExtractor(frame_rate=30)
    roll = extractor.extract(pm)
    assert roll.shape[1] == 88
    assert roll.shape[0] >= 25  # ~30 frames for 1s
    # MIDI 60 = C4 = index 60 - PIANO_LOW(21) = 39
    assert roll[:25, 39].sum() > 0


def test_piano_roll_streaming_not_implemented() -> None:
    extractor = PianoRollExtractor()
    with pytest.raises(NotImplementedError):
        extractor.step(np.zeros(1024, dtype=np.float32))


def test_piano_roll_extract_rejects_unknown_type() -> None:
    extractor = PianoRollExtractor()
    with pytest.raises(TypeError):
        extractor.extract(42)
