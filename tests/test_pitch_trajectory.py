"""Tests for PitchTrajectoryExtractor and helpers."""
from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from cantus.features.pitch_trajectory import (
    PitchTrajectoryExtractor,
    f0_to_one_hot,
    hz_to_midi,
)


def test_hz_to_midi_a4_is_69() -> None:
    assert abs(float(hz_to_midi(440.0)) - 69.0) < 1e-3


def test_hz_to_midi_c4_is_60() -> None:
    assert abs(float(hz_to_midi(261.6256)) - 60.0) < 1e-2


def test_hz_to_midi_zero_is_unvoiced_sentinel() -> None:
    out = hz_to_midi(np.array([0.0, 440.0, -1.0]))
    assert out[0] < 0
    assert abs(out[1] - 69.0) < 1e-3
    assert out[2] < 0


def test_f0_to_one_hot_basic() -> None:
    f0 = np.array([440.0, 440.0, 261.6256])
    voicing = np.array([0.9, 0.1, 0.8])
    one_hot = f0_to_one_hot(f0, voicing)
    assert one_hot.shape == (3, 88)
    # Frame 0: A4 voiced. MIDI 69, low_pitch=21 → idx 48.
    assert one_hot[0, 48] == 1.0
    assert one_hot[0].sum() == 1.0
    # Frame 1: voicing < threshold → all zeros.
    assert one_hot[1].sum() == 0.0
    # Frame 2: C4 voiced. MIDI 60 → idx 39.
    assert one_hot[2, 39] == 1.0


def test_f0_to_one_hot_clips_out_of_range() -> None:
    """Frequencies outside the piano range emit zero rows."""
    f0 = np.array([20.0, 8000.0])  # below A0, above C8
    voicing = np.array([0.9, 0.9])
    one_hot = f0_to_one_hot(f0, voicing)
    assert one_hot.sum() == 0.0


def test_f0_to_one_hot_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        f0_to_one_hot(np.array([440.0, 220.0]), np.array([0.9]))


def test_extractor_with_synthetic_tracker() -> None:
    expected_f0 = np.array([440.0, 261.6256, 0.0, 220.0], dtype=np.float32)
    expected_voicing = np.array([0.9, 0.9, 0.0, 0.9], dtype=np.float32)

    def fake_tracker(
        audio: NDArray[np.floating], sample_rate: int
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        return expected_f0, expected_voicing

    extractor = PitchTrajectoryExtractor(
        fake_tracker, sample_rate=22050, frame_rate=30
    )
    features = extractor.extract(np.zeros(22050, dtype=np.float32))
    assert features.shape == (4, 88)
    # Frame 0: A4 → MIDI 69 → idx 48
    assert features[0, 48] == 1.0
    # Frame 2: unvoiced
    assert features[2].sum() == 0.0
    # Frame 3: A3 → MIDI 57 → idx 36
    assert features[3, 36] == 1.0


def test_extractor_spec() -> None:
    def stub(
        audio: NDArray[np.floating], sample_rate: int
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        return (
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )

    extractor = PitchTrajectoryExtractor(
        stub, sample_rate=44100, frame_rate=100, tracker_latency_ms=12.5
    )
    spec = extractor.spec
    assert spec.feature_dim == 88
    assert spec.frame_rate == 100
    assert spec.latency_ms == 12.5


def test_extractor_low_high_validation() -> None:
    def stub(
        audio: NDArray[np.floating], sample_rate: int
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        return (
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )

    with pytest.raises(ValueError):
        PitchTrajectoryExtractor(stub, low_pitch=80, high_pitch=40)


def test_extractor_step_not_implemented() -> None:
    def stub(
        audio: NDArray[np.floating], sample_rate: int
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        return (
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )

    extractor = PitchTrajectoryExtractor(stub)
    with pytest.raises(NotImplementedError):
        extractor.step(np.zeros(1024, dtype=np.float32))


def test_extractor_rejects_unknown_input() -> None:
    def stub(
        audio: NDArray[np.floating], sample_rate: int
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        return (
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )

    extractor = PitchTrajectoryExtractor(stub)
    with pytest.raises(TypeError):
        extractor.extract(42)
