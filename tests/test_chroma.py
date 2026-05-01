"""Tests for ChromaExtractor."""
from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

librosa = pytest.importorskip("librosa")

from cantus.features.chroma import ChromaExtractor  # noqa: E402


def _make_sine(freq_hz: float, duration_s: float, sr: int = 22050) -> NDArray[np.float32]:
    t = np.arange(int(duration_s * sr)) / sr
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)


def test_chroma_spec() -> None:
    extractor = ChromaExtractor(sample_rate=22050, frame_rate=30)
    spec = extractor.spec
    assert spec.feature_dim == 12
    assert spec.frame_rate == 30
    assert spec.latency_ms > 0
    assert "chroma_cqt" in spec.name


def test_chroma_concentrates_on_target_pitch_class() -> None:
    """A pure C4 sine should put most energy on chroma bin 0 (C)."""
    sample_rate = 22050
    audio = _make_sine(261.6, duration_s=2.0, sr=sample_rate)
    extractor = ChromaExtractor(sample_rate=sample_rate, frame_rate=30)
    chroma = extractor.extract(audio)
    assert chroma.shape[1] == 12
    average = chroma.mean(axis=0)
    # librosa convention: chroma index 0 = C
    assert int(np.argmax(average)) == 0


def test_chroma_concentrates_on_a_for_a4_sine() -> None:
    """A pure A4 sine should put most energy on chroma bin 9 (A)."""
    sample_rate = 22050
    audio = _make_sine(440.0, duration_s=2.0, sr=sample_rate)
    extractor = ChromaExtractor(sample_rate=sample_rate, frame_rate=30)
    chroma = extractor.extract(audio)
    average = chroma.mean(axis=0)
    assert int(np.argmax(average)) == 9


def test_chroma_rejects_unknown_input() -> None:
    extractor = ChromaExtractor()
    with pytest.raises(TypeError):
        extractor.extract(42)


def test_chroma_streaming_not_implemented() -> None:
    extractor = ChromaExtractor()
    with pytest.raises(NotImplementedError):
        extractor.step(np.zeros(1024, dtype=np.float32))
