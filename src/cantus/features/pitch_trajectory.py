"""
Pitch-trajectory features for monophonic continuous-pitch sources.

The differentiator angle of cantus: chroma fragments under vibrato,
glissando, and portamento on voice and bowed strings. A feature derived
from the f0 trajectory of a monophonic pitch tracker (CREPE, FCPE, pYIN)
remains stable across these expressive variations.

This module ships a **pluggable** extractor: pass any callable that takes
``(waveform, sample_rate) -> (f0_hz, voicing)`` and the extractor turns
its output into a binary 88-key feature compatible with cantus alignment
algorithms. Concrete CREPE/FCPE adapters are scheduled for v0.0.2.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from cantus.features.base import FeatureExtractor, FeatureSpec
from cantus.features.piano_roll import PIANO_HIGH, PIANO_LOW

PitchTracker = Callable[
    [NDArray[np.floating], int],
    tuple[NDArray[np.float32], NDArray[np.float32]],
]
"""A monophonic pitch tracker.

Takes a waveform and its sample rate, returns ``(f0_hz, voicing)``:
both 1-D arrays of equal length where ``f0_hz`` is the estimated
fundamental in Hz (any value when unvoiced) and ``voicing`` is a
confidence in ``[0, 1]``.
"""


def hz_to_midi(hz: float | NDArray[np.floating]) -> NDArray[np.float32]:
    """Convert frequency in Hz to fractional MIDI pitch number.

    Returns ``-1`` for non-positive frequencies (sentinel for unvoiced).
    """
    arr = np.asarray(hz, dtype=np.float32)
    midi = np.where(
        arr > 0,
        69.0 + 12.0 * np.log2(np.clip(arr, 1e-6, None) / 440.0),
        -1.0,
    )
    return midi.astype(np.float32)


def f0_to_one_hot(
    f0_hz: NDArray[np.floating],
    voicing: NDArray[np.floating],
    low_pitch: int = PIANO_LOW,
    high_pitch: int = PIANO_HIGH,
    voicing_threshold: float = 0.5,
) -> NDArray[np.float32]:
    """Convert a pitch trajectory to a binary piano-roll-style feature.

    Args:
        f0_hz: per-frame f0 in Hz, shape ``[T]``.
        voicing: per-frame voicing confidence in ``[0, 1]``, shape ``[T]``.
        low_pitch: lowest MIDI note represented (default 21 = A0).
        high_pitch: highest MIDI note represented (default 108 = C8).
        voicing_threshold: frames whose voicing falls below this are
            emitted as all-zero rows.

    Returns:
        Binary feature matrix of shape ``[T, high_pitch - low_pitch + 1]``.
    """
    f0_arr = np.asarray(f0_hz, dtype=np.float32)
    voicing_arr = np.asarray(voicing, dtype=np.float32)
    if f0_arr.shape != voicing_arr.shape:
        raise ValueError(
            f"f0_hz and voicing must have the same shape, got "
            f"{f0_arr.shape} vs {voicing_arr.shape}"
        )

    n_frames = len(f0_arr)
    n_keys = high_pitch - low_pitch + 1
    out = np.zeros((n_frames, n_keys), dtype=np.float32)
    midi = hz_to_midi(f0_arr)
    rounded = np.round(midi).astype(np.int64)
    voiced_mask = voicing_arr >= voicing_threshold
    for t in range(n_frames):
        if not voiced_mask[t]:
            continue
        idx = int(rounded[t]) - low_pitch
        if 0 <= idx < n_keys:
            out[t, idx] = 1.0
    return out


class PitchTrajectoryExtractor(FeatureExtractor):
    """Pitch-trajectory extractor for continuous-pitch monophonic sources.

    Pluggable: takes any ``PitchTracker`` callable. The extractor binds
    the tracker to a binary 88-key feature representation that aligns
    well with symbolic piano-roll references — useful for benchmarking
    pitch trackers against MIDI ground truth.

    Concrete adapters for CREPE (Kim et al. 2018) and FCPE (2025) ship
    in cantus v0.0.2.

    Example:
        >>> def yin_tracker(audio, sr):
        ...     # any tracker producing (f0_hz, voicing)
        ...     return f0, voicing
        >>> ext = PitchTrajectoryExtractor(
        ...     yin_tracker, sample_rate=22050, frame_rate=30
        ... )
        >>> features = ext.extract(audio)  # shape [T, 88]
    """

    def __init__(
        self,
        pitch_tracker: PitchTracker,
        sample_rate: int = 22050,
        frame_rate: int = 30,
        low_pitch: int = PIANO_LOW,
        high_pitch: int = PIANO_HIGH,
        voicing_threshold: float = 0.5,
        tracker_latency_ms: float = 0.0,
    ) -> None:
        if low_pitch >= high_pitch:
            raise ValueError(
                f"low_pitch ({low_pitch}) must be < high_pitch ({high_pitch})"
            )
        self._pitch_tracker = pitch_tracker
        self._sample_rate = sample_rate
        self._low_pitch = low_pitch
        self._high_pitch = high_pitch
        self._voicing_threshold = voicing_threshold
        self._spec = FeatureSpec(
            feature_dim=high_pitch - low_pitch + 1,
            frame_rate=frame_rate,
            latency_ms=tracker_latency_ms,
            name=f"pitch_trajectory@{frame_rate}fps",
        )

    @property
    def spec(self) -> FeatureSpec:
        return self._spec

    def extract(self, source: object) -> NDArray[np.float32]:
        """Run the pitch tracker and convert to a binary feature.

        Args:
            source: a 1-D ``numpy.ndarray`` waveform, or a path to an
                audio file readable by librosa.
        """
        if isinstance(source, np.ndarray):
            waveform = source.astype(np.float32)
        elif isinstance(source, str | Path):
            try:
                import librosa
            except ImportError as e:
                raise ImportError(
                    "Loading audio from a path requires librosa. "
                    "Install with: pip install 'cantus[audio]'"
                ) from e
            waveform, _ = librosa.load(
                str(source), sr=self._sample_rate, mono=True
            )
        else:
            raise TypeError(
                f"PitchTrajectoryExtractor.extract expected np.ndarray "
                f"or path, got {type(source).__name__}"
            )

        f0, voicing = self._pitch_tracker(waveform, self._sample_rate)
        return f0_to_one_hot(
            f0,
            voicing,
            low_pitch=self._low_pitch,
            high_pitch=self._high_pitch,
            voicing_threshold=self._voicing_threshold,
        )

    def step(self, chunk: NDArray[np.floating]) -> NDArray[np.float32] | None:
        raise NotImplementedError(
            "PitchTrajectoryExtractor streaming is deferred to v0.0.2 "
            "alongside CREPE/FCPE adapters."
        )
