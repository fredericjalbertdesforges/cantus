"""
Piano-roll feature extractor — symbolic baseline for MIDI inputs.

Used for parity benchmarking against piano-focused score-following tools
and for end-to-end testing without an audio synthesis dependency.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi
from numpy.typing import NDArray

from cantus.features.base import FeatureExtractor, FeatureSpec

PIANO_LOW = 21   # MIDI A0
PIANO_HIGH = 108  # MIDI C8 inclusive — 88 keys total


class PianoRollExtractor(FeatureExtractor):
    """88-key binary piano-roll extractor at a fixed frame rate.

    Streaming mode is not implemented for this extractor — piano rolls
    are an offline-only baseline used for tests and benchmarks. Use a
    chroma or pitch-trajectory extractor for live audio.
    """

    def __init__(self, frame_rate: int = 30) -> None:
        self._spec = FeatureSpec(
            feature_dim=PIANO_HIGH - PIANO_LOW + 1,  # 88
            frame_rate=frame_rate,
            latency_ms=0.0,
            name=f"piano_roll@{frame_rate}fps",
        )

    @property
    def spec(self) -> FeatureSpec:
        return self._spec

    def extract(self, source: object) -> NDArray[np.float32]:
        """Convert a MIDI file path or :class:`pretty_midi.PrettyMIDI` to a
        binary 88-key piano roll of shape ``[T, 88]``.
        """
        if isinstance(source, str | Path):
            pm = pretty_midi.PrettyMIDI(str(source))
        elif isinstance(source, pretty_midi.PrettyMIDI):
            pm = source
        else:
            raise TypeError(
                f"PianoRollExtractor.extract expected a path or "
                f"PrettyMIDI, got {type(source).__name__}"
            )
        roll = pm.get_piano_roll(fs=self._spec.frame_rate)
        roll = roll[PIANO_LOW : PIANO_HIGH + 1]
        binary: NDArray[np.float32] = (roll > 0).astype(np.float32).T
        return binary  # [T, 88]

    def step(self, chunk: NDArray[np.floating]) -> NDArray[np.float32] | None:
        raise NotImplementedError(
            "PianoRollExtractor does not support streaming mode. "
            "Use an audio-domain extractor (chroma, CREPE, FCPE) instead."
        )
