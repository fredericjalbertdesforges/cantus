"""
Pluggable feature-extractor abstraction for cantus.

Subclass :class:`FeatureExtractor` to provide your own front-end (chroma,
CREPE/FCPE pitch trajectory, mel-spectrogram, phonetic posteriorgram,
piano roll, ...) — every cantus alignment algorithm consumes the
extractor's output through this single interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class FeatureSpec:
    """Static metadata about a feature extractor's output.

    Attributes:
        feature_dim: dimensionality of each frame vector.
        frame_rate: frames per second produced by this extractor.
        latency_ms: inherent processing latency from raw audio to a feature
            frame, including any internal buffering. Used to compose
            end-to-end latency budgets in real-time setups.
        name: short identifier used in logs and benchmark reports.
    """

    feature_dim: int
    frame_rate: int
    latency_ms: float
    name: str


class FeatureExtractor(ABC):
    """Abstract base class for cantus feature extractors.

    Two execution modes are exposed:

    - :meth:`extract` — offline, takes a full input (audio waveform, MIDI
      file, etc.) and returns the entire feature matrix at once. Used for
      reference-side feature extraction and for simulation benchmarks.
    - :meth:`step` — streaming, takes one chunk of input and returns one
      feature frame. Used for live performance tracking.

    Subclasses must implement both. Implementations that do not support
    one mode should raise :class:`NotImplementedError` with a clear message.
    """

    @property
    @abstractmethod
    def spec(self) -> FeatureSpec:
        """Return static metadata about the extractor's output."""

    @abstractmethod
    def extract(self, source: object) -> NDArray[np.float32]:
        """Compute the full feature matrix for an offline source.

        Args:
            source: the input to extract from. Concrete type depends on
                the subclass (file path, numpy array, MIDI object, ...).

        Returns:
            Feature matrix of shape ``[T, feature_dim]``.
        """

    @abstractmethod
    def step(self, chunk: NDArray[np.floating]) -> NDArray[np.float32] | None:
        """Process one streaming chunk and emit zero or one feature frame.

        Args:
            chunk: raw input chunk (audio buffer, MIDI events, etc.).

        Returns:
            A single feature vector of shape ``[feature_dim]`` if a frame
            is ready, or ``None`` if more input is needed before the next
            frame can be emitted.
        """
