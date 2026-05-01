"""
Chroma features via constant-Q transform.

A 12-dim chromagram derived with ``librosa.feature.chroma_cqt`` —
the standard audio-domain baseline for piano-focused score-following.
For continuous-pitch sources (voice, violin, flute), prefer
:class:`cantus.features.pitch_trajectory.PitchTrajectoryExtractor`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from cantus.features.base import FeatureExtractor, FeatureSpec


class ChromaExtractor(FeatureExtractor):
    """12-dim chromagram extractor backed by ``librosa.feature.chroma_cqt``.

    Requires the ``cantus[audio]`` extra. Streaming mode is deferred to
    a future release; ``extract`` operates on a full waveform or audio
    file path.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        frame_rate: int = 30,
        n_chroma: int = 12,
    ) -> None:
        try:
            import librosa  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "ChromaExtractor requires librosa. "
                "Install with: pip install 'cantus[audio]'"
            ) from e

        self._sample_rate = sample_rate
        self._hop_length = sample_rate // frame_rate
        self._n_chroma = n_chroma
        self._spec = FeatureSpec(
            feature_dim=n_chroma,
            frame_rate=frame_rate,
            # CQT inherent latency ≈ half the analysis window in ms.
            # librosa default is n_fft = 2048 for chroma_cqt internals.
            latency_ms=1000.0 * 1024.0 / sample_rate,
            name=f"chroma_cqt@{frame_rate}fps_{sample_rate}Hz",
        )

    @property
    def spec(self) -> FeatureSpec:
        return self._spec

    def extract(self, source: object) -> NDArray[np.float32]:
        """Compute a chromagram from a waveform array or audio file path.

        Args:
            source: a 1-D ``numpy.ndarray`` waveform sampled at
                ``self.spec.frame_rate``-compatible sample rate, or a
                path-like pointing to an audio file readable by librosa.

        Returns:
            Chromagram of shape ``[T, n_chroma]``, ``float32``.
        """
        import librosa

        if isinstance(source, np.ndarray):
            waveform = source.astype(np.float32)
        elif isinstance(source, str | Path):
            waveform, _ = librosa.load(
                str(source), sr=self._sample_rate, mono=True
            )
        else:
            raise TypeError(
                f"ChromaExtractor.extract expected np.ndarray or path, "
                f"got {type(source).__name__}"
            )

        chroma = librosa.feature.chroma_cqt(
            y=waveform,
            sr=self._sample_rate,
            hop_length=self._hop_length,
            n_chroma=self._n_chroma,
        )
        result: NDArray[np.float32] = chroma.T.astype(np.float32)
        return result

    def step(self, chunk: NDArray[np.floating]) -> NDArray[np.float32] | None:
        raise NotImplementedError(
            "ChromaExtractor streaming is deferred to a future release. "
            "Use extract() with a full waveform for now."
        )
