"""
cantus — real-time score-following for voice and continuous-pitch instruments.
"""
from cantus.algorithms.oltw_dixon import OLTWResult, oltw_align
from cantus.features.base import FeatureExtractor
from cantus.features.piano_roll import PianoRollExtractor
from cantus.metrics import (
    alignment_error_frames,
    alignment_rate,
    summarise,
)

__version__ = "0.0.1"

__all__ = [
    "OLTWResult",
    "oltw_align",
    "FeatureExtractor",
    "PianoRollExtractor",
    "alignment_error_frames",
    "alignment_rate",
    "summarise",
    "__version__",
]
