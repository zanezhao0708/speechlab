"""SpeechLab: a toolkit and LLM agent for speech-science research.

Modules
-------
audio    : loading, resampling and basic audio utilities.
features : acoustic feature extraction (F0, formants, MFCC, energy, jitter/shimmer, HNR).
quality  : recording quality checks (clipping, DC offset, SNR, silence ratio).
dataset  : corpus statistics and speaker-independent dataset splitting.
agent    : an LLM research assistant that can call the local analysis tools.
cli      : command line interface.
"""

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "agent",
    "audio",
    "cli",
    "dataset",
    "features",
    "quality",
]
