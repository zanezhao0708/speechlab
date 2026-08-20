"""SpeechLab: an LLM research agent for speech science.

The agent talks to any OpenAI-compatible chat API and grounds its answers
in locally measured acoustics through tool calling.

Modules
-------
audio    : loading, resampling and basic audio utilities.
features : the acoustic analysis core the agent calls as a tool
           (F0, formants, jitter/shimmer, HNR).
agent    : the LLM research assistant with the tool-calling loop.
cli      : command line interface.
"""

__version__ = "0.3.0"

__all__ = [
    "__version__",
    "agent",
    "audio",
    "cli",
    "features",
]
