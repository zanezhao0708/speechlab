"""Analyse a single recording with SpeechLab.

Usage: python examples/analyze_single.py path/to/audio.wav
"""

import json
import sys

from speechlab.audio import load_audio
from speechlab.features import analyze, f0_track
from speechlab.quality import quality_report


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    audio = load_audio(sys.argv[1], target_sr=16000)

    report = analyze(audio)
    report["quality"] = quality_report(audio)

    track = f0_track(audio.samples, audio.sample_rate)
    voiced = track.f0[track.voiced]
    if len(voiced):
        report["f0_contour_first_10_hz"] = [round(v, 1) for v in voiced[:10]]

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
