"""Corpus statistics and a speaker-independent split.

Usage: python examples/dataset_report.py /path/to/corpus
"""

import json
import sys

from speechlab.dataset import dataset_report, scan_dataset, speaker_independent_split


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    utterances = scan_dataset(sys.argv[1])
    report = dataset_report(utterances)
    print("--- corpus report ---")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    split = speaker_independent_split(utterances, 0.8, 0.1, 0.1, seed=42)
    print("--- speaker-independent split ---")
    for name, paths in split.items():
        print(f"{name}: {len(paths)} files")


if __name__ == "__main__":
    main()
