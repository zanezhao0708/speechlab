#!/usr/bin/env python
"""Download CMU Arctic utterances from the kth-tmh/cmu_arctic HF mirror.

Fetches individual 16 kHz wav files (no multi-hundred-MB tarballs) from
https://hf-mirror.com/datasets/kth-tmh/cmu_arctic (branch tars-unpacked),
which mirrors the permissive-licence CMU Arctic corpus (attribution
required, free for research use).

The sandbox egress is flaky (SSL resets), so every file is retried with
backoff.  Run::

    python benchmarks/download_cmu_arctic.py --out bench_data/raw

Downloads 4 speakers x 25 utterances = 100 files (~10 MB total).
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = ("https://hf-mirror.com/datasets/kth-tmh/cmu_arctic/"
        "resolve/tars-unpacked")
SPEAKERS = ("bdl", "rms", "slt", "clb")  # 2 male (bdl, rms) + 2 female
N_UTT = 25  # arctic_a0001 .. arctic_a0025 per speaker
RETRIES = 10


def fetch(url: str, dest: Path) -> tuple[str, str]:
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 speechlab-bench"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if len(data) < 1000 or not data.startswith(b"RIFF"):
                raise ValueError(f"suspicious payload ({len(data)} bytes)")
            dest.write_bytes(data)
            return dest.name, "ok"
        except Exception as exc:  # noqa: BLE001 — retry any network hiccup
            if attempt == RETRIES - 1:
                return dest.name, f"FAILED: {exc}"
            time.sleep(min(0.5 * (attempt + 1), 5.0))
    return dest.name, "unreachable"  # pragma: no cover


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="bench_data/raw",
                    help="destination directory (default: bench_data/raw)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for spk in SPEAKERS:
        for i in range(1, N_UTT + 1):
            name = f"{spk}_arctic_a{i:04d}.wav"
            dest = out / name
            if dest.exists() and dest.stat().st_size > 1000:
                continue  # resumable
            url = f"{BASE}/cmu_us_{spk}_arctic/wav/arctic_a{i:04d}.wav"
            jobs.append((url, dest))

    if not jobs:
        print(f"all {len(SPEAKERS) * N_UTT} files already present in {out}")
        return
    print(f"downloading {len(jobs)} files -> {out}")
    fails = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for name, status in pool.map(lambda j: fetch(*j), jobs):
            if status != "ok":
                fails.append((name, status))
                print(f"  {name}: {status}", file=sys.stderr)
    if fails:
        print(f"\n{len(fails)} file(s) failed — re-run to resume", file=sys.stderr)
        sys.exit(1)
    print(f"done: {len(jobs)} files in {out}")


if __name__ == "__main__":
    main()
