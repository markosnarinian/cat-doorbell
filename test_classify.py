"""
test_classify.py – Send miaow_16k.wav to the /waveform endpoint and print results.

Usage:
    python test_classify.py
"""

import json
import sys
from pathlib import Path

import requests

ENDPOINT = "http://localhost:8000/waveform"
WAV_FILE = Path(__file__).parent / "miaow_16k.wav"


def main() -> None:
    if not WAV_FILE.exists():
        print(f"ERROR: WAV file not found at {WAV_FILE}", file=sys.stderr)
        sys.exit(1)

    print(f"Posting {WAV_FILE.name} to {ENDPOINT} …")

    with WAV_FILE.open("rb") as f:
        response = requests.post(
            ENDPOINT,
            files={"file": (WAV_FILE.name, f, "audio/wav")},
            timeout=30,
        )

    print(f"Status: {response.status_code}")

    if not response.ok:
        print(f"ERROR: {response.text}", file=sys.stderr)
        sys.exit(1)

    result = response.json()
    cat_detected = result.get("cat_detected", False)
    top_results = result.get("top_results", [])

    print(f"\ncat_detected: {cat_detected}")
    print("\nTop predictions:")
    for label, score in top_results[:10]:
        bar = "█" * int(score * 40)
        print(f"  {label:<30} {score:.4f}  {bar}")


if __name__ == "__main__":
    main()
