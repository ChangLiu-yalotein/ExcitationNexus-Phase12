#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from excitationnexus_phase12.gate1b3_evaluation import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--unlock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args()
    result = evaluate(args.config, args.unlock, args.output, args.physical_gpu)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
