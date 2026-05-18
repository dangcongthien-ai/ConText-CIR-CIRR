from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cir_pipeline.config import load_config
from cir_pipeline.data.validate_cirr_files import inspect_cirr_files
from cir_pipeline.utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect CIRR metadata and image availability.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--scan-extra", action="store_true", help="Also count extra PNGs in image folders.")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    report = inspect_cirr_files(cfg, scan_extra=args.scan_extra)
    print(json.dumps(report, indent=2))
    if args.output:
        write_json(report, args.output)


if __name__ == "__main__":
    main()
