from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

EDIT_HEAVY_VERBS = ("make", "change", "turn", "remove", "add", "replace", "put", "place", "move")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample edit-heavy CIRR captions for manual audit.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["samples"] if isinstance(payload, dict) and "samples" in payload else payload
    if not isinstance(rows, list):
        raise ValueError(f"Expected samples list in {path}")
    return rows


def _is_edit_heavy(row: dict[str, Any]) -> bool:
    caption = str(row.get("caption") or "").lower()
    if any(caption.startswith(f"{verb} ") or f" {verb} " in caption for verb in EDIT_HEAVY_VERBS):
        return True
    return bool(row.get("edit_actions"))


def main() -> None:
    args = parse_args()
    rows = [row for row in _load_rows(Path(args.input)) if _is_edit_heavy(row)]
    random.Random(args.seed).shuffle(rows)
    selected = rows[: max(1, args.n)]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pair_id",
                "caption",
                "noun_phrases",
                "edit_actions",
                "manual_object_correct",
                "manual_attribute_correct",
                "manual_notes",
            ],
        )
        writer.writeheader()
        for row in selected:
            writer.writerow(
                {
                    "pair_id": row.get("pair_id"),
                    "caption": row.get("caption"),
                    "noun_phrases": json.dumps(row.get("noun_phrases") or [], ensure_ascii=False),
                    "edit_actions": json.dumps(row.get("edit_actions") or [], ensure_ascii=False),
                    "manual_object_correct": "",
                    "manual_attribute_correct": "",
                    "manual_notes": "",
                }
            )
    print(output_path)


if __name__ == "__main__":
    main()
