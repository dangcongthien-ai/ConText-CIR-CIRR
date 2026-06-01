from __future__ import annotations

import argparse
import json
from pathlib import Path


HARD_THRESHOLDS = {
    "zero_phrase_pct": ("<=", 0.01),
    "prep_first_phrase_pct": ("<=", 0.02),
    "pronoun_only_phrase_pct": ("<=", 0.01),
    "edit_verb_phrase_pct": ("<=", 0.10),
    "attribute_capture_rate_make": (">=", 0.65),
    "attribute_capture_rate_change": (">=", 0.60),
}

SOFT_THRESHOLDS = {
    "unknown_edit_action_pct": ("<=", 0.15),
    "attribute_capture_rate_overall_edit_heavy": (">=", 0.30),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether parser audit metrics are strong enough to proceed to training.")
    parser.add_argument("--input", required=True, help="Path to the audit JSON report.")
    return parser.parse_args()


def _passes(value: float, operator: str, threshold: float) -> bool:
    if operator == "<=":
        return value <= threshold
    if operator == ">=":
        return value >= threshold
    raise ValueError(f"Unsupported operator: {operator}")


def _format_rule(metric: str, operator: str, threshold: float, value: float) -> str:
    status = "PASS" if _passes(value, operator, threshold) else "FAIL"
    return f"{status} | {metric}: {value:.4f} {operator} {threshold:.4f}"


def main() -> int:
    args = parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))

    looks_legacy_or_incomplete = (
        float(payload.get("attribute_capture_rate_make", 0.0) or 0.0) == 0.0
        and float(payload.get("attribute_capture_rate_change", 0.0) or 0.0) == 0.0
        and float(payload.get("attribute_capture_rate_overall_edit_heavy", 0.0) or 0.0) == 0.0
        and int(payload.get("num_unknown_edit_actions", 0) or 0) == 0
    )

    hard_failures: list[str] = []
    soft_failures: list[str] = []

    print("Parser audit gate")
    print(f"input: {args.input}")

    if looks_legacy_or_incomplete:
        print("\nResult: HOLD")
        print("Reason: this audit looks like it came from a legacy or incomplete export without the new edit_actions fields.")
        print("Action: rerun export with the current parser, then regenerate audit from that fresh JSON before training.")
        return 2

    print("\nHard checks:")
    for metric, (operator, threshold) in HARD_THRESHOLDS.items():
        value = float(payload.get(metric, 0.0) or 0.0)
        line = _format_rule(metric, operator, threshold, value)
        print(line)
        if not _passes(value, operator, threshold):
            hard_failures.append(line)

    print("\nSoft checks:")
    for metric, (operator, threshold) in SOFT_THRESHOLDS.items():
        value = float(payload.get(metric, 0.0) or 0.0)
        line = _format_rule(metric, operator, threshold, value)
        print(line)
        if not _passes(value, operator, threshold):
            soft_failures.append(line)

    if hard_failures:
        print("\nResult: HOLD")
        print("Reason: at least one hard gate failed, so parser supervision still looks too noisy for the next training round.")
        return 1

    if soft_failures:
        print("\nResult: READY WITH WARNINGS")
        print("Reason: noun phrase supervision looks trainable, but edit-action quality still has some rough edges.")
        return 0

    print("\nResult: READY")
    print("Reason: parser supervision looks clean enough to move into training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
