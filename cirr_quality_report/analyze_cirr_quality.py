from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import hashlib
import json
import math
import random
import subprocess

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "CIRR"
META_ROOT = DATA_ROOT / "cirr"
OUT = Path(__file__).resolve().parent
SPLITS = ["train", "val", "test1"]
IMAGE_DIR = {"train": "train", "val": "dev", "test1": "test1"}

COLORS = {
    "blue": (47, 95, 159),
    "green": (79, 157, 105),
    "orange": (208, 140, 53),
    "purple": (141, 95, 179),
    "red": (182, 75, 75),
    "teal": (57, 125, 125),
    "gray": (108, 127, 143),
    "brown": (154, 106, 79),
    "ink": (34, 34, 34),
    "grid": (225, 229, 234),
}

try:
    FONT = ImageFont.truetype("arial.ttf", 12)
    FONT_SMALL = ImageFont.truetype("arial.ttf", 10)
    FONT_TITLE = ImageFont.truetype("arial.ttf", 16)
except Exception:
    FONT = ImageFont.load_default()
    FONT_SMALL = FONT
    FONT_TITLE = FONT

JSON_DUP_KEYS: dict[str, Counter] = defaultdict(Counter)


def dup_hook(label: str):
    def hook(pairs):
        seen = set()
        out = {}
        for key, value in pairs:
            if key in seen:
                JSON_DUP_KEYS[label][key] += 1
            seen.add(key)
            out[key] = value
        return out

    return hook


def load_json(path: Path, label: str):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f, object_pairs_hook=dup_hook(label))


def norm_json_path(value: str) -> str:
    value = str(value).replace("\\", "/")
    return value[2:] if value.startswith("./") else value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def words(text) -> list[str]:
    return [w for w in str(text).strip().split() if w]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(values: list[int], pct: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(values[lo])
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def git_info(path: Path) -> dict:
    def run(args):
        try:
            cp = subprocess.run(
                args,
                cwd=path,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            return cp.stdout.strip()
        except Exception:
            return ""

    status = run(["git", "status", "--short"])
    remote = run(["git", "remote", "-v"]).splitlines()
    return {
        "branch": run(["git", "branch", "--show-current"]),
        "commit": run(["git", "rev-parse", "HEAD"]),
        "status_short": status,
        "status_clean": status == "",
        "remote": remote[0] if remote else "",
    }


def draw_text(draw: ImageDraw.ImageDraw, xy, text, font=FONT, fill=COLORS["ink"], anchor=None):
    draw.text(xy, str(text), fill=fill, font=font, anchor=anchor)


def axes(draw: ImageDraw.ImageDraw, box, title: str, xlabel="", ylabel="", max_y: float | None = None):
    x0, y0, x1, y1 = box
    draw.rectangle([x0, y0, x1, y1], outline=(180, 187, 196))
    draw_text(draw, ((x0 + x1) // 2, 18), title, FONT_TITLE, anchor="ma")
    if xlabel:
        draw_text(draw, ((x0 + x1) // 2, y1 + 36), xlabel, FONT_SMALL, anchor="ma")
    if ylabel:
        draw_text(draw, (12, (y0 + y1) // 2), ylabel, FONT_SMALL)
    if max_y and max_y > 0:
        for i in range(6):
            val = max_y * i / 5
            y = y1 - (y1 - y0) * i / 5
            draw.line([x0, y, x1, y], fill=COLORS["grid"])
            draw_text(draw, (x0 - 8, y - 7), f"{int(val):,}", FONT_SMALL, fill=(90, 96, 106), anchor="ra")


def grouped_bar(path: Path, title: str, categories: list[str], series: list[tuple[str, list[int], tuple]]):
    width, height = 1100, 620
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    box = (90, 70, width - 60, height - 110)
    max_y = max(max(vals) for _, vals, _ in series) * 1.12
    axes(draw, box, title, ylabel="count", max_y=max_y)
    x0, y0, x1, y1 = box
    group_w = (x1 - x0) / len(categories)
    bar_w = group_w * 0.72 / len(series)
    for gi, cat in enumerate(categories):
        gx = x0 + gi * group_w + group_w * 0.14
        for si, (_, vals, color) in enumerate(series):
            val = vals[gi]
            bx0 = gx + si * bar_w
            bx1 = bx0 + bar_w * 0.88
            by0 = y1 - val / max_y * (y1 - y0)
            draw.rectangle([bx0, by0, bx1, y1], fill=color)
            draw_text(draw, ((bx0 + bx1) / 2, by0 - 15), f"{val:,}", FONT_SMALL, anchor="ma")
        draw_text(draw, (x0 + gi * group_w + group_w / 2, y1 + 10), cat, FONT, anchor="ma")
    lx, ly = x0, height - 65
    for name, _, color in series:
        draw.rectangle([lx, ly, lx + 14, ly + 14], fill=color)
        draw_text(draw, (lx + 20, ly - 1), name, FONT_SMALL)
        lx += 145
    image.save(path)


def stacked_bar(path: Path, title: str, categories: list[str], stacks: list[list[int]], names: list[str], colors: list[tuple]):
    width, height = 1000, 600
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    box = (90, 70, width - 60, height - 120)
    totals = [sum(row) for row in stacks]
    max_y = max(totals + [1]) * 1.15
    axes(draw, box, title, ylabel="issue count", max_y=max_y)
    x0, y0, x1, y1 = box
    group_w = (x1 - x0) / len(categories)
    bar_w = group_w * 0.42
    for gi, cat in enumerate(categories):
        cx = x0 + gi * group_w + group_w / 2
        top = y1
        for val, color in zip(stacks[gi], colors):
            h = val / max_y * (y1 - y0)
            if val:
                draw.rectangle([cx - bar_w / 2, top - h, cx + bar_w / 2, top], fill=color)
            top -= h
        draw.rectangle([cx - bar_w / 2, top, cx + bar_w / 2, y1], outline=(120, 126, 136))
        draw_text(draw, (cx, top - 15), totals[gi], FONT, anchor="ma")
        draw_text(draw, (cx, y1 + 10), cat, FONT, anchor="ma")
    lx, ly = x0, height - 75
    for name, color in zip(names, colors):
        draw.rectangle([lx, ly, lx + 14, ly + 14], fill=color)
        draw_text(draw, (lx + 20, ly - 1), name, FONT_SMALL)
        lx += 140
        if lx > width - 180:
            lx = x0
            ly += 22
    image.save(path)


def histogram(path: Path, title: str, values_by_split: dict[str, list[int]]):
    width, height = 1000, 600
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    box = (85, 70, width - 55, height - 105)
    all_values = [v for vals in values_by_split.values() for v in vals]
    max_value = max(all_values) if all_values else 1
    bins = list(range(max_value + 1))
    counts = {split: [Counter(vals).get(b, 0) for b in bins] for split, vals in values_by_split.items()}
    max_y = max([max(v) for v in counts.values()] + [1]) * 1.12
    axes(draw, box, title, xlabel="caption word count", ylabel="count", max_y=max_y)
    x0, y0, x1, y1 = box
    split_colors = {"train": COLORS["blue"], "val": COLORS["green"], "test1": COLORS["orange"]}
    group_w = (x1 - x0) / max(1, len(bins))
    bar_w = max(1, group_w * 0.82 / len(values_by_split))
    for bi, b in enumerate(bins):
        gx = x0 + bi * group_w
        for si, split in enumerate(values_by_split):
            val = counts[split][bi]
            h = val / max_y * (y1 - y0)
            bx = gx + si * bar_w
            draw.rectangle([bx, y1 - h, bx + bar_w, y1], fill=split_colors[split])
    step = max(1, math.ceil(max_value / 10))
    for b in range(0, max_value + 1, step):
        draw_text(draw, (x0 + b * group_w, y1 + 10), b, FONT_SMALL, anchor="ma")
    lx, ly = x0, height - 55
    for split, color in split_colors.items():
        draw.rectangle([lx, ly, lx + 14, ly + 14], fill=color)
        draw_text(draw, (lx + 20, ly - 1), split, FONT_SMALL)
        lx += 90
    image.save(path)


def dimensions_chart(path: Path, records: list[dict]):
    width, height = 1120, 560
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left = (75, 70, 530, 465)
    right = (630, 70, 1060, 465)
    axes(draw, left, "Image dimensions", xlabel="width", ylabel="height")
    axes(draw, right, "Aspect ratio distribution", xlabel="width / height", ylabel="count")
    split_colors = {"train": COLORS["blue"], "val": COLORS["green"], "test1": COLORS["orange"]}
    widths = [r["width"] for r in records]
    heights = [r["height"] for r in records]
    min_w, max_w = min(widths), max(widths)
    min_h, max_h = min(heights), max(heights)

    def sx(w):
        return left[0] + (w - min_w) / max(1, max_w - min_w) * (left[2] - left[0])

    def sy(h):
        return left[3] - (h - min_h) / max(1, max_h - min_h) * (left[3] - left[1])

    for rec in records:
        x, y = sx(rec["width"]), sy(rec["height"])
        draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=split_colors[rec["split"]])
    draw_text(draw, (left[0], left[3] + 8), min_w, FONT_SMALL, anchor="ma")
    draw_text(draw, (left[2], left[3] + 8), max_w, FONT_SMALL, anchor="ma")
    draw_text(draw, (left[0] - 8, left[3]), min_h, FONT_SMALL, anchor="ra")
    draw_text(draw, (left[0] - 8, left[1]), max_h, FONT_SMALL, anchor="ra")

    ratios = defaultdict(list)
    for rec in records:
        if rec["height"]:
            ratios[rec["split"]].append(rec["width"] / rec["height"])
    all_ratios = [r for vals in ratios.values() for r in vals]
    rmin, rmax = min(all_ratios), max(all_ratios)
    bins = 36
    counts = {}
    max_count = 1
    for split, vals in ratios.items():
        row = [0] * bins
        for val in vals:
            idx = min(bins - 1, int((val - rmin) / max(1e-9, rmax - rmin) * bins))
            row[idx] += 1
        counts[split] = row
        max_count = max(max_count, max(row))
    bar_w = (right[2] - right[0]) / bins / len(SPLITS)
    for bi in range(bins):
        for si, split in enumerate(SPLITS):
            val = counts.get(split, [0] * bins)[bi]
            h = val / max_count * (right[3] - right[1])
            bx = right[0] + bi * (right[2] - right[0]) / bins + si * bar_w
            draw.rectangle([bx, right[3] - h, bx + bar_w * 0.9, right[3]], fill=split_colors[split])
    draw_text(draw, (right[0], right[3] + 8), f"{rmin:.2f}", FONT_SMALL, anchor="ma")
    draw_text(draw, (right[2], right[3] + 8), f"{rmax:.2f}", FONT_SMALL, anchor="ma")
    lx, ly = 75, 500
    for split in SPLITS:
        draw.rectangle([lx, ly, lx + 14, ly + 14], fill=split_colors[split])
        draw_text(draw, (lx + 20, ly - 1), split, FONT_SMALL)
        lx += 85
    image.save(path)


def bucket_chart(path: Path, bucket_counts: Counter):
    width, height = 1300, 560
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    box = (65, 65, width - 35, height - 95)
    buckets = [str(i) for i in range(100)]
    vals = [bucket_counts.get(b, 0) for b in buckets]
    max_y = max(vals + [1]) * 1.12
    axes(draw, box, "Train image distribution across 100 subfolders", xlabel="train subfolder", ylabel="PNG count", max_y=max_y)
    x0, y0, x1, y1 = box
    bar_w = (x1 - x0) / len(buckets)
    for i, val in enumerate(vals):
        h = val / max_y * (y1 - y0)
        bx = x0 + i * bar_w
        draw.rectangle([bx, y1 - h, bx + bar_w * 0.85, y1], fill=COLORS["teal"])
        if i % 5 == 0:
            draw_text(draw, (bx + bar_w / 2, y1 + 8), buckets[i], FONT_SMALL, anchor="ma")
    image.save(path)


def sample_sheet(path: Path, split: str, records: list[dict], sample_size=20, cols=5):
    rng = random.Random(42)
    valid = sorted([r for r in records if r["split"] == split], key=lambda r: r["id"])
    sample = rng.sample(valid, min(sample_size, len(valid)))
    thumb_w, thumb_h, label_h = 160, 130, 30
    rows = math.ceil(len(sample) / cols)
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, rec in enumerate(sample):
        x = (idx % cols) * thumb_w
        y = (idx // cols) * (thumb_h + label_h)
        try:
            with Image.open(DATA_ROOT / rec["rel_path"]) as im:
                im = im.convert("RGB")
                im.thumbnail((thumb_w - 8, thumb_h - 8), Image.LANCZOS)
                sheet.paste(im, (x + (thumb_w - im.width) // 2, y + (thumb_h - im.height) // 2))
        except Exception:
            draw.rectangle([x + 4, y + 4, x + thumb_w - 4, y + thumb_h - 4], outline="red")
            draw_text(draw, (x + 8, y + 8), "open error", FONT, fill="red")
        label = rec["id"][:24]
        draw_text(draw, (x + 6, y + thumb_h + 6), label, FONT)
    sheet.save(path)


def audit_split(split: str, meta: dict, all_records: list[dict]) -> tuple[dict, list[int], set[str], set[str]]:
    split_json = meta["image_splits"]
    captions = meta["captions"]
    ext_caps = meta["captions_ext"]
    expected = {img_id: DATA_ROOT / norm_json_path(rel_path) for img_id, rel_path in split_json.items()}
    expected_paths = {p.resolve() for p in expected.values()}
    img_dir = DATA_ROOT / IMAGE_DIR[split]
    actual_paths = sorted(img_dir.rglob("*.png")) if img_dir.exists() else []
    actual_ids = [p.stem for p in actual_paths]
    actual_id_counts = Counter(actual_ids)

    missing_images = sorted([img_id for img_id, path in expected.items() if not path.exists()])
    extra_paths = sorted([p for p in actual_paths if p.resolve() not in expected_paths])
    path_id_mismatch = sorted([img_id for img_id, path in expected.items() if path.stem != img_id])
    duplicate_actual_ids = {img_id: count for img_id, count in actual_id_counts.items() if count > 1}

    zero_byte = []
    corrupt = []
    modes = Counter()
    formats = Counter()
    dims = Counter()
    aspect_ratios = []
    size_groups = defaultdict(list)
    for path in actual_paths:
        rel = path.relative_to(DATA_ROOT).as_posix()
        try:
            size_bytes = path.stat().st_size
        except OSError as exc:
            corrupt.append({"path": rel, "error": f"stat failed: {exc}"})
            continue
        if size_bytes == 0:
            zero_byte.append(rel)
        size_groups[size_bytes].append(path)
        try:
            with Image.open(path) as im:
                fmt = im.format
                mode = im.mode
                width, height = im.size
                im.verify()
            modes[mode] += 1
            formats[fmt] += 1
            dims[f"{width}x{height}"] += 1
            if height:
                aspect_ratios.append(width / height)
            all_records.append(
                {
                    "split": split,
                    "id": path.stem,
                    "rel_path": rel,
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "format": fmt,
                    "bytes": size_bytes,
                }
            )
        except Exception as exc:
            corrupt.append({"path": rel, "error": str(exc)})

    duplicate_hash_groups = []
    for size_bytes, paths in size_groups.items():
        if len(paths) < 2 or size_bytes == 0:
            continue
        hashes = defaultdict(list)
        for path in paths:
            try:
                hashes[sha256_file(path)].append(path.relative_to(DATA_ROOT).as_posix())
            except Exception:
                pass
        duplicate_hash_groups.extend(
            {"sha256": digest, "count": len(paths2), "sample_paths": paths2[:10]}
            for digest, paths2 in hashes.items()
            if len(paths2) > 1
        )
    duplicate_hash_groups.sort(key=lambda x: (-x["count"], x["sha256"]))

    split_img_ids = set(split_json.keys())
    required = ["pairid", "reference", "caption", "img_set"]
    if split in {"train", "val"}:
        required += ["target_hard", "target_soft"]

    cap_pairids = []
    cap_lengths = []
    cap_missing_fields = []
    cap_null_fields = []
    refs_missing = []
    targets_missing = []
    soft_missing = []
    hard_not_positive = []
    empty_caption = []
    img_set_issues = []
    member_counts = Counter()

    for idx, item in enumerate(captions):
        pairid = item.get("pairid")
        cap_pairids.append(pairid)
        missing = [key for key in required if key not in item]
        nulls = [key for key in required if key in item and item[key] is None]
        if missing:
            cap_missing_fields.append({"index": idx, "missing": missing})
        if nulls:
            cap_null_fields.append({"index": idx, "null": nulls})
        ref = item.get("reference")
        if ref and ref not in split_img_ids:
            refs_missing.append({"pairid": pairid, "reference": ref})
        if "caption" in item:
            word_count = len(words(item.get("caption")))
            cap_lengths.append(word_count)
            if word_count == 0:
                empty_caption.append(pairid)
        if split in {"train", "val"}:
            target = item.get("target_hard")
            if target and target not in split_img_ids:
                targets_missing.append({"pairid": pairid, "target_hard": target})
            target_soft = item.get("target_soft")
            if isinstance(target_soft, dict):
                for soft_id, score in target_soft.items():
                    if soft_id not in split_img_ids:
                        soft_missing.append({"pairid": pairid, "target_soft": soft_id, "score": score})
                if target and not (target in target_soft and target_soft.get(target, 0) > 0):
                    hard_not_positive.append(pairid)
            elif "target_soft" in item:
                hard_not_positive.append(pairid)
        img_set = item.get("img_set")
        if isinstance(img_set, dict) and isinstance(img_set.get("members"), list):
            members = img_set["members"]
            member_counts[len(members)] += 1
            missing_members = [member for member in members if member not in split_img_ids]
            if missing_members:
                img_set_issues.append({"pairid": pairid, "reason": "missing_members", "members": missing_members[:10]})
            if ref and ref not in members:
                img_set_issues.append({"pairid": pairid, "reason": "reference_not_in_members", "reference": ref})
            target = item.get("target_hard")
            if target and target not in members:
                img_set_issues.append({"pairid": pairid, "reason": "target_not_in_members", "target_hard": target})
        elif "img_set" in item:
            img_set_issues.append({"pairid": pairid, "reason": "bad_img_set"})

    pairid_counts = Counter(cap_pairids)
    dup_pairids = {str(k): v for k, v in pairid_counts.items() if v > 1}
    numeric_pairids = [x for x in cap_pairids if isinstance(x, int)]
    pairid_gaps = None
    if numeric_pairids:
        pairid_gaps = max(numeric_pairids) - min(numeric_pairids) + 1 - len(set(numeric_pairids))

    ext_pairids = [item.get("pairid") for item in ext_caps]
    ext_pairid_counts = Counter(ext_pairids)
    dup_ext_pairids = {str(k): v for k, v in ext_pairid_counts.items() if v > 1}
    cap_pairid_set = set(cap_pairids)
    ext_pairid_set = set(ext_pairids)
    ext_missing_vs_cap = sorted([x for x in cap_pairid_set - ext_pairid_set if x is not None])
    ext_extra_vs_cap = sorted([x for x in ext_pairid_set - cap_pairid_set if x is not None])
    ext_bad = []
    ext_empty = []
    placeholders = Counter()
    expected_ext_keys = {"0", "1", "2", "3"}
    for idx, item in enumerate(ext_caps):
        pairid = item.get("pairid")
        ext = item.get("caption_extend")
        if not isinstance(ext, dict):
            ext_bad.append({"index": idx, "pairid": pairid, "reason": "caption_extend_not_object"})
            continue
        if set(ext.keys()) != expected_ext_keys:
            ext_bad.append({"index": idx, "pairid": pairid, "reason": "caption_extend_keys", "keys": sorted(ext.keys())})
        for key, value in ext.items():
            text = str(value).strip()
            if not text:
                ext_empty.append({"pairid": pairid, "key": key})
            if "[cr0]" in text:
                placeholders["[cr0] Nothing worth mentioning"] += 1
            if "[cr1]" in text:
                placeholders["[cr1] Covered in query"] += 1
            if "[c]" in text:
                placeholders["[c] None existed"] += 1

    result = {
        "image_dir": IMAGE_DIR[split],
        "expected_images_from_split_json": len(expected),
        "actual_png_files": len(actual_paths),
        "missing_images_count": len(missing_images),
        "missing_images_sample": missing_images[:20],
        "extra_png_count": len(extra_paths),
        "extra_png_sample": [p.relative_to(DATA_ROOT).as_posix() for p in extra_paths[:20]],
        "path_id_mismatch_count": len(path_id_mismatch),
        "path_id_mismatch_sample": path_id_mismatch[:20],
        "duplicate_actual_ids_count": len(duplicate_actual_ids),
        "duplicate_actual_ids_sample": dict(list(duplicate_actual_ids.items())[:20]),
        "zero_byte_png_count": len(zero_byte),
        "zero_byte_png_sample": zero_byte[:20],
        "corrupt_png_count": len(corrupt),
        "corrupt_png_sample": corrupt[:20],
        "image_modes": dict(modes),
        "image_formats": dict(formats),
        "top_dimensions": dict(dims.most_common(15)),
        "dimension_unique_count": len(dims),
        "aspect_ratio_min": min(aspect_ratios) if aspect_ratios else None,
        "aspect_ratio_max": max(aspect_ratios) if aspect_ratios else None,
        "aspect_ratio_mean": mean(aspect_ratios),
        "duplicate_byte_hash_group_count": len(duplicate_hash_groups),
        "duplicate_byte_file_count": sum(group["count"] for group in duplicate_hash_groups),
        "duplicate_byte_hash_groups_sample": duplicate_hash_groups[:10],
        "captions_count": len(captions),
        "captions_ext_count": len(ext_caps),
        "caption_missing_fields_count": len(cap_missing_fields),
        "caption_missing_fields_sample": cap_missing_fields[:20],
        "caption_null_fields_count": len(cap_null_fields),
        "caption_null_fields_sample": cap_null_fields[:20],
        "caption_duplicate_pairids_count": len(dup_pairids),
        "caption_duplicate_pairids_sample": dict(list(dup_pairids.items())[:20]),
        "caption_pairid_min": min(numeric_pairids) if numeric_pairids else None,
        "caption_pairid_max": max(numeric_pairids) if numeric_pairids else None,
        "caption_pairid_gaps": pairid_gaps,
        "caption_reference_missing_count": len(refs_missing),
        "caption_reference_missing_sample": refs_missing[:20],
        "caption_target_missing_count": len(targets_missing),
        "caption_target_missing_sample": targets_missing[:20],
        "caption_target_soft_missing_count": len(soft_missing),
        "caption_target_soft_missing_sample": soft_missing[:20],
        "caption_target_hard_not_positive_soft_count": len(hard_not_positive),
        "caption_target_hard_not_positive_soft_sample": hard_not_positive[:20],
        "caption_empty_count": len(empty_caption),
        "caption_empty_sample": empty_caption[:20],
        "caption_word_count_min": min(cap_lengths) if cap_lengths else None,
        "caption_word_count_mean": mean(cap_lengths),
        "caption_word_count_p50": percentile(cap_lengths, 50),
        "caption_word_count_p95": percentile(cap_lengths, 95),
        "caption_word_count_max": max(cap_lengths) if cap_lengths else None,
        "img_set_member_counts": dict(member_counts),
        "img_set_issue_count": len(img_set_issues),
        "img_set_issue_sample": img_set_issues[:20],
        "captions_ext_duplicate_pairids_count": len(dup_ext_pairids),
        "captions_ext_duplicate_pairids_sample": dict(list(dup_ext_pairids.items())[:20]),
        "captions_ext_missing_pairids_vs_captions_count": len(ext_missing_vs_cap),
        "captions_ext_missing_pairids_vs_captions_sample": ext_missing_vs_cap[:20],
        "captions_ext_extra_pairids_vs_captions_count": len(ext_extra_vs_cap),
        "captions_ext_extra_pairids_vs_captions_sample": ext_extra_vs_cap[:20],
        "captions_ext_bad_structure_count": len(ext_bad),
        "captions_ext_bad_structure_sample": ext_bad[:20],
        "captions_ext_empty_values_count": len(ext_empty),
        "captions_ext_empty_values_sample": ext_empty[:20],
        "captions_ext_placeholder_counts": dict(placeholders),
    }
    result["matches_image_splits_exactly"] = (
        result["missing_images_count"] == 0
        and result["extra_png_count"] == 0
        and result["path_id_mismatch_count"] == 0
        and result["duplicate_actual_ids_count"] == 0
    )
    result["images_are_readable"] = result["zero_byte_png_count"] == 0 and result["corrupt_png_count"] == 0
    result["captions_are_structurally_clean"] = (
        result["caption_missing_fields_count"] == 0
        and result["caption_null_fields_count"] == 0
        and result["caption_duplicate_pairids_count"] == 0
        and result["caption_reference_missing_count"] == 0
        and result["caption_target_missing_count"] == 0
        and result["caption_target_soft_missing_count"] == 0
        and result["caption_target_hard_not_positive_soft_count"] == 0
        and result["caption_empty_count"] == 0
        and result["img_set_issue_count"] == 0
        and result["captions_ext_duplicate_pairids_count"] == 0
        and result["captions_ext_missing_pairids_vs_captions_count"] == 0
        and result["captions_ext_extra_pairids_vs_captions_count"] == 0
        and result["captions_ext_bad_structure_count"] == 0
        and result["captions_ext_empty_values_count"] == 0
    )
    return result, cap_lengths, set(expected), set(actual_ids)


def fmt_float(value, digits=2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def yn(value: bool) -> str:
    return "OK" if value else "NO"


def write_markdown(report: dict):
    summary_rows = []
    caption_rows = []
    placeholder_rows = []
    duplicate_rows = []
    for split in SPLITS:
        s = report["splits"][split]
        summary_rows.append(
            f"| {split} | {s['expected_images_from_split_json']:,} | {s['actual_png_files']:,} | "
            f"{s['captions_count']:,} | {s['captions_ext_count']:,} | {s['missing_images_count']} | "
            f"{s['extra_png_count']} | {s['corrupt_png_count']} | {yn(s['matches_image_splits_exactly'])} |"
        )
        caption_rows.append(
            f"| {split} | {s['caption_word_count_min']} | {fmt_float(s['caption_word_count_mean'])} | "
            f"{fmt_float(s['caption_word_count_p50'])} | {fmt_float(s['caption_word_count_p95'])} | "
            f"{s['caption_word_count_max']} | gaps: {s['caption_pairid_gaps']} |"
        )
        placeholders = s["captions_ext_placeholder_counts"]
        placeholder_rows.append(
            f"| {split} | {placeholders.get('[cr0] Nothing worth mentioning', 0):,} | "
            f"{placeholders.get('[cr1] Covered in query', 0):,} | {placeholders.get('[c] None existed', 0):,} |"
        )
        duplicate_rows.append(
            f"| {split} | {s['duplicate_byte_hash_group_count']} | {s['duplicate_byte_file_count']} | "
            f"{s['dimension_unique_count']} | {s['top_dimensions']} |"
        )

    text = f"""# CIRR Data Quality Report

Generated: `{report['generated_at']}`

Scope: kiểm tra chỉ đọc thư mục `CIRR`; mọi biểu đồ và file báo cáo được ghi vào `{OUT.name}`. Metadata gốc nằm trong repo git `{META_ROOT.relative_to(ROOT).as_posix()}`.

## Kết luận nhanh

- Metadata git: branch `{report['metadata_git']['branch']}`, commit `{report['metadata_git']['commit']}`, working tree `{('clean' if report['metadata_git']['status_clean'] else 'dirty')}`.
- Ảnh local khớp `image_splits`: `{yn(report['overall']['all_splits_match_image_splits_exactly'])}`.
- Tất cả PNG đọc được, không 0 byte/hỏng: `{yn(report['overall']['all_images_readable'])}`.
- Caption/caption_ext sạch về cấu trúc: `{yn(report['overall']['all_captions_structurally_clean'])}`.
- Không có image id trùng giữa các split: `{yn(report['overall']['cross_split_has_no_expected_id_overlap'])}`.
- Đánh giá tổng thể cho pipeline: `{yn(report['overall']['clean_for_pipeline'])}`.

## Đối chiếu với metadata gốc

| split | expected images | actual PNG | captions | captions_ext | missing | extra | corrupt | exact match |
|---|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(summary_rows)}

![Counts overview](counts_overview.png)

## Lỗi cấu trúc phát hiện

![Quality issues](quality_issues.png)

## Caption

| split | min words | mean | p50 | p95 | max words | pairid gaps |
|---|---:|---:|---:|---:|---:|---|
{chr(10).join(caption_rows)}

![Caption length distribution](caption_length_distribution.png)

Các placeholder trong `captions_ext` là annotation hợp lệ của CIRR, không phải lỗi kỹ thuật, nhưng nên biết khi huấn luyện/phân tích text:

| split | [cr0] | [cr1] | [c] |
|---|---:|---:|---:|
{chr(10).join(placeholder_rows)}

## Ảnh

| split | duplicate byte hash groups | files in duplicate groups | unique dimensions | top dimensions |
|---|---:|---:|---:|---|
{chr(10).join(duplicate_rows)}

![Image dimensions](image_dimensions.png)

![Train bucket distribution](train_bucket_distribution.png)

## Sample trực quan

![Train sample](sample_sheet_train.png)

![Val sample](sample_sheet_val.png)

![Test1 sample](sample_sheet_test1.png)

## Ghi chú

- So sánh "giống dữ liệu gốc" ở đây dùng `CIRR/cirr/image_splits/*.json` và repo git metadata `cirr_dataset` làm nguồn gốc cục bộ.
- File chi tiết máy đọc được: `cirr_quality_report.json`.
- SHA-256 của các JSON metadata đã được lưu trong report JSON để tái kiểm tra.
"""
    (OUT / "report.md").write_text(text, encoding="utf-8")


def main():
    metadata = {}
    for split in SPLITS:
        metadata[split] = {
            "captions": load_json(META_ROOT / "captions" / f"cap.rc2.{split}.json", f"captions/{split}"),
            "captions_ext": load_json(META_ROOT / "captions_ext" / f"cap.ext.rc2.{split}.json", f"captions_ext/{split}"),
            "image_splits": load_json(META_ROOT / "image_splits" / f"split.rc2.{split}.json", f"image_splits/{split}"),
        }

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(DATA_ROOT),
        "metadata_root": str(META_ROOT),
        "metadata_git": git_info(META_ROOT),
        "splits": {},
        "cross_split": {},
        "json_duplicate_keys": {},
        "outputs": {},
    }

    all_records = []
    lengths_by_split = {}
    expected_ids = {}
    actual_ids = {}
    for split in SPLITS:
        result, lengths, expected, actual = audit_split(split, metadata[split], all_records)
        report["splits"][split] = result
        lengths_by_split[split] = lengths
        expected_ids[split] = expected
        actual_ids[split] = actual

    for idx, left in enumerate(SPLITS):
        for right in SPLITS[idx + 1 :]:
            expected_overlap = sorted(expected_ids[left] & expected_ids[right])
            actual_overlap = sorted(actual_ids[left] & actual_ids[right])
            report["cross_split"][f"{left}_vs_{right}"] = {
                "expected_image_id_overlap_count": len(expected_overlap),
                "expected_image_id_overlap_sample": expected_overlap[:20],
                "actual_image_id_overlap_count": len(actual_overlap),
                "actual_image_id_overlap_sample": actual_overlap[:20],
            }

    report["json_duplicate_keys"] = {k: dict(v) for k, v in JSON_DUP_KEYS.items()}
    report["metadata_json_sha256"] = {
        path.relative_to(META_ROOT).as_posix(): sha256_file(path) for path in sorted(META_ROOT.rglob("*.json"))
    }
    report["overall"] = {
        "metadata_git_clean": report["metadata_git"]["status_clean"],
        "all_splits_match_image_splits_exactly": all(
            report["splits"][s]["matches_image_splits_exactly"] for s in SPLITS
        ),
        "all_images_readable": all(report["splits"][s]["images_are_readable"] for s in SPLITS),
        "all_captions_structurally_clean": all(
            report["splits"][s]["captions_are_structurally_clean"] for s in SPLITS
        ),
        "cross_split_has_no_expected_id_overlap": all(
            item["expected_image_id_overlap_count"] == 0 for item in report["cross_split"].values()
        ),
    }
    report["overall"]["clean_for_pipeline"] = all(report["overall"].values())

    grouped_bar(
        OUT / "counts_overview.png",
        "CIRR counts: metadata vs actual files",
        SPLITS,
        [
            ("image_splits", [report["splits"][s]["expected_images_from_split_json"] for s in SPLITS], COLORS["blue"]),
            ("actual PNG", [report["splits"][s]["actual_png_files"] for s in SPLITS], COLORS["green"]),
            ("captions", [report["splits"][s]["captions_count"] for s in SPLITS], COLORS["orange"]),
            ("captions_ext", [report["splits"][s]["captions_ext_count"] for s in SPLITS], COLORS["purple"]),
        ],
    )
    issue_names = ["missing", "extra", "corrupt", "zero-byte", "caption schema", "img_set", "ext schema"]
    issue_colors = [
        COLORS["red"],
        (214, 133, 69),
        COLORS["purple"],
        (201, 180, 88),
        COLORS["gray"],
        COLORS["green"],
        COLORS["brown"],
    ]
    issue_data = []
    for split in SPLITS:
        s = report["splits"][split]
        issue_data.append(
            [
                s["missing_images_count"],
                s["extra_png_count"],
                s["corrupt_png_count"],
                s["zero_byte_png_count"],
                s["caption_missing_fields_count"]
                + s["caption_null_fields_count"]
                + s["caption_duplicate_pairids_count"]
                + s["caption_empty_count"],
                s["img_set_issue_count"],
                s["captions_ext_bad_structure_count"] + s["captions_ext_empty_values_count"],
            ]
        )
    stacked_bar(OUT / "quality_issues.png", "Detected structural issues", SPLITS, issue_data, issue_names, issue_colors)
    histogram(OUT / "caption_length_distribution.png", "Caption length distribution", lengths_by_split)
    dimensions_chart(OUT / "image_dimensions.png", all_records)
    bucket_counts = Counter(
        Path(r["rel_path"]).parts[1]
        for r in all_records
        if r["split"] == "train" and len(Path(r["rel_path"]).parts) >= 3
    )
    bucket_chart(OUT / "train_bucket_distribution.png", bucket_counts)
    for split in SPLITS:
        sample_sheet(OUT / f"sample_sheet_{split}.png", split, all_records)

    report["outputs"] = {
        "markdown_report": str(OUT / "report.md"),
        "json_report": str(OUT / "cirr_quality_report.json"),
        "counts_overview_png": str(OUT / "counts_overview.png"),
        "quality_issues_png": str(OUT / "quality_issues.png"),
        "caption_length_distribution_png": str(OUT / "caption_length_distribution.png"),
        "image_dimensions_png": str(OUT / "image_dimensions.png"),
        "train_bucket_distribution_png": str(OUT / "train_bucket_distribution.png"),
        "sample_sheet_train_png": str(OUT / "sample_sheet_train.png"),
        "sample_sheet_val_png": str(OUT / "sample_sheet_val.png"),
        "sample_sheet_test1_png": str(OUT / "sample_sheet_test1.png"),
    }
    write_markdown(report)
    (OUT / "cirr_quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
