#!/usr/bin/env python3
"""
Augment a strawberry greenhouse image dataset with disease variants using the OpenAI Image API.

What it does:
1) Read an existing CSV dataset.
2) Randomly resample rows that have an image path.
3) Edit the corresponding images to add a selected strawberry disease.
4) Save augmented images into an output folder.
5) Append new rows into an augmented CSV.

Expected dataset behavior:
- Your CSV must contain an image path column (default: image_relpath).
- The image path is resolved relative to --dataset-root unless it is already absolute.

Example:
python augment_strawberry_disease_dataset.py \
    --csv /path/to/samples.csv \
    --dataset-root /path/to/dataset \
    --output-root /path/to/output_augmented \
    --num-samples 300 \
    --model gpt-image-1.5 \
    --quality high \
    --input-fidelity high \
    --seed 42

Environment:
export OPENAI_API_KEY=...

Notes:
- This script keeps original rows and appends augmented rows.
- New metadata columns are added automatically.
- If an image edit fails, the script logs the error and continues.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from openai import OpenAI


DEFAULT_DISEASE_WEIGHTS = {
    "healthy_unripe": 0.0,   # keep at 0 for augmentation; original data already covers healthy
    "healthy_ripe": 0.0,     # keep at 0 for augmentation; original data already covers healthy
    "powdery_mildew": 0.35,
    "botrytis": 0.35,
    "anthracnose": 0.30,
}

DISEASE_PROMPTS = {
    "powdery_mildew": """
Edit this greenhouse image to show realistic strawberry fruit with early-to-moderate powdery mildew infection.
Preserve the original camera pose, greenhouse structure, leaves, lighting, shadows, depth, and composition.
Only modify visible strawberry fruits.
Add subtle white powdery fungal patches on the fruit surface, realistic and agricultural, not exaggerated.
Do not add text, labels, watermarks, new objects, or extra fruit.
The result must stay photorealistic and consistent with the original image.
""".strip(),
    "botrytis": """
Edit this greenhouse image to show realistic strawberry fruit with botrytis gray mold infection.
Preserve the original camera pose, greenhouse structure, leaves, lighting, shadows, depth, and composition.
Only modify visible strawberry fruits.
Add realistic gray-brown mold and slight fruit rot on some visible strawberries, agricultural and believable, not exaggerated.
Do not add text, labels, watermarks, new objects, or extra fruit.
The result must stay photorealistic and consistent with the original image.
""".strip(),
    "anthracnose": """
Edit this greenhouse image to show realistic strawberry fruit with anthracnose symptoms.
Preserve the original camera pose, greenhouse structure, leaves, lighting, shadows, depth, and composition.
Only modify visible strawberry fruits.
Add realistic dark sunken lesions and disease spots on some visible strawberries, agricultural and believable, not exaggerated.
Do not add text, labels, watermarks, new objects, or extra fruit.
The result must stay photorealistic and consistent with the original image.
""".strip(),
}

ALLOWED_DISEASES = tuple(DISEASE_PROMPTS.keys())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Augment strawberry dataset with disease variants using OpenAI image editing.")
    p.add_argument("--csv", required=True, help="Path to original CSV file.")
    p.add_argument("--dataset-root", required=True, help="Root folder of the dataset. image_relpath is resolved relative to this.")
    p.add_argument("--output-root", required=True, help="Output folder for augmented images and CSV.")
    p.add_argument("--image-column", default="image_relpath", help="CSV column containing image relative path.")
    p.add_argument("--num-samples", type=int, default=200, help="Number of source rows to resample.")
    p.add_argument("--aug-per-image", type=int, default=1, help="Number of augmented variants per sampled image.")
    p.add_argument("--model", default="gpt-image-1.5", help="OpenAI image model.")
    p.add_argument("--quality", default="high", choices=["low", "medium", "high", "auto"], help="Requested image quality.")
    p.add_argument("--input-fidelity", default="high", choices=["low", "high"], help="Preserve source image fidelity during edits.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducible row/disease sampling.")
    p.add_argument("--save-every", type=int, default=25, help="Save progress every N successful augmentations.")
    p.add_argument("--max-retries", type=int, default=4, help="Retries per failed API call.")
    p.add_argument("--sleep-between", type=float, default=0.5, help="Base pause between requests in seconds.")
    p.add_argument(
        "--disease-weights-json",
        default="",
        help='Optional JSON string like \'{"powdery_mildew":0.4,"botrytis":0.4,"anthracnose":0.2}\'',
    )
    p.add_argument(
        "--only-diseases",
        default="",
        help="Comma-separated subset, e.g. powdery_mildew,botrytis",
    )
    p.add_argument(
        "--aug-subdir",
        default="augmented/images",
        help="Subdirectory under output-root where augmented images are saved.",
    )
    p.add_argument(
        "--copy-original-columns-only",
        action="store_true",
        help="If set, only append new metadata columns and do not try to update any existing label columns.",
    )
    return p.parse_args()


def resolve_disease_weights(args: argparse.Namespace) -> Dict[str, float]:
    weights = dict(DEFAULT_DISEASE_WEIGHTS)
    if args.only_diseases:
        keep = {x.strip() for x in args.only_diseases.split(",") if x.strip()}
        weights = {k: v for k, v in weights.items() if k in keep}
    else:
        weights = {k: v for k, v in weights.items() if v > 0.0}

    if args.disease_weights_json:
        user_weights = json.loads(args.disease_weights_json)
        weights = {k: float(v) for k, v in user_weights.items() if k in ALLOWED_DISEASES}

    if not weights:
        raise ValueError("No disease classes selected.")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Disease weights must sum to a positive value.")
    return {k: v / total for k, v in weights.items()}


def weighted_choice(rng: random.Random, weights: Dict[str, float]) -> str:
    keys = list(weights.keys())
    vals = list(weights.values())
    return rng.choices(keys, weights=vals, k=1)[0]


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_image_path(dataset_root: Path, rel_or_abs: str) -> Path:
    p = Path(str(rel_or_abs))
    return p if p.is_absolute() else dataset_root / p


def encode_image_b64(image_path: Path) -> Tuple[str, str]:
    mime, _ = mimetypes.guess_type(str(image_path))
    if mime is None:
        suffix = image_path.suffix.lower()
        if suffix in [".jpg", ".jpeg"]:
            mime = "image/jpeg"
        elif suffix == ".png":
            mime = "image/png"
        elif suffix == ".webp":
            mime = "image/webp"
        else:
            raise ValueError(f"Unsupported or unknown image type: {image_path}")
    data = image_path.read_bytes()
    return base64.b64encode(data).decode("utf-8"), mime


def decode_image_result_to_bytes(result) -> bytes:
    # OpenAI Image API commonly returns b64_json per image item.
    if not hasattr(result, "data") or not result.data:
        raise RuntimeError("No image data returned from OpenAI.")
    item = result.data[0]
    b64 = getattr(item, "b64_json", None)
    if not b64:
        raise RuntimeError("Returned image object does not include b64_json.")
    return base64.b64decode(b64)


def edit_with_openai(
    client: OpenAI,
    image_path: Path,
    disease_name: str,
    model: str,
    quality: str,
    input_fidelity: str,
    max_retries: int,
    sleep_between: float,
) -> bytes:
    prompt = DISEASE_PROMPTS[disease_name]
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            with open(image_path, "rb") as f:
                result = client.images.edit(
                    model=model,
                    image=[f],
                    prompt=prompt,
                    quality=quality,
                    input_fidelity=input_fidelity,
                )
            return decode_image_result_to_bytes(result)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt == max_retries:
                break
            backoff = sleep_between * (2 ** (attempt - 1))
            time.sleep(backoff)

    raise RuntimeError(f"Image edit failed for {image_path.name} / {disease_name}: {last_err}") from last_err


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    needed = [
        "is_augmented",
        "augmentation_parent_sample_idx",
        "augmentation_type",
        "disease_class",
        "augmentation_prompt",
        "augmentation_model",
        "augmentation_quality",
        "augmentation_input_fidelity",
        "augmentation_created_utc",
        "source_image_relpath",
    ]
    for col in needed:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def maybe_update_existing_label_columns(new_row: dict, disease_name: str, original_row: pd.Series) -> dict:
    # Conservative defaults so you can still train with the appended CSV immediately.
    if "label_status" in new_row:
        new_row["label_status"] = "synthetic_augmented"
    if "label_source" in new_row:
        new_row["label_source"] = "openai_image_edit"
    if "reason_tags_json" in new_row:
        tags = [f"disease:{disease_name}", "synthetic_augmentation"]
        new_row["reason_tags_json"] = json.dumps(tags, ensure_ascii=False)
    return new_row


def save_progress(df_all: pd.DataFrame, df_new_only: pd.DataFrame, output_root: Path) -> Tuple[Path, Path]:
    all_csv = output_root / "samples_augmented.csv"
    new_csv = output_root / "samples_augmented_only.csv"
    df_all.to_csv(all_csv, index=False)
    df_new_only.to_csv(new_csv, index=False)
    return all_csv, new_csv


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    csv_path = Path(args.csv).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    aug_dir = output_root / args.aug_subdir

    safe_mkdir(output_root)
    safe_mkdir(aug_dir)

    df = pd.read_csv(csv_path)
    if args.image_column not in df.columns:
        raise ValueError(f"Image column '{args.image_column}' not found in CSV columns: {list(df.columns)}")

    df = prepare_dataframe(df)

    # Keep only rows with non-empty image paths
    work_df = df[df[args.image_column].notna()].copy()
    work_df = work_df[work_df[args.image_column].astype(str).str.strip() != ""].copy()

    if work_df.empty:
        raise ValueError("No rows with valid image paths found.")

    disease_weights = resolve_disease_weights(args)

    # Sample with replacement so you can freely oversample even if num_samples > dataset size
    sampled_indices = [rng.choice(list(work_df.index)) for _ in range(args.num_samples)]
    
    client = OpenAI()

    appended_rows: List[dict] = []
    success_count = 0
    fail_count = 0

    for sample_num, src_idx in enumerate(sampled_indices, start=1):
        row = work_df.loc[src_idx]
        source_rel = str(row[args.image_column])
        source_img = resolve_image_path(dataset_root, source_rel)

        if not source_img.exists():
            print(f"[WARN] Missing image, skip: {source_img}", file=sys.stderr)
            fail_count += 1
            continue

        for aug_k in range(args.aug_per_image):
            disease_name = weighted_choice(rng, disease_weights)
            try:
                out_bytes = edit_with_openai(
                    client=client,
                    image_path=source_img,
                    disease_name=disease_name,
                    model=args.model,
                    quality=args.quality,
                    input_fidelity=args.input_fidelity,
                    max_retries=args.max_retries,
                    sleep_between=args.sleep_between,
                )

                src_stem = source_img.stem
                out_name = f"{src_stem}__disease_{disease_name}__sample_{sample_num:05d}__v{aug_k+1}.png"
                out_path = aug_dir / out_name
                out_path.write_bytes(out_bytes)

                rel_saved = str(out_path.relative_to(output_root)).replace("\\", "/")

                new_row = row.to_dict()
                new_row[args.image_column] = rel_saved
                new_row["source_image_relpath"] = source_rel
                new_row["is_augmented"] = True
                new_row["augmentation_parent_sample_idx"] = row.get("sample_idx", src_idx)
                new_row["augmentation_type"] = "disease_image_edit"
                new_row["disease_class"] = disease_name
                new_row["augmentation_prompt"] = DISEASE_PROMPTS[disease_name]
                new_row["augmentation_model"] = args.model
                new_row["augmentation_quality"] = args.quality
                new_row["augmentation_input_fidelity"] = args.input_fidelity
                new_row["augmentation_created_utc"] = pd.Timestamp.utcnow().isoformat()

                if not args.copy_original_columns_only:
                    new_row = maybe_update_existing_label_columns(new_row, disease_name, row)

                appended_rows.append(new_row)
                success_count += 1
                print(f"[OK] {success_count} -> {rel_saved}")

                if success_count % max(1, args.save_every) == 0:
                    df_new = pd.DataFrame(appended_rows)
                    df_all = pd.concat([df, df_new], ignore_index=True)
                    all_csv, new_csv = save_progress(df_all, df_new, output_root)
                    print(f"[SAVE] {all_csv}")
                    print(f"[SAVE] {new_csv}")

                time.sleep(args.sleep_between)

            except Exception as exc:  # noqa: BLE001
                fail_count += 1
                print(f"[ERR] {source_img.name} / {disease_name}: {exc}", file=sys.stderr)
                continue

    df_new = pd.DataFrame(appended_rows)
    df_all = pd.concat([df, df_new], ignore_index=True)
    all_csv, new_csv = save_progress(df_all, df_new, output_root)

    summary = {
        "input_csv": str(csv_path),
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "image_column": args.image_column,
        "num_requested_source_samples": args.num_samples,
        "aug_per_image": args.aug_per_image,
        "num_created_rows": int(len(df_new)),
        "success_count": int(success_count),
        "fail_count": int(fail_count),
        "model": args.model,
        "quality": args.quality,
        "input_fidelity": args.input_fidelity,
        "disease_weights": disease_weights,
        "all_csv": str(all_csv),
        "new_rows_csv": str(new_csv),
    }
    (output_root / "augmentation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
