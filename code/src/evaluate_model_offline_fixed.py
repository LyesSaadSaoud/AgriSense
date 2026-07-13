#!/usr/bin/env python3
"""
Offline evaluator for the Greenhouse VLA model.

This script actually loads a trained checkpoint, runs inference on a CSV/image dataset,
and writes:
  - predictions.csv
  - metrics.json
  - confusion matrices / classification reports

It is different from the IEEE plotting script, which only plots an existing prediction CSV.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)
from torch.utils.data import DataLoader

from greenhouse_vla_dataset_fixed import (
    GreenhouseVLADataset,
    discover_target_columns,
    load_preprocessing_artifacts,
)
from greenhouse_vla_model_bert_dino_pesticide import GreenhouseVLA


def safe_torch_load(path: str, map_location="cpu"):
    """Compatible with both older PyTorch and PyTorch versions that default to weights_only."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def pred_col_name_from_reg_col(col: str) -> str:
    """Convert a target/GT regression column name to the plotting-friendly pred name."""
    if col.startswith("gt_"):
        return "pred_" + col[len("gt_") :]
    if col.startswith("target_zone_"):
        return "pred_" + col
    if col.startswith("robot_target_"):
        return "pred_" + col[len("robot_") :]
    if col.startswith("pred_"):
        return col
    return "pred_" + col


def gt_col_name_from_reg_col(col: str) -> str:
    """Convert a regression target column name to an explicit GT name."""
    if col.startswith("gt_"):
        return col
    if col.startswith("target_zone_"):
        return "gt_" + col
    return "gt_" + col


def build_model_from_checkpoint(checkpoint: dict, args, regression_dim: int) -> GreenhouseVLA:
    config = checkpoint.get("config", {}) if isinstance(checkpoint.get("config", {}), dict) else {}

    sensor_dim = int(checkpoint.get("sensor_dim", config.get("sensor_dim", 19)))
    image_model_name = checkpoint.get("image_model_name", config.get("image_model_name", args.image_model_name))
    text_model_name = checkpoint.get("text_model_name", config.get("text_model_name", args.text_model_name))
    image_pooling = checkpoint.get("image_pooling", config.get("image_pooling", "mean"))
    text_pooling = checkpoint.get("text_pooling", config.get("text_pooling", "mean"))

    freeze_image_backbone = checkpoint.get(
        "freeze_image_backbone", config.get("freeze_image_backbone", not args.unfreeze_image_backbone)
    )
    freeze_text_backbone = checkpoint.get(
        "freeze_text_backbone", config.get("freeze_text_backbone", not args.unfreeze_text_backbone)
    )

    model = GreenhouseVLA(
        sensor_dim=sensor_dim,
        image_model_name=image_model_name,
        text_model_name=text_model_name,
        regression_dim=int(checkpoint.get("regression_dim", regression_dim)),
        num_zone_classes=int(checkpoint.get("num_zone_classes", 4)),
        freeze_image_backbone=bool(freeze_image_backbone),
        freeze_text_backbone=bool(freeze_text_backbone),
        image_pooling=image_pooling,
        text_pooling=text_pooling,
    )
    return model


@torch.no_grad()
def run_inference(model, loader, device, target_standardizer):
    model.eval()

    rows = []
    all_pred_reg = []
    all_true_reg = []
    all_pred_zone = []
    all_true_zone = []
    all_pred_pesticide = []
    all_true_pesticide = []
    all_pesticide_prob = []

    for batch_idx, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        sensor = batch["sensor"].to(device, non_blocking=True)
        target_reg = batch["target_regression"].to(device, non_blocking=True)
        target_zone = batch["target_zone"].to(device, non_blocking=True)
        target_pesticide = batch["target_pesticide"].float().view(-1).to(device, non_blocking=True)

        outputs = model(
            image=image,
            input_ids=input_ids,
            attention_mask=attention_mask,
            sensor=sensor,
        )

        pred_reg = outputs["pred_regression"]
        pred_zone_logits = outputs["pred_zone_logits"]
        pred_pesticide_logits = outputs["pred_pesticide_logits"].view(-1)

        pred_zone = torch.argmax(pred_zone_logits, dim=1)
        pesticide_prob = torch.sigmoid(pred_pesticide_logits)
        pred_pesticide = (pesticide_prob > 0.5).long()

        pred_reg_np = pred_reg.detach().cpu().numpy() * target_standardizer.std + target_standardizer.mean
        true_reg_np = target_reg.detach().cpu().numpy() * target_standardizer.std + target_standardizer.mean

        all_pred_reg.append(pred_reg_np)
        all_true_reg.append(true_reg_np)
        all_pred_zone.append(pred_zone.detach().cpu().numpy())
        all_true_zone.append(target_zone.detach().cpu().numpy())
        all_pred_pesticide.append(pred_pesticide.detach().cpu().numpy())
        all_true_pesticide.append(target_pesticide.long().detach().cpu().numpy())
        all_pesticide_prob.append(pesticide_prob.detach().cpu().numpy())

    return {
        "pred_reg": np.concatenate(all_pred_reg, axis=0),
        "true_reg": np.concatenate(all_true_reg, axis=0),
        "pred_zone_zero_based": np.concatenate(all_pred_zone, axis=0),
        "true_zone_zero_based": np.concatenate(all_true_zone, axis=0),
        "pred_pesticide": np.concatenate(all_pred_pesticide, axis=0),
        "true_pesticide": np.concatenate(all_true_pesticide, axis=0),
        "pred_pesticide_prob": np.concatenate(all_pesticide_prob, axis=0),
    }


def compute_metrics(results: dict, reg_cols: List[str]) -> Dict:
    pred_reg = results["pred_reg"]
    true_reg = results["true_reg"]
    pred_zone = results["pred_zone_zero_based"]
    true_zone = results["true_zone_zero_based"]
    pred_pest = results["pred_pesticide"]
    true_pest = results["true_pesticide"]

    metrics: Dict = {}
    metrics["num_samples"] = int(len(true_zone))
    metrics["regression_mae_overall"] = float(mean_absolute_error(true_reg, pred_reg))

    per_reg = {}
    for i, col in enumerate(reg_cols):
        per_reg[col] = float(mean_absolute_error(true_reg[:, i], pred_reg[:, i]))
    metrics["regression_mae_by_column"] = per_reg

    metrics["zone_accuracy"] = float(accuracy_score(true_zone, pred_zone))
    metrics["zone_macro_f1"] = float(f1_score(true_zone, pred_zone, average="macro", zero_division=0))
    metrics["zone_confusion_matrix_1_based_rows_true_cols_pred"] = confusion_matrix(
        true_zone + 1, pred_zone + 1, labels=[1, 2, 3, 4]
    ).tolist()

    metrics["pesticide_accuracy"] = float(accuracy_score(true_pest, pred_pest))
    metrics["pesticide_f1"] = float(f1_score(true_pest, pred_pest, average="binary", zero_division=0))
    metrics["pesticide_confusion_matrix_rows_true_cols_pred"] = confusion_matrix(
        true_pest, pred_pest, labels=[0, 1]
    ).tolist()

    return metrics


def make_predictions_csv(df: pd.DataFrame, results: dict, reg_cols: List[str], zone_col: str, pesticide_col: str) -> pd.DataFrame:
    n = len(df)
    out = pd.DataFrame()

    if "sample_idx" in df.columns:
        out["sample_idx"] = df["sample_idx"].values
    elif "index" in df.columns:
        out["sample_idx"] = df["index"].values
    else:
        out["sample_idx"] = np.arange(n)

    # Keep useful source fields for plotting/debugging.
    for col in ["image_relpath", "image_path", "front_image_path", "user_prompt", "task_prompt", "prompt"]:
        if col in df.columns:
            out[col] = df[col].values

    # Zone labels are 1-based in the CSV/plots.
    out["gt_target_zone"] = results["true_zone_zero_based"] + 1
    out["pred_target_zone"] = results["pred_zone_zero_based"] + 1

    # Pesticide.
    out["gt_apply_pesticide"] = results["true_pesticide"].astype(int)
    out["pred_apply_pesticide"] = results["pred_pesticide"].astype(int)
    out["pred_apply_pesticide_prob"] = results["pred_pesticide_prob"].astype(float)

    # Regression values, both explicit GT and pred columns.
    true_reg = results["true_reg"]
    pred_reg = results["pred_reg"]
    for i, col in enumerate(reg_cols):
        gt_name = gt_col_name_from_reg_col(col)
        pred_name = pred_col_name_from_reg_col(col)
        out[gt_name] = true_reg[:, i].astype(float)
        out[pred_name] = pred_reg[:, i].astype(float)

    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--preprocessing_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_text_len", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)

    # Only used as fallback if the checkpoint does not store these names.
    parser.add_argument("--image_model_name", type=str, default="facebook/dinov2-small")
    parser.add_argument("--text_model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--unfreeze_image_backbone", action="store_true")
    parser.add_argument("--unfreeze_text_backbone", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if args.preprocessing_path is None:
        args.preprocessing_path = str(checkpoint_path.parent / "preprocessing.json")

    if args.output_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(checkpoint_path.parent / f"offline_eval_{stamp}")

    ensure_dir(args.output_dir)

    csv_path = Path(args.csv_path).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve() if args.image_root else csv_path.parent

    df = pd.read_csv(csv_path)
    if args.max_samples is not None and args.max_samples > 0:
        df = df.iloc[: args.max_samples].reset_index(drop=True)

    reg_cols, zone_col, pesticide_col = discover_target_columns(df)
    prep = load_preprocessing_artifacts(args.preprocessing_path)

    text_tokenizer = prep["text_tokenizer"]
    image_processor = prep["image_processor"]
    sensor_standardizer = prep["sensor_standardizer"]
    target_standardizer = prep["target_standardizer"]
    max_text_len = args.max_text_len if args.max_text_len is not None else prep["max_text_len"]

    dataset = GreenhouseVLADataset(
        df,
        text_tokenizer=text_tokenizer,
        image_processor=image_processor,
        image_root=str(image_root),
        max_text_len=max_text_len,
        sensor_standardizer=sensor_standardizer,
        target_standardizer=target_standardizer,
    )

    pin_memory = torch.cuda.is_available()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    checkpoint = safe_torch_load(str(checkpoint_path), map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model_from_checkpoint(checkpoint, args, regression_dim=len(reg_cols)).to(device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as exc:
        print(f"[WARN] Strict checkpoint load failed, retrying strict=False:\n{exc}")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    results = run_inference(model, loader, device, target_standardizer)
    metrics = compute_metrics(results, reg_cols)
    pred_df = make_predictions_csv(df, results, reg_cols, zone_col, pesticide_col)

    pred_csv_path = os.path.join(args.output_dir, "predictions.csv")
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    zone_report_path = os.path.join(args.output_dir, "zone_classification_report.txt")
    pest_report_path = os.path.join(args.output_dir, "pesticide_classification_report.txt")

    pred_df.to_csv(pred_csv_path, index=False)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(zone_report_path, "w", encoding="utf-8") as f:
        f.write(
            classification_report(
                results["true_zone_zero_based"] + 1,
                results["pred_zone_zero_based"] + 1,
                labels=[1, 2, 3, 4],
                zero_division=0,
            )
        )

    with open(pest_report_path, "w", encoding="utf-8") as f:
        f.write(
            classification_report(
                results["true_pesticide"],
                results["pred_pesticide"],
                labels=[0, 1],
                zero_division=0,
            )
        )

    print("\nOffline evaluation complete")
    print(f"CSV:        {csv_path}")
    print(f"Image root: {image_root}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Pred CSV:   {pred_csv_path}")
    print(f"Metrics:    {metrics_path}")
    print("\nKey metrics:")
    print(json.dumps({
        "num_samples": metrics["num_samples"],
        "regression_mae_overall": metrics["regression_mae_overall"],
        "zone_accuracy": metrics["zone_accuracy"],
        "zone_macro_f1": metrics["zone_macro_f1"],
        "pesticide_accuracy": metrics["pesticide_accuracy"],
        "pesticide_f1": metrics["pesticide_f1"],
    }, indent=2))


if __name__ == "__main__":
    main()
