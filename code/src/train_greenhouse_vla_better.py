#!/usr/bin/env python3
"""
Improved training script for GreenhouseVLA.

Main changes versus the previous script:
- Best checkpoint is selected by a composite validation score instead of raw loss only.
- SmoothL1/Huber regression loss reduces mean/smoothing behaviour compared with MSE.
- Optional focal loss for pesticide classification.
- Label smoothing for target-zone CE.
- Optional weighted sampler for imbalanced combined zone+pesticide classes.
- Differential learning rates for pretrained backbones and task heads.
- ReduceLROnPlateau scheduler, gradient clipping, better logging, and prediction CSV export.

Keep this file in the same folder as:
  greenhouse_vla_dataset_fixed.py
  greenhouse_vla_model_bert_dino_pesticide.py
"""

import argparse
import json
import math
import os
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, confusion_matrix
from torch.utils.data import DataLoader, WeightedRandomSampler

from greenhouse_vla_dataset_fixed import (
    GreenhouseVLADataset,
    build_hf_image_processor,
    build_hf_text_tokenizer,
    discover_target_columns,
    fit_standardizers,
    save_preprocessing_artifacts,
)
from greenhouse_vla_model_bert_dino_pesticide import GreenhouseVLA


# =============================================================================
# Reproducibility
# =============================================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# =============================================================================
# Splitting and class balancing
# =============================================================================
def custom_classwise_split(
    labels: List[int],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Stratified split by any integer label, robust to small class counts."""
    rng = random.Random(seed)
    by_class: Dict[int, List[int]] = {}
    for idx, y in enumerate(labels):
        by_class.setdefault(int(y), []).append(idx)

    train_idx, val_idx, test_idx = [], [], []

    for _, idxs in by_class.items():
        rng.shuffle(idxs)
        n = len(idxs)

        if n == 1:
            train_idx.extend(idxs)
            continue

        n_train = max(1, int(round(n * train_ratio)))
        n_val = int(round(n * val_ratio))

        if n >= 3 and n_val == 0:
            n_val = 1
        if n_train + n_val > n:
            n_val = max(0, n - n_train)

        n_test = n - n_train - n_val
        if n >= 3 and n_test == 0 and n_train > 1:
            n_train -= 1
            n_test = 1

        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train:n_train + n_val])
        test_idx.extend(idxs[n_train + n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def compute_zone_class_weights(labels_zero_based: List[int], num_classes: int = 4) -> torch.Tensor:
    counts = Counter(int(x) for x in labels_zero_based)
    total = max(1, sum(counts.values()))
    weights = []
    for c in range(num_classes):
        count = max(1, counts.get(c, 0))
        weights.append(total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float32)


def compute_pesticide_pos_weight(labels: List[int]) -> torch.Tensor:
    labels = [int(x) for x in labels]
    pos = sum(labels)
    neg = len(labels) - pos
    if pos <= 0:
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(max(1.0, neg / max(1, pos)), dtype=torch.float32)


def make_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    counts = Counter(int(x) for x in labels)
    weights = torch.tensor([1.0 / max(1, counts[int(y)]) for y in labels], dtype=torch.double)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


# =============================================================================
# Losses
# =============================================================================
class BinaryFocalWithLogitsLoss(nn.Module):
    """Binary focal loss using BCEWithLogits as the base loss."""
    def __init__(self, gamma: float = 1.5, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("pos_weight", pos_weight.clone().detach() if pos_weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.float().view(-1)
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        if self.gamma <= 0.0:
            return bce.mean()
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = (1.0 - p_t).clamp(min=1e-6).pow(self.gamma)
        return (focal_weight * bce).mean()


def build_regression_loss(name: str, huber_beta: float) -> nn.Module:
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    if name in ["l1", "mae"]:
        return nn.L1Loss()
    if name in ["smooth_l1", "huber"]:
        return nn.SmoothL1Loss(beta=float(huber_beta))
    raise ValueError(f"Unknown regression loss: {name}")


# =============================================================================
# Optimization helpers
# =============================================================================
def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable), "frozen": int(total - trainable)}


def build_optimizer(model: nn.Module, args) -> torch.optim.Optimizer:
    """Use low LR for large pretrained backbones and higher LR for fusion/task heads."""
    backbone_keywords = [
        "image_model", "image_backbone", "vision", "dinov2", "dino",
        "text_model", "text_backbone", "bert", "roberta", "transformer",
    ]

    backbone_params = []
    head_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        name_l = name.lower()
        if any(k in name_l for k in backbone_keywords):
            backbone_params.append(p)
        else:
            head_params.append(p)

    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": args.head_lr, "name": "heads"})
    if backbone_params:
        groups.append({"params": backbone_params, "lr": args.backbone_lr, "name": "backbones"})

    if not groups:
        raise RuntimeError("No trainable parameters found. Check freeze/unfreeze settings.")

    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


# =============================================================================
# Evaluation
# =============================================================================
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_std,
    reg_cols: List[str],
    zone_loss_fn: nn.Module,
    reg_loss_fn: nn.Module,
    pesticide_loss_fn: nn.Module,
    regression_loss_weight: float,
    zone_loss_weight: float,
    pesticide_loss_weight: float,
    pesticide_threshold: float,
    return_predictions: bool = False,
):
    model.eval()

    all_pred_reg, all_true_reg = [], []
    all_pred_zone, all_true_zone = [], []
    all_pesticide_prob, all_pred_pesticide, all_true_pesticide = [], [], []

    total_losses, reg_losses, zone_losses, pesticide_losses = [], [], [], []

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        sensor = batch["sensor"].to(device, non_blocking=True)
        target_reg = batch["target_regression"].to(device, non_blocking=True)
        target_zone = batch["target_zone"].to(device, non_blocking=True)
        target_pesticide = batch["target_pesticide"].to(device, non_blocking=True).float().view(-1)

        outputs = model(image=image, input_ids=input_ids, attention_mask=attention_mask, sensor=sensor)

        pred_reg = outputs["pred_regression"]
        pred_zone_logits = outputs["pred_zone_logits"]
        pred_pesticide_logits = outputs["pred_pesticide_logits"].view(-1)

        reg_loss = reg_loss_fn(pred_reg, target_reg)
        zone_loss = zone_loss_fn(pred_zone_logits, target_zone)
        pesticide_loss = pesticide_loss_fn(pred_pesticide_logits, target_pesticide)
        loss = (
            regression_loss_weight * reg_loss
            + zone_loss_weight * zone_loss
            + pesticide_loss_weight * pesticide_loss
        )

        total_losses.append(float(loss.item()))
        reg_losses.append(float(reg_loss.item()))
        zone_losses.append(float(zone_loss.item()))
        pesticide_losses.append(float(pesticide_loss.item()))

        pred_reg_np = pred_reg.detach().cpu().numpy() * target_std.std + target_std.mean
        true_reg_np = target_reg.detach().cpu().numpy() * target_std.std + target_std.mean

        pesticide_prob = torch.sigmoid(pred_pesticide_logits)
        pesticide_pred = (pesticide_prob > pesticide_threshold).long()

        all_pred_reg.append(pred_reg_np)
        all_true_reg.append(true_reg_np)
        all_pred_zone.append(pred_zone_logits.argmax(dim=1).detach().cpu().numpy())
        all_true_zone.append(target_zone.detach().cpu().numpy())
        all_pesticide_prob.append(pesticide_prob.detach().cpu().numpy())
        all_pred_pesticide.append(pesticide_pred.detach().cpu().numpy())
        all_true_pesticide.append(target_pesticide.long().detach().cpu().numpy())

    all_pred_reg = np.concatenate(all_pred_reg, axis=0)
    all_true_reg = np.concatenate(all_true_reg, axis=0)
    all_pred_zone = np.concatenate(all_pred_zone, axis=0)
    all_true_zone = np.concatenate(all_true_zone, axis=0)
    all_pesticide_prob = np.concatenate(all_pesticide_prob, axis=0)
    all_pred_pesticide = np.concatenate(all_pred_pesticide, axis=0)
    all_true_pesticide = np.concatenate(all_true_pesticide, axis=0)

    reg_mae_all = np.abs(all_true_reg - all_pred_reg).mean(axis=0)
    reg_mae = float(np.mean(reg_mae_all))

    zone_labels_present = sorted(np.unique(np.concatenate([all_true_zone, all_pred_zone])).tolist())
    pesticide_labels_present = sorted(np.unique(np.concatenate([all_true_pesticide, all_pred_pesticide])).tolist())

    metrics = {
        "loss": float(np.mean(total_losses)),
        "reg_loss": float(np.mean(reg_losses)),
        "zone_loss": float(np.mean(zone_losses)),
        "pesticide_loss": float(np.mean(pesticide_losses)),
        "reg_mae": reg_mae,
        "zone_acc": float(accuracy_score(all_true_zone, all_pred_zone)),
        "zone_macro_f1": float(f1_score(all_true_zone, all_pred_zone, average="macro", labels=zone_labels_present, zero_division=0)),
        "pesticide_acc": float(accuracy_score(all_true_pesticide, all_pred_pesticide)),
        "pesticide_f1": float(f1_score(
            all_true_pesticide,
            all_pred_pesticide,
            average="binary" if len(pesticide_labels_present) == 2 else "macro",
            labels=pesticide_labels_present,
            zero_division=0,
        )),
    }

    for i, col in enumerate(reg_cols):
        metrics[f"mae_{col}"] = float(reg_mae_all[i])

    pred_df = None
    if return_predictions:
        rows = {}
        for i, col in enumerate(reg_cols):
            rows[f"gt_{col}"] = all_true_reg[:, i]
            rows[f"pred_{col}"] = all_pred_reg[:, i]
            rows[f"abs_err_{col}"] = np.abs(all_true_reg[:, i] - all_pred_reg[:, i])
        rows["gt_zone"] = all_true_zone + 1
        rows["pred_zone"] = all_pred_zone + 1
        rows["gt_pesticide"] = all_true_pesticide.astype(int)
        rows["pred_pesticide"] = all_pred_pesticide.astype(int)
        rows["pred_pesticide_prob"] = all_pesticide_prob
        pred_df = pd.DataFrame(rows)

    return metrics, pred_df


def validation_score(metrics: Dict[str, float], args) -> float:
    """Lower is better. Selects a checkpoint that balances regression and classifiers."""
    reg_term = metrics["reg_mae"] / max(1e-9, args.reg_mae_norm)
    zone_term = 1.0 - metrics["zone_macro_f1"]
    pesticide_term = 1.0 - metrics["pesticide_f1"]
    return float(
        args.score_reg_weight * reg_term
        + args.score_zone_weight * zone_term
        + args.score_pesticide_weight * pesticide_term
    )


# =============================================================================
# Main training
# =============================================================================
def main(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    df = pd.read_csv(args.csv_path)
    reg_cols, zone_col, pesticide_col = discover_target_columns(df)

    # Remove only rows missing supervised targets. Image-path validation is left to the Dataset.
    before = len(df)
    df = df.dropna(subset=reg_cols + [zone_col, pesticide_col]).reset_index(drop=True)
    after = len(df)
    if after < before:
        print(f"Dropped {before - after} rows with missing targets.")

    zone_labels_zero = (df[zone_col].astype(int) - 1).tolist()
    pesticide_labels = df[pesticide_col].astype(float).round().astype(int).tolist()
    combined_labels = [int(z * 2 + p) for z, p in zip(zone_labels_zero, pesticide_labels)]

    train_idx, val_idx, test_idx = custom_classwise_split(
        combined_labels,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    print("Split sizes:", {"train": len(train_df), "val": len(val_df), "test": len(test_df)})
    print("Train zone counts:", Counter((train_df[zone_col].astype(int) - 1).tolist()))
    print("Train pesticide counts:", Counter(train_df[pesticide_col].astype(float).round().astype(int).tolist()))

    text_tokenizer = build_hf_text_tokenizer(args.text_model_name)
    image_processor = build_hf_image_processor(args.image_model_name)
    sensor_std, target_std = fit_standardizers(train_df)

    sensor_dim = int(np.asarray(sensor_std.mean).shape[0]) if hasattr(sensor_std, "mean") else args.sensor_dim
    regression_dim = len(reg_cols)

    save_preprocessing_artifacts(
        os.path.join(args.output_dir, "preprocessing.json"),
        text_model_name=args.text_model_name,
        image_model_name=args.image_model_name,
        sensor_std=sensor_std,
        target_std=target_std,
        max_text_len=args.max_text_len,
    )

    # Save split CSVs for reproducibility/debugging.
    train_df.to_csv(os.path.join(args.output_dir, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(args.output_dir, "val_split.csv"), index=False)
    test_df.to_csv(os.path.join(args.output_dir, "test_split.csv"), index=False)

    train_ds = GreenhouseVLADataset(
        train_df,
        text_tokenizer=text_tokenizer,
        image_processor=image_processor,
        image_root=args.image_root,
        max_text_len=args.max_text_len,
        sensor_standardizer=sensor_std,
        target_standardizer=target_std,
    )
    val_ds = GreenhouseVLADataset(
        val_df,
        text_tokenizer=text_tokenizer,
        image_processor=image_processor,
        image_root=args.image_root,
        max_text_len=args.max_text_len,
        sensor_standardizer=sensor_std,
        target_standardizer=target_std,
    )
    test_ds = GreenhouseVLADataset(
        test_df,
        text_tokenizer=text_tokenizer,
        image_processor=image_processor,
        image_root=args.image_root,
        max_text_len=args.max_text_len,
        sensor_standardizer=sensor_std,
        target_standardizer=target_std,
    )

    pin_memory = torch.cuda.is_available()
    persistent_workers = args.num_workers > 0

    sampler = None
    shuffle_train = True
    if args.use_weighted_sampler:
        train_combined = [int((int(z) - 1) * 2 + int(round(float(p)))) for z, p in zip(train_df[zone_col], train_df[pesticide_col])]
        sampler = make_weighted_sampler(train_combined)
        shuffle_train = False

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle_train,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GreenhouseVLA(
        sensor_dim=sensor_dim,
        image_model_name=args.image_model_name,
        text_model_name=args.text_model_name,
        regression_dim=regression_dim,
        num_zone_classes=4,
        freeze_image_backbone=not args.unfreeze_image_backbone,
        freeze_text_backbone=not args.unfreeze_text_backbone,
        image_pooling=args.image_pooling,
        text_pooling=args.text_pooling,
    ).to(device)

    print("Parameter counts:", count_parameters(model))

    zone_class_weights = compute_zone_class_weights(
        (train_df[zone_col].astype(int) - 1).tolist(),
        num_classes=4,
    ).to(device)
    pesticide_pos_weight = compute_pesticide_pos_weight(
        train_df[pesticide_col].astype(float).round().astype(int).tolist()
    ).to(device)

    zone_loss_fn = nn.CrossEntropyLoss(
        weight=zone_class_weights,
        label_smoothing=args.zone_label_smoothing,
    )
    reg_loss_fn = build_regression_loss(args.regression_loss, args.huber_beta)
    if args.pesticide_focal_gamma > 0.0:
        pesticide_loss_fn = BinaryFocalWithLogitsLoss(
            gamma=args.pesticide_focal_gamma,
            pos_weight=pesticide_pos_weight,
        ).to(device)
    else:
        pesticide_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pesticide_pos_weight)

    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )

    use_amp = device.type == "cuda" and (not args.no_amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_score = math.inf
    best_metrics = None
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_total_losses, epoch_reg_losses, epoch_zone_losses, epoch_pesticide_losses = [], [], [], []

        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=pin_memory)
            input_ids = batch["input_ids"].to(device, non_blocking=pin_memory)
            attention_mask = batch["attention_mask"].to(device, non_blocking=pin_memory)
            sensor = batch["sensor"].to(device, non_blocking=pin_memory)
            target_reg = batch["target_regression"].to(device, non_blocking=pin_memory)
            target_zone = batch["target_zone"].to(device, non_blocking=pin_memory)
            target_pesticide = batch["target_pesticide"].to(device, non_blocking=pin_memory).float().view(-1)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(image=image, input_ids=input_ids, attention_mask=attention_mask, sensor=sensor)
                reg_loss = reg_loss_fn(outputs["pred_regression"], target_reg)
                zone_loss = zone_loss_fn(outputs["pred_zone_logits"], target_zone)
                pesticide_loss = pesticide_loss_fn(outputs["pred_pesticide_logits"].view(-1), target_pesticide)
                loss = (
                    args.regression_loss_weight * reg_loss
                    + args.zone_loss_weight * zone_loss
                    + args.pesticide_loss_weight * pesticide_loss
                )

            scaler.scale(loss).backward()
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            epoch_total_losses.append(float(loss.item()))
            epoch_reg_losses.append(float(reg_loss.item()))
            epoch_zone_losses.append(float(zone_loss.item()))
            epoch_pesticide_losses.append(float(pesticide_loss.item()))

        val_metrics, _ = evaluate(
            model,
            val_loader,
            device,
            target_std,
            reg_cols,
            zone_loss_fn,
            reg_loss_fn,
            pesticide_loss_fn,
            regression_loss_weight=args.regression_loss_weight,
            zone_loss_weight=args.zone_loss_weight,
            pesticide_loss_weight=args.pesticide_loss_weight,
            pesticide_threshold=args.pesticide_threshold,
            return_predictions=False,
        )

        val_score = validation_score(val_metrics, args)
        scheduler.step(val_score)

        lrs = [group["lr"] for group in optimizer.param_groups]
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_total_losses)),
            "train_reg_loss": float(np.mean(epoch_reg_losses)),
            "train_zone_loss": float(np.mean(epoch_zone_losses)),
            "train_pesticide_loss": float(np.mean(epoch_pesticide_losses)),
            "val_score": float(val_score),
            "lr": lrs,
            **val_metrics,
        }
        history.append(record)
        print(json.dumps(record))

        if val_score < best_score:
            best_score = val_score
            best_metrics = val_metrics
            patience_counter = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": vars(args),
                "image_model_name": args.image_model_name,
                "text_model_name": args.text_model_name,
                "freeze_image_backbone": not args.unfreeze_image_backbone,
                "freeze_text_backbone": not args.unfreeze_text_backbone,
                "image_pooling": args.image_pooling,
                "text_pooling": args.text_pooling,
                "sensor_dim": sensor_dim,
                "regression_dim": regression_dim,
                "num_zone_classes": 4,
                "has_pesticide_head": True,
                "target_columns": {
                    "regression": reg_cols,
                    "zone_id": zone_col,
                    "pesticide": pesticide_col,
                },
                "best_val_score": best_score,
                "best_val_metrics": best_metrics,
            }
            torch.save(checkpoint, os.path.join(args.output_dir, "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Final evaluation using best checkpoint.
    checkpoint = torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics, test_pred_df = evaluate(
        model,
        test_loader,
        device,
        target_std,
        reg_cols,
        zone_loss_fn,
        reg_loss_fn,
        pesticide_loss_fn,
        regression_loss_weight=args.regression_loss_weight,
        zone_loss_weight=args.zone_loss_weight,
        pesticide_loss_weight=args.pesticide_loss_weight,
        pesticide_threshold=args.pesticide_threshold,
        return_predictions=True,
    )

    with open(os.path.join(args.output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    if test_pred_df is not None:
        test_pred_df.to_csv(os.path.join(args.output_dir, "test_predictions.csv"), index=False)
        try:
            zone_cm = confusion_matrix(test_pred_df["gt_zone"], test_pred_df["pred_zone"], labels=[1, 2, 3, 4])
            np.savetxt(os.path.join(args.output_dir, "test_zone_confusion_matrix.csv"), zone_cm, delimiter=",", fmt="%d")
        except Exception:
            pass

    split_info = {
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": test_idx,
        "best_val_score": best_score,
        "best_val": best_metrics,
        "test": test_metrics,
        "target_columns": {
            "regression": reg_cols,
            "zone_id": zone_col,
            "pesticide": pesticide_col,
        },
    }
    with open(os.path.join(args.output_dir, "split_info.json"), "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    print("Final test metrics:")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_text_len", type=int, default=128)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)

    # Optimisation. Use low backbone LR if you unfreeze pretrained DINO/BERT.
    parser.add_argument("--head_lr", type=float, default=3e-4)
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true")

    # Losses.
    parser.add_argument("--regression_loss", type=str, default="smooth_l1", choices=["smooth_l1", "huber", "l1", "mae", "mse"])
    parser.add_argument("--huber_beta", type=float, default=0.50)
    parser.add_argument("--regression_loss_weight", type=float, default=1.5)
    parser.add_argument("--zone_loss_weight", type=float, default=1.0)
    parser.add_argument("--pesticide_loss_weight", type=float, default=1.0)
    parser.add_argument("--zone_label_smoothing", type=float, default=0.02)
    parser.add_argument("--pesticide_focal_gamma", type=float, default=1.5)
    parser.add_argument("--pesticide_threshold", type=float, default=0.5)
    parser.add_argument("--use_weighted_sampler", action="store_true")

    # Checkpoint selection score.
    parser.add_argument("--reg_mae_norm", type=float, default=2.0)
    parser.add_argument("--score_reg_weight", type=float, default=1.0)
    parser.add_argument("--score_zone_weight", type=float, default=1.0)
    parser.add_argument("--score_pesticide_weight", type=float, default=0.7)

    # Models.
    parser.add_argument("--image_model_name", type=str, default="facebook/dinov2-small")
    parser.add_argument("--text_model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--image_pooling", type=str, default="mean", choices=["cls", "mean"])
    parser.add_argument("--text_pooling", type=str, default="mean", choices=["cls", "mean"])
    parser.add_argument("--unfreeze_image_backbone", action="store_true")
    parser.add_argument("--unfreeze_text_backbone", action="store_true")
    parser.add_argument("--sensor_dim", type=int, default=19)

    args = parser.parse_args()
    main(args)
