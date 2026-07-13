import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


TARGET_COLUMN_CANDIDATES = {
    "regression": [
        "assistant_target_zone_1_temp_c",
        "assistant_target_zone_1_humidity_pct",
        "assistant_target_zone_1_soil_moisture_pct",
        "assistant_target_zone_2_temp_c",
        "assistant_target_zone_2_humidity_pct",
        "assistant_target_zone_2_soil_moisture_pct",
        "assistant_target_zone_3_temp_c",
        "assistant_target_zone_3_humidity_pct",
        "assistant_target_zone_3_soil_moisture_pct",
        "assistant_target_zone_4_temp_c",
        "assistant_target_zone_4_humidity_pct",
        "assistant_target_zone_4_soil_moisture_pct",
    ],
    "regression_fallback": [
        "target_zone_1_temp_c",
        "target_zone_1_humidity_pct",
        "target_zone_1_soil_moisture_pct",
        "target_zone_2_temp_c",
        "target_zone_2_humidity_pct",
        "target_zone_2_soil_moisture_pct",
        "target_zone_3_temp_c",
        "target_zone_3_humidity_pct",
        "target_zone_3_soil_moisture_pct",
        "target_zone_4_temp_c",
        "target_zone_4_humidity_pct",
        "target_zone_4_soil_moisture_pct",
    ],
    "zone_id": "assistant_robot_target_zone_id",
    "zone_id_fallback": "robot_target_zone_id",
    "pesticide": "assistant_apply_pesticide",
    "pesticide_fallback": "apply_pesticide",
}

PROMPT_COLUMN_CANDIDATES = [
    "user_prompt",
    "task_prompt",
    "gpt_refined_prompt",
    "gpt_short_instruction",
]

SENSOR_COLUMNS = [
    "global_temp",
    "global_humidity",
    "global_soil",
    "zone_1_temp",
    "zone_1_humidity",
    "zone_1_soil",
    "zone_2_temp",
    "zone_2_humidity",
    "zone_2_soil",
    "zone_3_temp",
    "zone_3_humidity",
    "zone_3_soil",
    "zone_4_temp",
    "zone_4_humidity",
    "zone_4_soil",
]


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def to_dict(self) -> Dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: Dict) -> "Standardizer":
        return cls(
            mean=np.asarray(d["mean"], dtype=np.float32),
            std=np.asarray(d["std"], dtype=np.float32),
        )


def discover_prompt_column(df: pd.DataFrame) -> str:
    for col in PROMPT_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not find a prompt column. Expected one of: {PROMPT_COLUMN_CANDIDATES}"
    )


def discover_target_columns(df: pd.DataFrame) -> Tuple[List[str], str, str]:
    if (
        all(col in df.columns for col in TARGET_COLUMN_CANDIDATES["regression"])
        and TARGET_COLUMN_CANDIDATES["zone_id"] in df.columns
    ):
        pesticide_col = (
            TARGET_COLUMN_CANDIDATES["pesticide"]
            if TARGET_COLUMN_CANDIDATES["pesticide"] in df.columns
            else TARGET_COLUMN_CANDIDATES["pesticide_fallback"]
        )
        if pesticide_col not in df.columns:
            raise ValueError("Could not find pesticide target column in the CSV.")
        return (
            TARGET_COLUMN_CANDIDATES["regression"],
            TARGET_COLUMN_CANDIDATES["zone_id"],
            pesticide_col,
        )

    if (
        all(col in df.columns for col in TARGET_COLUMN_CANDIDATES["regression_fallback"])
        and TARGET_COLUMN_CANDIDATES["zone_id_fallback"] in df.columns
    ):
        pesticide_col = (
            TARGET_COLUMN_CANDIDATES["pesticide_fallback"]
            if TARGET_COLUMN_CANDIDATES["pesticide_fallback"] in df.columns
            else TARGET_COLUMN_CANDIDATES["pesticide"]
        )
        if pesticide_col not in df.columns:
            raise ValueError("Could not find pesticide target column in the CSV.")
        return (
            TARGET_COLUMN_CANDIDATES["regression_fallback"],
            TARGET_COLUMN_CANDIDATES["zone_id_fallback"],
            pesticide_col,
        )

    raise ValueError("Could not find the target regression/zone columns in the CSV.")


def validate_sensor_columns(df: pd.DataFrame) -> None:
    missing = [c for c in SENSOR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required sensor columns: {missing}")


def safe_bool(value) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        return int(float(value) > 0.5)
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "apply", "spray"}:
        return 1
    if s in {"0", "false", "no", "n", "skip", "none", ""}:
        return 0
    raise ValueError(f"Could not parse boolean value: {value}")


def compute_sensor_vector(row: pd.Series) -> np.ndarray:
    current_zone = int(row["sensor_current_zone_id"]) if "sensor_current_zone_id" in row.index else 1
    current_zone = max(1, min(4, current_zone))

    zone_one_hot = np.zeros(4, dtype=np.float32)
    zone_one_hot[current_zone - 1] = 1.0

    continuous = row[SENSOR_COLUMNS].to_numpy(dtype=np.float32)
    return np.concatenate([zone_one_hot, continuous], axis=0).astype(np.float32)


def build_hf_text_tokenizer(model_name: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name)


def build_hf_image_processor(model_name: str):
    from transformers import AutoImageProcessor
    return AutoImageProcessor.from_pretrained(model_name)


def resolve_image_path(image_relpath: str, image_root: Optional[str]) -> Optional[str]:
    if not image_relpath:
        return None

    candidates = []
    rel = str(image_relpath)

    if os.path.isabs(rel):
        candidates.append(rel)
    if image_root is not None:
        candidates.append(os.path.join(image_root, rel))
        candidates.append(os.path.join(image_root, os.path.basename(rel)))
        if rel.startswith("images/") or rel.startswith("images\\"):
            candidates.append(os.path.join(image_root, rel.split("/", 1)[-1].split("\\", 1)[-1]))

    candidates.append(rel)
    candidates.append(os.path.basename(rel))

    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


class GreenhouseVLADataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        text_tokenizer,
        image_processor,
        image_root: Optional[str] = None,
        max_text_len: int = 128,
        sensor_standardizer: Optional[Standardizer] = None,
        target_standardizer: Optional[Standardizer] = None,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.text_tokenizer = text_tokenizer
        self.image_processor = image_processor
        self.image_root = image_root
        self.max_text_len = max_text_len
        self.sensor_standardizer = sensor_standardizer
        self.target_standardizer = target_standardizer

        validate_sensor_columns(self.df)
        self.prompt_col = discover_prompt_column(self.df)
        (
            self.target_regression_cols,
            self.target_zone_col,
            self.target_pesticide_col,
        ) = discover_target_columns(self.df)

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, image_relpath: str) -> torch.Tensor:
        image = None
        image_path = resolve_image_path(image_relpath, self.image_root)

        if image_path is not None:
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception:
                image = None

        if image is None:
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        return pixel_values

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        image_relpath = str(row.get("image_relpath", "")).strip()
        image = self._load_image(image_relpath)

        prompt = str(row.get(self.prompt_col, "")).strip()
        enc = self.text_tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        sensor_vec = compute_sensor_vector(row)
        if self.sensor_standardizer is not None:
            sensor_vec = self.sensor_standardizer.transform(sensor_vec)

        target_reg = row[self.target_regression_cols].to_numpy(dtype=np.float32)
        if self.target_standardizer is not None:
            target_reg = self.target_standardizer.transform(target_reg)

        target_zone = int(row[self.target_zone_col]) - 1
        target_zone = max(0, min(3, target_zone))

        target_pesticide = safe_bool(row[self.target_pesticide_col])

        sample_idx_value = row["sample_idx"] if "sample_idx" in row.index else idx

        return {
            "image": image,
            "input_ids": input_ids.to(torch.long),
            "attention_mask": attention_mask.to(torch.long),
            "sensor": torch.tensor(sensor_vec, dtype=torch.float32),
            "target_regression": torch.tensor(target_reg, dtype=torch.float32),
            "target_zone": torch.tensor(target_zone, dtype=torch.long),
            "target_pesticide": torch.tensor(target_pesticide, dtype=torch.float32),
            "sample_idx": torch.tensor(int(sample_idx_value), dtype=torch.long),
        }


def fit_standardizers(train_df: pd.DataFrame) -> Tuple[Standardizer, Standardizer]:
    validate_sensor_columns(train_df)
    reg_cols, _, _ = discover_target_columns(train_df)

    sensor_matrix = np.stack(
        [compute_sensor_vector(train_df.iloc[i]) for i in range(len(train_df))],
        axis=0,
    )
    target_matrix = train_df[reg_cols].to_numpy(dtype=np.float32)

    sensor_std = Standardizer.fit(sensor_matrix)
    target_std = Standardizer.fit(target_matrix)
    return sensor_std, target_std


def save_preprocessing_artifacts(
    path: str,
    text_model_name: str,
    image_model_name: str,
    sensor_std: Standardizer,
    target_std: Standardizer,
    max_text_len: int,
):
    payload = {
        "text_model_name": text_model_name,
        "image_model_name": image_model_name,
        "max_text_len": max_text_len,
        "sensor_standardizer": sensor_std.to_dict(),
        "target_standardizer": target_std.to_dict(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_preprocessing_artifacts(path: str):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    sensor_std = Standardizer.from_dict(payload["sensor_standardizer"])
    target_std = Standardizer.from_dict(payload["target_standardizer"])

    text_tokenizer = build_hf_text_tokenizer(payload["text_model_name"])
    image_processor = build_hf_image_processor(payload["image_model_name"])

    return {
        "text_model_name": payload["text_model_name"],
        "image_model_name": payload["image_model_name"],
        "max_text_len": payload["max_text_len"],
        "text_tokenizer": text_tokenizer,
        "image_processor": image_processor,
        "sensor_standardizer": sensor_std,
        "target_standardizer": target_std,
    }


def inspect_csv_schema(csv_path: str) -> Dict[str, object]:
    df = pd.read_csv(csv_path)
    validate_sensor_columns(df)
    prompt_col = discover_prompt_column(df)
    reg_cols, zone_col, pesticide_col = discover_target_columns(df)
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "prompt_column": prompt_col,
        "regression_target_columns": reg_cols,
        "zone_target_column": zone_col,
        "pesticide_target_column": pesticide_col,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test the greenhouse dataset loader.")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--text_model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--image_model", type=str, default="google/vit-base-patch16-224")
    parser.add_argument("--max_text_len", type=int, default=128)
    args = parser.parse_args()

    info = inspect_csv_schema(args.csv_path)
    print("Schema:")
    print(json.dumps(info, indent=2))

    df = pd.read_csv(args.csv_path)
    sensor_std, target_std = fit_standardizers(df)
    tokenizer = build_hf_text_tokenizer(args.text_model)
    image_processor = build_hf_image_processor(args.image_model)

    dataset = GreenhouseVLADataset(
        df=df,
        text_tokenizer=tokenizer,
        image_processor=image_processor,
        image_root=args.image_root,
        max_text_len=args.max_text_len,
        sensor_standardizer=sensor_std,
        target_standardizer=target_std,
    )

    sample = dataset[0]
    print("\nSample keys:", list(sample.keys()))
    print("image:", tuple(sample["image"].shape))
    print("input_ids:", tuple(sample["input_ids"].shape))
    print("attention_mask:", tuple(sample["attention_mask"].shape))
    print("sensor:", tuple(sample["sensor"].shape))
    print("target_regression:", tuple(sample["target_regression"].shape))
    print("target_zone:", int(sample["target_zone"]))
    print("target_pesticide:", float(sample["target_pesticide"]))
