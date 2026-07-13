#!/usr/bin/env python3
import argparse
import base64
import csv
import json
import os
import shutil
import time
from typing import Dict, List, Tuple

from openai import OpenAI


# -----------------------------------------------------------------------------
# Numeric helpers
# -----------------------------------------------------------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        s = str(value).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def safe_int(value, default: int = 0) -> int:
    try:
        if value is None:
            return default
        s = str(value).strip()
        if s == "":
            return default
        return int(float(s))
    except Exception:
        return default


# -----------------------------------------------------------------------------
# CSV IO
# -----------------------------------------------------------------------------
def read_csv_rows(csv_path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def write_csv_rows(csv_path: str, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    tmp_path = csv_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, csv_path)


# -----------------------------------------------------------------------------
# Image helpers
# -----------------------------------------------------------------------------
def resolve_image_path(dataset_root: str, image_relpath: str) -> str:
    if os.path.isabs(image_relpath):
        return image_relpath
    return os.path.join(dataset_root, image_relpath)


def image_file_to_data_url(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/jpeg"
    if ext == ".png":
        mime = "image/png"
    elif ext == ".webp":
        mime = "image/webp"

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime};base64,{b64}"


# -----------------------------------------------------------------------------
# Dataset schema helpers
# -----------------------------------------------------------------------------
def ensure_columns(rows: List[Dict[str, str]], fieldnames: List[str], zones_count: int) -> List[str]:
    required = []

    for i in range(1, zones_count + 1):
        required += [
            f"target_zone_{i}_temp_c",
            f"target_zone_{i}_humidity_pct",
            f"target_zone_{i}_soil_moisture_pct",
        ]

    required += [
        "gpt_refined_prompt",
        "gpt_short_instruction",
        "gpt_action_summary",
        "prompt_profile",
        "robot_target_zone_id",
        "robot_zone_priority_json",
        "reason_tags_json",
        "apply_pesticide",
        "pesticide_target_zone_id",
        "pesticide_reason",
        "pesticide_confidence",
        "label_source",
        "label_status",
        "label_error",
    ]

    for col in required:
        if col not in fieldnames:
            fieldnames.append(col)
            for row in rows:
                row[col] = ""

    return fieldnames


def build_row_payload(row: Dict[str, str], zones_count: int) -> Dict:
    zones = {}
    profiles = {}
    scores = {}

    for i in range(1, zones_count + 1):
        zones[f"zone_{i}"] = {
            "temp_c": safe_float(row.get(f"zone_{i}_temp")),
            "humidity_pct": safe_float(row.get(f"zone_{i}_humidity")),
            "soil_moisture_pct": safe_float(row.get(f"zone_{i}_soil")),
        }

        profiles[f"zone_{i}"] = {
            "profile_temp_c": safe_float(row.get(f"profile_zone_{i}_temp")),
            "profile_humidity_pct": safe_float(row.get(f"profile_zone_{i}_humidity")),
            "profile_soil_moisture_pct": safe_float(row.get(f"profile_zone_{i}_soil")),
        }

        if f"zone_{i}_score" in row:
            scores[f"zone_{i}"] = safe_float(row.get(f"zone_{i}_score"))

    payload = {
        "sample_idx": safe_int(row.get("sample_idx"), -1),
        "task_prompt": row.get("task_prompt", "").strip(),
        "current_zone_id": safe_int(row.get("sensor_current_zone_id"), 1),
        "global_env": {
            "temp_c": safe_float(row.get("global_temp")),
            "humidity_pct": safe_float(row.get("global_humidity")),
            "soil_moisture_pct": safe_float(row.get("global_soil")),
        },
        "zones": zones,
        "zone_profiles": profiles,
        "zone_scores": scores,
        "dataset_hints": {
            "is_augmented": str(row.get("is_augmented", "")).strip(),
            "disease_class": str(row.get("disease_class", "")).strip(),
            "source_image_relpath": str(row.get("source_image_relpath", "")).strip(),
            "measured_most_critical_zone_id": safe_int(row.get("measured_most_critical_zone_id"), 0),
            "scenario_primary_bad_zone_id": safe_int(row.get("scenario_primary_bad_zone_id"), 0),
        },
        "objective": (
            "Generate stable and realistic greenhouse supervision labels for strawberries. "
            "Use both the image and the structured sensor state. "
            "Also generate concise, useful natural-language prompts for the robot/operator. "
            "Additionally decide whether pesticide application is needed. "
            "Be conservative: only set apply_pesticide=true when visible disease or pest-like symptoms "
            "strongly justify it and environmental correction alone is unlikely to be enough."
        ),
    }
    return payload


def build_schema(zones_count: int) -> Dict:
    return {
        "name": "greenhouse_label_with_prompts_and_pesticide",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "task_prompt": {"type": "string"},
                "gpt_refined_prompt": {"type": "string"},
                "gpt_short_instruction": {"type": "string"},
                "gpt_action_summary": {"type": "string"},
                "prompt_profile": {
                    "type": "string",
                    "enum": [
                        "balanced",
                        "fruit_quality",
                        "power_save",
                        "fungal_risk",
                        "water_recovery",
                    ],
                },
                "current_zone_id": {"type": "integer"},
                "zone_targets": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        f"zone_{i}": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "target_temp_c": {"type": "number"},
                                "target_humidity_pct": {"type": "number"},
                                "target_soil_moisture_pct": {"type": "number"},
                            },
                            "required": [
                                "target_temp_c",
                                "target_humidity_pct",
                                "target_soil_moisture_pct",
                            ],
                        }
                        for i in range(1, zones_count + 1)
                    },
                    "required": [f"zone_{i}" for i in range(1, zones_count + 1)],
                },
                "robot_plan": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "target_zone_id": {"type": "integer"},
                        "zone_priority": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "zone": {"type": "integer"},
                                    "score": {"type": "number"},
                                    "reason": {"type": "string"},
                                },
                                "required": ["zone", "score", "reason"],
                            },
                        },
                    },
                    "required": ["target_zone_id", "zone_priority"],
                },
                "apply_pesticide": {"type": "boolean"},
                "pesticide_target_zone_id": {"type": "integer"},
                "pesticide_reason": {"type": "string"},
                "pesticide_confidence": {"type": "number"},
                "reason_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "task_prompt",
                "gpt_refined_prompt",
                "gpt_short_instruction",
                "gpt_action_summary",
                "prompt_profile",
                "current_zone_id",
                "zone_targets",
                "robot_plan",
                "apply_pesticide",
                "pesticide_target_zone_id",
                "pesticide_reason",
                "pesticide_confidence",
                "reason_tags",
            ],
        },
    }


def sanitize_label(label: Dict, zones_count: int) -> Dict:
    for i in range(1, zones_count + 1):
        z = label["zone_targets"][f"zone_{i}"]
        z["target_temp_c"] = round(clamp(float(z["target_temp_c"]), 16.0, 24.0), 2)
        z["target_humidity_pct"] = round(clamp(float(z["target_humidity_pct"]), 45.0, 65.0), 2)
        z["target_soil_moisture_pct"] = round(clamp(float(z["target_soil_moisture_pct"]), 55.0, 75.0), 2)

    label["current_zone_id"] = int(clamp(int(label["current_zone_id"]), 1, zones_count))
    label["robot_plan"]["target_zone_id"] = int(
        clamp(int(label["robot_plan"]["target_zone_id"]), 1, zones_count)
    )

    pesticide_zone_id = int(label["pesticide_target_zone_id"])
    label["pesticide_target_zone_id"] = int(clamp(pesticide_zone_id, 1, zones_count))

    label["apply_pesticide"] = bool(label["apply_pesticide"])
    label["pesticide_reason"] = str(label["pesticide_reason"]).strip()
    label["pesticide_confidence"] = round(clamp(float(label["pesticide_confidence"]), 0.0, 1.0), 3)

    cleaned_priority = []
    for item in label["robot_plan"]["zone_priority"]:
        zone_id = int(clamp(int(item["zone"]), 1, zones_count))
        score = float(item["score"])
        reason = str(item["reason"]).strip()
        cleaned_priority.append({
            "zone": zone_id,
            "score": score,
            "reason": reason,
        })

    label["robot_plan"]["zone_priority"] = cleaned_priority[:zones_count]

    label["task_prompt"] = str(label["task_prompt"]).strip()
    label["gpt_refined_prompt"] = str(label["gpt_refined_prompt"]).strip()
    label["gpt_short_instruction"] = str(label["gpt_short_instruction"]).strip()
    label["gpt_action_summary"] = str(label["gpt_action_summary"]).strip()
    label["prompt_profile"] = str(label["prompt_profile"]).strip()
    label["reason_tags"] = [str(x).strip() for x in label["reason_tags"] if str(x).strip()]

    if not label["apply_pesticide"]:
        label["pesticide_confidence"] = min(label["pesticide_confidence"], 0.5)

    return label


# -----------------------------------------------------------------------------
# Error formatting
# -----------------------------------------------------------------------------
def format_exception(e: Exception) -> str:
    parts = [repr(e)]

    body = getattr(e, "body", None)
    if body is not None:
        try:
            parts.append(f"body={json.dumps(body, ensure_ascii=False)}")
        except Exception:
            parts.append(f"body={body}")

    response = getattr(e, "response", None)
    if response is not None:
        try:
            parts.append(f"status_code={response.status_code}")
        except Exception:
            pass
        try:
            parts.append(f"response_text={response.text}")
        except Exception:
            pass

    return " | ".join(parts)


# -----------------------------------------------------------------------------
# OpenAI request
# -----------------------------------------------------------------------------
def request_label(
    client: OpenAI,
    model_name: str,
    row_payload: Dict,
    image_data_url: str,
    schema: Dict,
    image_detail: str,
) -> Dict:
    developer_text = (
        "You are an expert strawberry greenhouse supervisor. "
        "Use BOTH the greenhouse image and the structured environmental sensor data. "
        "The image can reveal wilt, discoloration, fungal symptoms, fruit condition, canopy stress, "
        "mold, overwatering, underwatering, or ripeness issues. "
        "Return only valid JSON matching the schema exactly. "
        "Generate stable, realistic target labels for all zones. "
        "Also generate: "
        "gpt_refined_prompt, gpt_short_instruction, gpt_action_summary, "
        "apply_pesticide, pesticide_target_zone_id, pesticide_reason, pesticide_confidence. "
        "Be conservative about pesticide use. "
        "Do not recommend pesticide for mild stress that can likely be corrected with climate or watering alone. "
        "Prefer gradual correction toward healthy strawberry greenhouse conditions. "
        "Pick the robot target zone based on the most urgent and useful intervention."
    )

    user_text = json.dumps(row_payload, ensure_ascii=False)

    request_common = dict(
        model=model_name,
        input=[
            {
                "role": "developer",
                "content": [
                    {"type": "input_text", "text": developer_text}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_text},
                    {
                        "type": "input_image",
                        "image_url": image_data_url,
                        "detail": image_detail,
                    },
                ],
            },
        ],
    )

    # First try strict mode. If the API rejects the schema subset, retry non-strict.
    attempts = [
        {
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema["name"],
                    "schema": schema["schema"],
                    "strict": True,
                }
            }
        },
        {
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema["name"],
                    "schema": schema["schema"],
                    "strict": False,
                }
            }
        },
    ]

    last_error = None

    for attempt_idx, extra in enumerate(attempts, start=1):
        try:
            response = client.responses.create(**request_common, **extra)
            output_text = getattr(response, "output_text", None)
            if not output_text:
                raise RuntimeError("Empty response.output_text")
            return json.loads(output_text)
        except Exception as e:
            last_error = e
            print(f"[DEBUG] request attempt {attempt_idx} failed -> {format_exception(e)}")

    raise RuntimeError(f"Responses API request failed after fallback: {format_exception(last_error)}")


# -----------------------------------------------------------------------------
# Row write-back
# -----------------------------------------------------------------------------
def fill_row_with_label(row: Dict[str, str], label: Dict, zones_count: int) -> None:
    row["gpt_refined_prompt"] = label["gpt_refined_prompt"]
    row["gpt_short_instruction"] = label["gpt_short_instruction"]
    row["gpt_action_summary"] = label["gpt_action_summary"]

    row["prompt_profile"] = label["prompt_profile"]
    row["robot_target_zone_id"] = str(label["robot_plan"]["target_zone_id"])
    row["robot_zone_priority_json"] = json.dumps(label["robot_plan"]["zone_priority"], ensure_ascii=False)
    row["reason_tags_json"] = json.dumps(label["reason_tags"], ensure_ascii=False)

    row["apply_pesticide"] = "true" if label["apply_pesticide"] else "false"
    row["pesticide_target_zone_id"] = str(label["pesticide_target_zone_id"])
    row["pesticide_reason"] = label["pesticide_reason"]
    row["pesticide_confidence"] = str(label["pesticide_confidence"])

    row["label_source"] = "openai_offline_image_teacher_v4"
    row["label_status"] = "labeled"
    row["label_error"] = ""

    for i in range(1, zones_count + 1):
        z = label["zone_targets"][f"zone_{i}"]
        row[f"target_zone_{i}_temp_c"] = str(z["target_temp_c"])
        row[f"target_zone_{i}_humidity_pct"] = str(z["target_humidity_pct"])
        row[f"target_zone_{i}_soil_moisture_pct"] = str(z["target_soil_moisture_pct"])


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate greenhouse target labels, GPT prompts, and pesticide decision from images + sensor state."
    )
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV to label")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root directory for resolving image_relpath")
    parser.add_argument("--zones_count", type=int, default=4)
    parser.add_argument("--model", type=str, default=os.getenv("OPENAI_MODEL", "").strip() or "gpt-4o-mini")
    parser.add_argument("--image_detail", type=str, default="low", choices=["low", "high", "auto"])
    parser.add_argument("--sleep_sec", type=float, default=0.25)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--max_rows", type=int, default=-1)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--force_relabel", action="store_true")
    parser.add_argument("--backup", action="store_true")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)

    rows, fieldnames = read_csv_rows(args.csv_path)
    fieldnames = ensure_columns(rows, fieldnames, args.zones_count)

    if args.backup:
        backup_path = args.csv_path + ".bak"
        if not os.path.exists(backup_path):
            shutil.copy2(args.csv_path, backup_path)
            print(f"[INFO] Backup created: {backup_path}")

    schema = build_schema(args.zones_count)

    processed = 0
    skipped = 0
    failed = 0
    total = len(rows)

    print(f"[INFO] model={args.model}")
    print(f"[INFO] image_detail={args.image_detail}")
    print(f"[INFO] total_rows={total}")

    for idx, row in enumerate(rows):
        if idx < args.start_index:
            continue

        if args.max_rows > 0 and processed >= args.max_rows:
            break

        status = row.get("label_status", "").strip().lower()
        if (not args.force_relabel) and status == "labeled":
            skipped += 1
            continue

        try:
            image_relpath = str(row.get("image_relpath", "")).strip()
            if not image_relpath:
                raise ValueError("Missing image_relpath")

            image_path = resolve_image_path(args.dataset_root, image_relpath)
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Missing image: {image_path}")

            image_data_url = image_file_to_data_url(image_path)
            payload = build_row_payload(row, args.zones_count)

            label = request_label(
                client=client,
                model_name=args.model,
                row_payload=payload,
                image_data_url=image_data_url,
                schema=schema,
                image_detail=args.image_detail,
            )

            label = sanitize_label(label, args.zones_count)
            fill_row_with_label(row, label, args.zones_count)

            processed += 1
            print(
                f"[OK] row={idx+1}/{total} "
                f"sample_idx={row.get('sample_idx', '')} "
                f"target_zone={row.get('robot_target_zone_id', '')} "
                f"apply_pesticide={row.get('apply_pesticide', '')}"
            )

            if processed % max(1, args.save_every) == 0:
                write_csv_rows(args.csv_path, rows, fieldnames)
                print(f"[SAVE] Progress written to {args.csv_path}")

            time.sleep(args.sleep_sec)

        except Exception as e:
            failed += 1
            err_msg = format_exception(e)
            row["label_status"] = "error"
            row["label_error"] = err_msg
            print(f"[ERR] row={idx+1}/{total} -> {err_msg}")

            if failed % max(1, args.save_every) == 0:
                write_csv_rows(args.csv_path, rows, fieldnames)
                print(f"[SAVE] Progress written to {args.csv_path}")

    write_csv_rows(args.csv_path, rows, fieldnames)

    print("")
    print("Done.")
    print(f"  labeled : {processed}")
    print(f"  skipped : {skipped}")
    print(f"  failed  : {failed}")
    print(f"  output  : {args.csv_path}")


if __name__ == "__main__":
    main()