# AgriSense

## A Digital-Twin-Driven VLM Framework for GNSS-Denied Robotic Greenhouse Autonomy

[![Project Website](https://img.shields.io/badge/Project-Website-green)](https://lyessaadsaoud.github.io/AgriSense/)
[![Repository](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/LyesSaadSaoud/AgriSense)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](#dataset)
[![Status](https://img.shields.io/badge/Status-Under%20Review-orange)](#release-status)

AgriSense is a digital-twin-driven robotic greenhouse autonomy framework integrating:

- NVIDIA Isaac Sim greenhouse simulation
- Quadruped mobile manipulation
- RGB, LiDAR, IMU, and environmental sensing
- GNSS-denied LiDAR–inertial localization
- Vision-language decision-to-action reasoning
- Zone-level environmental monitoring
- Plant-level inspection and targeted intervention
- Safety-constrained command validation

## Project website

The interactive project website is available at:

https://lyessaadsaoud.github.io/AgriSense/

It includes the three recorded inspection missions:

- All-Zone inspection
- Zone 2 inspection
- Zone 4 inspection

## System overview

<p align="center">
  <img
    src="assets/images/system_pipeline.png"
    width="100%"
    alt="AgriSense system pipeline"
  >
</p>

The framework receives user instructions, robot-view RGB images, LiDAR observations, IMU measurements, greenhouse sensor states, and robot spatial context. The multimodal decision module produces structured navigation, environmental-control, and plant-level intervention proposals.

All proposals are checked for confidence, freshness, feasibility, localization consistency, safety, and manipulator reachability before execution.

## Repository structure

```text
AgriSense/
├── assets/
│   ├── css/
│   ├── images/
│   ├── js/
│   └── videos/
│
├── configs/
│   ├── training.yaml
│   ├── evaluation.yaml
│   └── dataset.yaml
│
├── dataset/
│   ├── README.md
│   ├── examples/
│   └── metadata/
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py
│   ├── generate_figures.py
│   └── evaluate_localization.py
│
├── src/
│   └── agrisense/
│       ├── data/
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       ├── localization/
│       ├── control/
│       └── utils/
│
├── checkpoints/
│   └── README.md
│
├── supplementary/
│   ├── figures/
│   ├── tables/
│   └── dashboards/
│
├── index.html
├── requirements.txt
├── environment.yml
├── CITATION.cff
├── LICENSE
└── README.md
```

The repository is being organized progressively. Directories marked as pending release will be populated according to the release status below.

## Installation

### Conda environment

```bash
conda env create -f environment.yml
conda activate agrisense
```

### Pip environment

```bash
python -m venv .venv
```

**Linux/macOS:**

```bash
source .venv/bin/activate
```

**Windows:**

```cmd
.venv\Scripts\activate
```

**Install dependencies:**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset

The AgriSense dataset contains:

- Robot-view greenhouse RGB observations
- Synthetic crop-condition examples
- User and mission prompts
- Global and per-zone greenhouse states
- Target environmental setpoints
- Target-zone labels
- Pesticide/intervention labels
- Ground-truth and LiDAR–inertial trajectories
- Synchronized localization-error logs
- Environmental monitoring logs
- Mission-level evaluation summaries

The dataset is currently hosted in a private Hugging Face repository during peer review. The public Hugging Face link will be added after manuscript acceptance. 

See: `dataset/README.md`

## Training

The multimodal model combines:

- Visual encoder
- Text encoder
- Greenhouse sensor-state encoder
- Multimodal fusion layers
- Setpoint-regression head
- Target-zone classification head
- Pesticide-decision head

**Expected training command:**

```bash
python scripts/train.py \
  --config configs/training.yaml
```

**Resume from a checkpoint:**

```bash
python scripts/train.py \
  --config configs/training.yaml \
  --resume checkpoints/best_model.pt
```

Training scripts and checkpoints will be released according to the release schedule below.

## Evaluation

**Run multimodal decision evaluation:**

```bash
python scripts/evaluate.py \
  --config configs/evaluation.yaml \
  --checkpoint checkpoints/best_model.pt
```

**Run inference on a sample:**

```bash
python scripts/infer.py \
  --config configs/evaluation.yaml \
  --checkpoint checkpoints/best_model.pt \
  --sample dataset/examples/example.json
```

**Run localization evaluation:**

```bash
python scripts/evaluate_localization.py \
  --input supplementary/logs/inspect_allzones/synced_errors.csv
```

**Generate manuscript figures:**

```bash
python scripts/generate_figures.py \
  --config configs/evaluation.yaml
```

## Inspection videos

The project website includes the following compressed mission videos:

- `assets/videos/inspect_allzones.webm`
- `assets/videos/inspect_zone2.webm`
- `assets/videos/inspect_zone4.webm`

## Experimental summary

| Mission | Path length | Duration | XY RMSE | 3D RMSE |
|---|---|---|---|---|
| All-Zone | 151.23 m | 607.95 s | 5.34 cm | 5.64 cm |
| Zone 2 | 50.21 m | 235.54 s | 3.43 cm | 3.69 cm |
| Zone 4 | 81.34 m | 222.94 s | 5.16 cm | 5.39 cm |

## VLDA evaluation

| Metric | Value |
|---|---|
| Test samples | 43 |
| Setpoint MAE | 1.266 |
| Target-zone accuracy | 95.35% |
| Target-zone macro-F1 | 93.11% |
| Pesticide accuracy | 99.00% |
| Pesticide F1 | 99.00% |

## Release status

| Resource | Status |
|---|---|
| Project website | Public |
| Mission videos | Public |
| Updated project figures | Public |
| Core implementation | Release in progress |
| Training scripts | Release in progress |
| Evaluation scripts | Release in progress |
| Configuration files | Release in progress |
| Dataset | Private during review |
| Model checkpoints | Planned after acceptance |
| Supplementary dashboards | Planned release |

## Reproducibility

The supplementary evaluation package includes, for each mission:

- `synced_errors.csv`
- `greenhouse_state.csv`
- `metrics_summary.csv`
- `live_dashboard.html`

These files support regeneration of the localization and environmental-monitoring results reported in the manuscript.

## Citation

Citation information will be updated after publication.

```bibtex
@article{saoud2026agrisense,
  title   = {AgriSense: A Digital-Twin-Driven VLM Framework for
             GNSS-Denied Robotic Greenhouse Autonomy},
  author  = {Saad Saoud, Lyes and Doukhi, Oualid and
             Ghorbani, Reza and Ayyash, Moussa},
  year    = {2026},
  note    = {Manuscript under review}
}
```

## Contact

Lyes Saad Saoud
Email: saadsaoudl@yahoo.fr

## License

A license file will accompany the public software release. Dataset and model artifacts may be distributed under separate terms.
