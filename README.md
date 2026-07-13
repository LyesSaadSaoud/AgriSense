# AgriSense

## A Digital-Twin-Driven VLA Framework for GNSS-Denied Robotic Greenhouse Autonomy

**Project website:** https://lyessaadsaoud.github.io/AgriSense/  
**Repository:** https://github.com/LyesSaadSaoud/AgriSense

AgriSense integrates an Isaac Sim greenhouse digital twin, a quadruped mobile manipulator, RGB/LiDAR/IMU sensing, GNSS-denied LiDAR–inertial localization, multimodal decision-to-action reasoning, four-zone environmental monitoring, and plant-level targeted intervention.

## Website contents

- Three completed mission videos: All-Zone, Zone 2, and Zone 4
- Updated manuscript figures and LiDAR point-cloud maps
- Navigation and greenhouse-state result tables
- System architecture and spatial-grounding diagrams
- Source code, training, dataset loading, evaluation, augmentation, and reproducibility documentation

## Repository structure

```text
AgriSense/
├── index.html
├── assets/
│   ├── css/style.css
│   ├── js/main.js
│   ├── images/
│   ├── tables/
│   ├── appendix/
│   └── videos/
│       ├── inspect_allzones.webm
│       ├── inspect_zone2.webm
│       └── inspect_zone4.webm
└── code/
    ├── src/
    │   ├── train_greenhouse_vla_better.py
    │   ├── greenhouse_vla_model_bert_dino_pesticide.py
    │   ├── greenhouse_vla_dataset_fixed.py
    │   ├── evaluate_model_offline_fixed.py
    │   ├── plot_greenhouse_vla_per_zone_state_action.py
    │   ├── auto_labler.py
    │   └── augment_strawberry_disease_dataset.py
    ├── configs/example_config.json
    ├── docs/REPRODUCIBILITY.md
    ├── requirements.txt
    └── environment.yml
```

## Installation

```bash
conda env create -f code/environment.yml
conda activate agrisense
```

or

```bash
pip install -r code/requirements.txt
```

## Training

```bash
python code/src/train_greenhouse_vla_better.py
```

## Offline evaluation

```bash
python code/src/evaluate_model_offline_fixed.py
```

## Dataset

The AgriSense dataset contains robot-view RGB observations, prompts, global and per-zone greenhouse states, target setpoints, target-zone labels, pesticide/intervention labels, synchronized ground-truth and LIO trajectories, localization-error logs, environmental logs, and mission summaries.

The dataset is currently hosted in a **private Hugging Face repository during peer review**. The public link will be added after manuscript acceptance.

## Experimental summary

| Mission | Path | Duration | XY RMSE | 3D RMSE |
|---|---:|---:|---:|---:|
| All-Zone | 151.23 m | 607.95 s | 5.34 cm | 5.64 cm |
| Zone 2 | 50.21 m | 235.54 s | 3.43 cm | 3.69 cm |
| Zone 4 | 81.34 m | 222.94 s | 5.16 cm | 5.39 cm |

## Release status

| Resource | Status |
|---|---|
| Project website | Public |
| Mission videos | Public |
| Updated figures and tables | Public |
| Core model/training/evaluation code | Included |
| Dataset | Private during review |
| Checkpoints | Planned after acceptance |

## Contact

Lyes Saad Saoud — saadsaoudl@yahoo.fr
