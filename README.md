# AgriSense

## A Digital-Twin-Driven VLA Framework for GNSS-Denied Robotic Greenhouse Autonomy

[![Project Website](https://img.shields.io/badge/Project-Website-green)](https://lyessaadsaoud.github.io/AgriSense/)
[![Repository](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/LyesSaadSaoud/AgriSense)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](#dataset)
[![Status](https://img.shields.io/badge/Status-Under%20Review-orange)](#release-status)

AgriSense is a digital-twin-driven robotic greenhouse autonomy
framework integrating:

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

The framework receives user instructions, robot-view RGB images,
LiDAR observations, IMU measurements, greenhouse sensor states, and
robot spatial context. The multimodal decision module produces
structured navigation, environmental-control, and plant-level
intervention proposals.

All proposals are checked for confidence, freshness, feasibility,
localization consistency, safety, and manipulator reachability before
execution.

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
