# AgriSense Algorithms

Reference implementation accompanying **AgriSense: A Digital-Twin-Driven VLA Framework for GNSS-Denied Robotic Greenhouse Autonomy**.

> **Release status:** research preview. Dataset links, trained checkpoints, ROS 2 deployment nodes, and final command examples are **Coming Soon**.

## Included scripts

| Script | Purpose |
|---|---|
| `train_greenhouse_vla_better.py` | Trains the multimodal greenhouse decision model. |
| `evaluate_model_offline_fixed.py` | Runs offline evaluation and exports metrics. |
| `greenhouse_vla_dataset_fixed.py` | Loads image, language, state, and action samples. |
| `greenhouse_vla_model_bert_dino_pesticide.py` | Defines the BERT/vision-based multi-head model. |
| `plot_greenhouse_vla_per_zone_state_action.py` | Produces per-zone state/action visualizations. |
| `auto_labler.py` | Assists structured sample labeling with an API-based workflow. |
| `augment_strawberry_disease_dataset.py` | Supports image-dataset augmentation. |

## Installation
```bash
conda env create -f environment.yml
conda activate agrisense
```
Or:
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Typical workflow
```bash
python src/train_greenhouse_vla_better.py --help
python src/evaluate_model_offline_fixed.py --help
python src/plot_greenhouse_vla_per_zone_state_action.py --help
```

## Repository structure
```text
AgriSense_Algorithms/
├── src/                 # Uploaded research scripts
├── configs/             # Example configuration
├── data/                # Dataset is not redistributed here
├── docs/                # Reproducibility and release notes
├── examples/            # Usage examples (Coming Soon)
├── tests/               # Unit/integration tests (Coming Soon)
├── requirements.txt
├── environment.yml
├── CITATION.cff
└── LICENSE
```

## Reproducibility status
The uploaded scripts are preserved with minimal packaging changes. Exact dataset paths, checkpoints, Isaac Sim assets, ROS 2 nodes, and hardware/deployment configuration were not present in the source archive and are therefore not fabricated. Add them before claiming full end-to-end reproducibility.

## Security
The labeling and augmentation scripts may use an API key. Store keys in environment variables; never commit credentials.

## Citation
Citation details will be updated after journal acceptance. Until then, cite the manuscript title and this repository version.
