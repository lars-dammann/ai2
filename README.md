# Corrosion Beneath the Crust - Code [![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) [![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/) [![DOI](https://img.shields.io/badge/doi-INSERT_DOI-blue)](https://doi.org/INSERT_DOI_HERE) [![WandB](https://img.shields.io/badge/WandB-enabled-yellow)](#)

This repository implements the U-Net regression code used in the paper "Corrosion Beneath the Crust: Determination of Concealed Volume Loss from Optical Corrosion Imprints by Quantitative Imaging". The model predicts pixel-wise corrosion depth from optical imprints, enabling estimation of concealed volume loss without removing corrosion products.

Paper: [Corrosion Beneath the Crust - DOI: INSERT_DOI_HERE]

Quick links
- Paper: [DOI placeholder]
- Dataset: [Dataset DOI placeholder]
- W&B: [link to project]

## Quickstart

1. Create and activate the environment:

```bash
mamba create -f ai2.yml -y
mamba activate ai2
```

2. Train on cross-validation folds:

```bash
# Use data directory from published dataset
export COMPLETE_DATA_DIR=/path/to/data
# Use data/modulator-list.csv from published dataset
export MODULATOR_LIST_FILE=/path/to/data/modulator-list.csv
export CONFIG_FILE=configs/crossval-config.json
# Note: The folds will be created in SPLIT_DATA_DIR by first running the python script
export SPLIT_DATA_DIR=/path/to/splits
export CHECKPOINT_DIR=/path/to/checkpoints
export WANDB_GROUP=your-wandb-group
export SLURM_NNODES=1
python ai2/crossvalidation.py
```

2. Run inference (adjust paths):

```bash
# Use the crossvalidation folds created above
export SPLIT_DATA_DIR=/path/to/data
# Use model-configs directory from published dataset
export CONFIG_DIR=/path/to/configs
# Use models directory from published dataset
export MODEL_DIR=/path/to/models
export PREDICTION_DIR=/path/to/predictions
python ai2/predict.py
```

### Hyperparameter optimization example

To run hyperparameter optimization (tuning), set the required environment variables and run `ai2/tune.py`:

```bash
# Requires a `train`, `val` and `test` subfolders structure. Could be one fold of the crossvalidation subfolders, e.g. in $SPLIT_DATA_DIR/0
export DATA_DIR=/path/to/data
export CONFIG_FILE=configs/tune-config.json
export WANDB_PROJECT=your-wandb-project
export WANDB_GROUP=your-wandb-group
export CHECKPOINT_DIR=/path/to/checkpoints
export SLURM_NNODES=1
python ai2/tune.py
```

## Repository layout

- `ai2/` — training, inference and tuning scripts
    - `ai2/crossvalidation.py` — run training for cross-validation folds
    - `ai2/predict.py` — generate predictions from trained models
    - `ai2/tune.py` — hyperparameter optimization
    - `ai2/datamodule/` — torch datamodules and dataloaders
    - `ai2/model/` — PyTorch Lightning model definitions (`lit_model.py`, `unet.py`)
    - `ai2/utils/` — helpers (data splitting, normalization, dataset)
- `configs/` — JSON configs for datamodule, model and trainer (e.g., `crossval-config.json`, `tune-config.json`)
- `preprocess/` — notebooks and scripts for preprocessing the original collected data
- `results/` — evaluation notebooks and scripts used to generate figures

## Data

The directories `data`, `model-configs`, `models`, and `predictions` are published separately as a dataset (DOI: INSERT_DATASET_DOI). The dataset is released under the CC BY 4.0 license.

## Configuration

- `configs/crossval-config.json` — configuration used for crossvalidation experiments.
- `configs/tune-config.json` — configuration for hyperparameter search.
- `model-configs/*` — configurations of the 18 individual trained models

## Weights & Biases (W&B)

Public project: [W&B project placeholder](https://wandb.ai/<entity>/<project>)

This repository logs experiments to Weights & Biases (W&B). The public project link above will list runs, metrics, configs and artifacts (models/checkpoints, predictions) used in the paper.

## Citation

Please cite the paper if you use this code. Example BibTeX (replace fields):

```bibtex
@article{your2026corrosion,
    title={Corrosion Beneath the Crust: Determination of Concealed Volume Loss from Optical Corrosion Imprints by Quantitative Imaging},
    author={Author, A. and Author, B.},
    journal={Journal Name},
    year={2026},
    doi={INSERT_DOI}
}
```

## License

This project is released under the BSD-3-Clause license.

## Contact

Author: Lars Dammann — https://github.com/lars-dammann

---
