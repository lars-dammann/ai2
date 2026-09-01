# U-Net-based Prediction of Concealed Volume Loss - Code [![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) [![Python](https://img.shields.io/badge/python-3.12-green)](https://www.python.org/) [![Paper](https://img.shields.io/badge/paper-INSERT_DOI-red)](https://doi.org/) [![Data](https://img.shields.io/badge/data-10.15480/882.17491-orange)](https://doi.org/10.15480/882.17491) [![WandB](https://img.shields.io/badge/WandB-INSERT_LINK-yellow)](#)

This repository implements the U-Net regression code used in the paper "U-Net-based Prediction of Concealed Volume Loss in AZ31 from Optical Corrosion Imprints and As-Exposed Surface Topography". The model predicts pixel-wise corrosion depth from optical imprints, enabling estimation of concealed volume loss without removing corrosion products.


Quick links
- Paper: [DOI placeholder]
- Dataset: https://doi.org/10.15480/882.17491
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

- `ai2/` — Implementation of the regression U-Net
    - `ai2/crossvalidation.py` — Script to run training for cross-validation folds
    - `ai2/predict.py` — Script to generate predictions from trained models
    - `ai2/tune.py` — Script to execute hyperparameter optimization with Optuna
    - `ai2/datamodule/` — PyTorch datamodules and dataloaders
    - `ai2/model/` — PyTorch Lightning model definitions
        - `ai2/model/unet.py` — Implementation of the PyTorch U-Net
        - `ai2/model/lit_model.py` — PyTorchLightning model implementing the training, valdiation and inference steps as well as the loss calculations
    - `ai2/utils/` — Utility code (data splitting, normalization and dataset classes)
- `configs/` — JSON configuration files for datamodule, model and trainer for crossvalidaton training (`crossval-config.json`) and hyperparameter optimization (`tune-config.json`)
- `preprocess/` — Notebooks and scripts describing the preprocessing of the original collected data (e.g. ROI selection and alignment)
- `results/` — Evaluation notebooks and scripts used to generate results and figures reported in the paper

## Data

The directories `data`, `model-configs`, `models`, and `predictions` are published separately as a dataset (https://doi.org/10.15480/882.17491). The dataset is released under the CC BY 4.0 license. The data in the `data` folder is a processed subset of the data orginially collected by

C. Song, B. Vaghefinazari, T. Würger, A. Lisitsyna, D. Mei, M. Nienaber, J. Bohlen, M. L. Zheludkevich, S. Albarqouni, C. Feiler, S. V. Lamaka, *Corrosion Science* **2025**, *250* 112903 (https://doi.org/10.1016/j.corsci.2025.112903).

## Configuration

- `configs/crossval-config.json` — configuration used for crossvalidation experiments.
- `configs/tune-config.json` — configuration for hyperparameter search.
- `model-configs/*` — configurations of the 18 individual trained models

## Weights & Biases (W&B)

Public project: [W&B project placeholder](https://wandb.ai/<entity>/<project>)

This repository logs experiments to Weights & Biases (W&B). The public project link above will list runs, metrics, configs and artifacts (models/checkpoints, predictions) used in the paper.

## Citation

Please cite the paper if you use this code.

L. Dammann, C. Song, B. Vaghefinazari, T. Würger, A. Lisitsyna,S. Albarqouni, M. L. Zheludkevich,
S. V. Lamaka, C. Feiler
(https://doi.org/)

## License

This project is released under the BSD-3-Clause license.

## Contact

Author: Lars Dammann — https://github.com/lars-dammann

---
