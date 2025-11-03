import random
import os

import numpy as np
import optuna
from optuna.integration import PyTorchLightningPruningCallback
# from optuna.integration.wandb import WeightsAndBiasesCallback
import torch
import torch.nn as nn
from lightning.pytorch import Trainer, Callback, seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
# from lightning.pytorch.callbacks import ModelCheckpoint
# from lightning.pytorch.strategies import DDPStrategy
import wandb

from datamodule.datamodule import CorrosionDataModule
from model.lit_model import CorrosionUNet
from utils.get_config import get_config



def weights_init(model):
    if isinstance(model, nn.Conv2d):
        nn.init.kaiming_normal_(model.weight, mode='fan_out', nonlinearity='relu')
        if model.bias is not None:
            nn.init.constant_(model.bias, 0)
    elif isinstance(model, nn.BatchNorm2d):
        nn.init.constant_(model.weight, 1)
        nn.init.constant_(model.bias, 0)

def objective(trial):
    wandb.finish()
    trial.suggest_int("udepth", 3, 3)
    trial.suggest_int("startfeature", 8, 8)
    trial.suggest_int("batchsize", 2, 4)
    trial.suggest_int("datasize", 256, 256)
    trial.suggest_float('lr', 1e-4, 1e-2, log=True)

    # Load and unite configs
    config = get_config(trial.params)

    try:
        config["n_workers"] = int(os.getenv('SLURM_CPUS_PER_TASK'))
    except TypeError:
        pass

    model = CorrosionUNet(config=config["unet"])
    model.apply(weights_init)

    datamodule = CorrosionDataModule(config=config["datamodule"])

    # initialise the wandb logger and name your wandb project
    wandb_logger = WandbLogger(project="ai2", name=f"Trial {trial.number}")
    wandb_logger.experiment.config.update(config)

    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=False,
        mode='min'
    )

    # Optuna pruning callback
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor='val_loss')

    nnodes = int(os.getenv("SLURM_NNODES"))
    trainer = Trainer(
        devices="auto",
        accelerator="auto",
        logger=wandb_logger,
        callbacks=[early_stop_callback, pruning_callback],
        max_epochs=2,
        num_nodes=nnodes,
        log_every_n_steps=20,
        )

    trainer.fit(model, datamodule=datamodule)
    return trainer.callback_metrics["val_loss"].item()

def run_optimization(n_trials=5):
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(direction='minimize', pruner=pruner)
    study.optimize(objective, n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return study

# def test_best_model(study):
#     # Getting the best hyperparameters
#     best_params = study.best_trial.params

#     # Creating the model with the best hyperparameters
#     # Load and unite configs
#     config = get_config(best_params)

#     # Creating trainer instance
#     trainer = Trainer(max_epochs=10)

#     datamodule = CorrosionDataModule(config=config["datamodule"])

#     # Training the model with the best hyperparameters
#     trainer.fit(model, datamodule=datamodule)

#     # Testing the model with the test data
#     results = trainer.test(model, test_loader)
#     return results

study = run_optimization(n_trials=5)