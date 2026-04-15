import os
from pathlib import Path
import logging
import traceback

import optuna
from optuna.integration import PyTorchLightningPruningCallback
import torch
import torch.nn as nn
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import wandb

from datamodule.datamodule import CorrosionDataModule
from model.lit_model import CorrosionUNet
from utils.get_data import get_config

seed_everything(0, workers=True)

group = os.getenv("WANDB_GROUP")


def objective(trial):
    wandb.finish()
    trial.suggest_int("udepth", 3, 6)
    trial.suggest_categorical("startfeature", [16, 32, 64])
    trial.suggest_categorical("datasize", [1024])
    trial.suggest_float('lr', 1e-6, 1e-1, log=True)
    trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)

    # Load and unite configs
    config_file = Path(os.getenv("CONFIG_FILE"))
    config = get_config(config_file, trial.params)

    model = CorrosionUNet(model_config=config["unet"])
    model.apply(model.weights_init)

    datamodule = CorrosionDataModule(datamodule_config=config["datamodule"])

    # initialise the wandb logger and name your wandb project
    wandb_logger = WandbLogger(project=os.getenv("WANDB_PROJECT"), group=group,
                               name=f"{group} Trial {trial.number}", log_model=True)
    wandb_logger.experiment.config.update(config)

    # Checkpoint callback
    checkpoint_dir = Path(os.getenv("CHECKPOINT_DIR")) / group / trial.number
    checkpoint_callback = ModelCheckpoint(
        monitor="val-loss", dirpath=checkpoint_dir, save_last=True, save_top_k=1,
        every_n_epochs=1, filename=f'{group}-trial={trial.number}' + '-{epoch}-{val-loss:.2f}')

    # Optuna pruning callback prunes on number of epochs
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor='val-loss')

    nnodes = int(os.getenv("SLURM_NNODES"))
    trainer = Trainer(
        devices="auto",
        accelerator="auto",
        logger=wandb_logger,
        callbacks=[pruning_callback, checkpoint_callback],
        max_epochs=150,
        num_nodes=nnodes,
        log_every_n_steps=20,
        precision='bf16-mixed',
        gradient_clip_val=100.0,
        accumulate_grad_batches=3,
    )

    # tuner = Tuner(trainer)
    # tuner.scale_batch_size(model, datamodule=datamodule, init_val=4, steps_per_trial=200, mode="binsearch")

    try:
        trainer.fit(model, datamodule=datamodule)
    except Exception as e:
        logging.error(traceback.format_exc())
        trainer.callback_metrics["val-loss"] = torch.tensor(torch.inf)

    return trainer.callback_metrics["val-loss"].item()


def run_optimization(n_trials=5):
    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=3, reduction_factor=4)
    sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=20)

    study = optuna.create_study(
        direction='minimize',
        pruner=pruner, sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Number: {trial.number}")
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return study


study = run_optimization(n_trials=200)
