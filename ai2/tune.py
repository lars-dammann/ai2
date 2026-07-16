import os
from pathlib import Path
import logging
import traceback
from functools import partial

import optuna
from optuna.integration import PyTorchLightningPruningCallback
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import wandb

from datamodule.datamodule import CorrosionDataModule
from model.lit_model import CorrosionUNet
from utils.normalizer import Normalizer
from utils.get_data import get_config, update_config

seed_everything(0, workers=True)


def objective(trial, base_config):
    # Finish any previous wandb run before starting a new one
    wandb.finish()
    # Suggest hyperparameters
    trial.suggest_int("udepth", 3, 6)
    trial.suggest_categorical("startfeature", [16, 32, 64])
    trial.suggest_categorical("datasize", [1024])
    trial.suggest_float('lr', 1e-6, 1e-1, log=True)
    trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)

    # Update config file with trial parameters
    config = base_config.copy()
    config = update_config(config, trial.params)

    # Initialize model and datamodule
    model = CorrosionUNet(model_config=config["unet"])
    model.apply(model.weights_init)

    data_dir = Path(os.getenv("DATA_DIR"))
    datamodule = CorrosionDataModule(data_dir=data_dir, datamodule_config=config["datamodule"])

    # Initialise the wandb logger
    group = os.getenv("WANDB_GROUP")
    wandb_logger = WandbLogger(project=os.getenv("WANDB_PROJECT"), group=group,
                               name=f"{group} Trial {trial.number}", log_model=True)
    wandb_logger.experiment.config.update(config)

    # Checkpoint callback
    checkpoint_dir = Path(os.getenv("CHECKPOINT_DIR")) / group / str(trial.number)
    checkpoint_callback = ModelCheckpoint(
        monitor="val-loss", dirpath=checkpoint_dir, save_last=True, save_top_k=1,
        every_n_epochs=1, filename=f'{group}-trial={trial.number}' + '-{epoch}-{val-loss:.2f}')

    # Optuna pruning callback prunes on number of epochs
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor='val-loss')

    # Initialize and return trainer
    nnodes = int(os.getenv("SLURM_NNODES"))
    trainer = Trainer(
        devices="auto",
        accelerator="auto",
        logger=wandb_logger,
        callbacks=[pruning_callback, checkpoint_callback],
        max_epochs=config["trainer"]["max_epochs"],
        num_nodes=nnodes,
        log_every_n_steps=config["trainer"]["log_every_n_steps"],
        precision='bf16-mixed',
        gradient_clip_val=config["trainer"]["gradient_clip_val"],
        accumulate_grad_batches=config["trainer"]["accumulate_grad_batches"],
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
    # Define the pruner and sampler for Optuna
    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=3, reduction_factor=4)
    sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=20)

    # Create an Optuna study and optimize the objective function
    study = optuna.create_study(
        direction='minimize',
        pruner=pruner, sampler=sampler)

    # Load base config file
    config_file = Path(os.getenv("CONFIG_FILE"))
    base_config = get_config(config_file)

    # Calculate the normaliztion values from the training data and add them to the base config
    data_dir = Path(os.getenv("DATA_DIR"))
    normalization_values = Normalizer.calculate_normalization_stats(data_dir)
    base_config["datamodule"]["normalization"] = normalization_values

    study.optimize(partial(objective, base_config=base_config), n_trials=n_trials)

    # Print the best trial results
    print("Best trial:")
    trial = study.best_trial
    print(f"  Number: {trial.number}")
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return study


# Run the Optuna optimization
study = run_optimization(n_trials=200)
