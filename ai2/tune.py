import os
from pathlib import Path

import optuna
from optuna.integration import PyTorchLightningPruningCallback
# from optuna.integration.wandb import WeightsAndBiasesCallback
import torch.nn as nn
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import wandb

from datamodule.datamodule import CorrosionDataModule
from model.lit_model import CorrosionUNet
from utils.get_data import get_config

seed_everything(0, workers=True)

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
    trial.suggest_int("udepth", 3, 6)
    trial.suggest_categorial("startfeature", [8, 16, 32, 64, 128])
    trial.suggest_categorial("batchsize", [8, 16, 32])
    trial.suggest_categorial("datasize", [256, 512, 1024])
    trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)

    # Load and unite configs
    config = get_config(trial.params)

    try:
        config["n_workers"] = int(os.getenv('SLURM_CPUS_PER_TASK'))
    except TypeError:
        pass

    model = CorrosionUNet(model_config=config["unet"])
    model.apply(weights_init)

    datamodule = CorrosionDataModule(datamodule_config=config["datamodule"])

    # initialise the wandb logger and name your wandb project
    wandb_logger = WandbLogger(project="ai2", group="", name=f"Trial {trial.number}", log_model=True)
    wandb_logger.experiment.config.update(config)

    # Checkpoint callback
    checkpoint_dir = Path(os.path.dirname(__file__)).parent
    checkpoint_dir = checkpoint_dir / "checkpoints"
    checkpoint_callback = ModelCheckpoint(monitor="val-mse-loss", dirpath=checkpoint_dir, save_last=True, save_top_k=1, every_n_epochs=1, filename='{epoch}-{val-mse-loss:.2f}')


    # Optuna pruning callback
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor='val-mse-loss')

    nnodes = int(os.getenv("SLURM_NNODES"))
    trainer = Trainer(
        devices="auto",
        accelerator="auto",
        logger=wandb_logger,
        callbacks=[pruning_callback, checkpoint_callback],
        max_epochs=200,
        num_nodes=nnodes,
        log_every_n_steps=20,
        precision='bf16-mixed',
        gradient_clip_val=1.0,
        )

    trainer.fit(model, datamodule=datamodule)
    return trainer.callback_metrics["val-mse-loss"].item()

def run_optimization(n_trials=5):
    pruner = optuna.pruners.HyperbandPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(direction='minimize', pruner=pruner)
    study.optimize(objective, n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return study

study = run_optimization(n_trials=5)