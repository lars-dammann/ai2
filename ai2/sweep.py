from datamodule.datamodule import CorrosionDataModule
from model.lit_model import CorrosionUNet
from utils.get_data import get_config

from pathlib import Path
import os

import torch.nn as nn
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import wandb

seed_everything(0, workers=True)

group = "NormVolLoss"


def weights_init(model):
    if isinstance(model, nn.Conv2d):
        nn.init.kaiming_normal_(model.weight, mode='fan_out', nonlinearity='relu')
        if model.bias is not None:
            nn.init.constant_(model.bias, 0)
    elif isinstance(model, nn.BatchNorm2d):
        nn.init.constant_(model.weight, 1)
        nn.init.constant_(model.bias, 0)


for volume_error_weight in [0.0001, 0.0005]:
    name = f"Weight{volume_error_weight}"

    # Load and unite configs
    config = get_config({"volume_error_weight": volume_error_weight})

    model = CorrosionUNet(model_config=config["unet"])
    model.apply(weights_init)

    datamodule = CorrosionDataModule(datamodule_config=config["datamodule"])

    # initialise the wandb logger and name your wandb project
    wandb_logger = WandbLogger(project="ai2", group=group, name=name, log_model=True)
    wandb_logger.experiment.config.update(config)

    # Checkpoint callback
    checkpoint_dir = Path(os.getenv("AI2_CHECKPOINT_DIR")) / group / name
    checkpoint_callback = ModelCheckpoint(
        monitor="val-loss", dirpath=checkpoint_dir, save_last=True, save_top_k=1, every_n_epochs=1,
        filename=f'{group}-{name}-' + '{epoch}-{val-loss:.2f}')

    nnodes = int(os.getenv("SLURM_NNODES"))
    trainer = Trainer(
        devices="auto",
        accelerator="auto",
        gradient_clip_val=100.0,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        max_epochs=30,
        num_nodes=nnodes,
        log_every_n_steps=20,
        precision='bf16-mixed',
        accumulate_grad_batches=3,
    )

    trainer.fit(model, datamodule=datamodule)
    wandb.finish()
