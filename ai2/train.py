from datamodule.datamodule import CorrosionDataModule
from model.lit_model import CorrosionUNet
from utils.get_data import get_config

import os
from pathlib import Path

import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner import Tuner

seed_everything(0, workers=True)


def weights_init(model):
    if isinstance(model, nn.Conv2d):
        nn.init.kaiming_normal_(model.weight, mode='fan_out', nonlinearity='relu')
        if model.bias is not None:
            nn.init.constant_(model.bias, 0)
    elif isinstance(model, nn.BatchNorm2d):
        nn.init.constant_(model.weight, 1)
        nn.init.constant_(model.bias, 0)


# Load config
config_file = Path(os.getenv("AI2_CONFIG_FILE"))
config = get_config(config_file)

model = CorrosionUNet(model_config=config["unet"])
model.apply(weights_init)

data_dir = Path(os.getenv("AI2_DATA_DIR"))
datamodule = CorrosionDataModule(data_dir=data_dir, datamodule_config=config["datamodule"])

# initialise the wandb logger and name your wandb project
name = "ResidualBaseline"
wandb_logger = WandbLogger(project="ai2", name=name, log_model=True)
wandb_logger.experiment.config.update(config)

# Checkpoint callback
checkpoint_dir = Path(os.getenv("AI2_CHECKPOINT_DIR"))  / name
checkpoint_callback = ModelCheckpoint(
    monitor="val-loss", dirpath=checkpoint_dir, save_last=True, save_top_k=1, every_n_epochs=1,
    filename='{epoch}-{val-loss:.2f}')

nnodes = int(os.getenv("SLURM_NNODES"))
trainer = Trainer(
    devices="auto",
    accelerator="auto",
    gradient_clip_val=1.0,
    logger=wandb_logger,
    callbacks=[checkpoint_callback],
    max_epochs=800,
    num_nodes=nnodes,
    log_every_n_steps=20,
    precision='bf16-mixed',
    accumulate_grad_batches=3
)

tuner = Tuner(trainer)
# tuner.scale_batch_size(model, datamodule=datamodule)

trainer.fit(model, datamodule=datamodule)
