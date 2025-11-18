from datamodule.datamodule import CorrosionDataModule
from model.lit_model import CorrosionUNet
from utils.get_data import get_config

from pathlib import Path
import os

import torch.nn as nn
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

seed_everything(0, workers=True)

def weights_init(model):
    if isinstance(model, nn.Conv2d):
        nn.init.kaiming_normal_(model.weight, mode='fan_out', nonlinearity='relu')
        if model.bias is not None:
            nn.init.constant_(model.bias, 0)
    elif isinstance(model, nn.BatchNorm2d):
        nn.init.constant_(model.weight, 1)
        nn.init.constant_(model.bias, 0)

# Load and unite configs
config = get_config({"batchsize": 2}, file="debug-configs.json")

model = CorrosionUNet(model_config=config["unet"])
model.apply(weights_init)

datamodule = CorrosionDataModule(datamodule_config=config["datamodule"])

# initialise the wandb logger and name your wandb project
wandb_logger = WandbLogger(project="ai2", group="Debug", log_model=True)
wandb_logger.experiment.config.update(config)

# Checkpoint callback
checkpoint_dir = Path(os.path.dirname(__file__)).parent
checkpoint_dir = checkpoint_dir / "checkpoints"
checkpoint_callback = ModelCheckpoint(monitor="val_loss", dirpath=checkpoint_dir, save_last=True, save_top_k=1, every_n_epochs=1, filename='{epoch}-{val_loss:.2f}')

# Early stopping callback
early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=20,
    verbose=False,
    mode='min'
)

# nnodes = int(os.getenv("SLURM_NNODES"))
trainer = Trainer(
    devices="auto",
    accelerator="auto",
    gradient_clip_val=1.0,
    logger=wandb_logger,
    callbacks=[early_stop_callback, checkpoint_callback],
    max_epochs=1,
    num_nodes=1,
    log_every_n_steps=1,
    precision='bf16-mixed'
    )

trainer.fit(model, datamodule=datamodule)