from datamodule.datamodule import CorrosionDataModule
from model.lit_model import CorrosionUNet
from utils.get_config import get_config

from pathlib import Path

import torch.nn as nn
import lightning as pl
from lightning.pytorch.loggers import WandbLogger


def weights_init(model):
    if isinstance(model, nn.Conv2d):
        nn.init.kaiming_normal_(model.weight, mode='fan_out', nonlinearity='relu')
        if model.bias is not None:
            nn.init.constant_(model.bias, 0)
    elif isinstance(model, nn.BatchNorm2d):
        nn.init.constant_(model.weight, 1)
        nn.init.constant_(model.bias, 0)

wandb_logger = WandbLogger(project="ai2")
trainer = pl.Trainer(limit_train_batches=100, max_epochs=1, logger=wandb_logger)

config = get_config()

model = CorrosionUNet(config=config["unet"])
model.apply(weights_init)
datamodule = CorrosionDataModule(config=config["datamodule"])

trainer.fit(model, datamodule=datamodule)