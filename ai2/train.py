from datamodule.datamodule import CorrosionDataModule
from model.lit_model import CorrosionUNet
from utils.get_config import get_config

from pathlib import Path

import lightning as pl
from lightning.pytorch.loggers import WandbLogger


wandb_logger = WandbLogger(project="ai2")
trainer = pl.Trainer(limit_train_batches=100, max_epochs=1, logger=wandb_logger)

config = get_config()

model = CorrosionUNet(config=config["unet"])
datamodule = CorrosionDataModule(config=config["datamodule"])

trainer.fit(model, datamodule=datamodule)