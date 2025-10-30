from datamodule.datamodule import CorrosionDataModule
from model.lit_model import CorrosionUNet

import json
from pathlib import Path

import lightning as pl
from lightning.pytorch.loggers import WandbLogger


wandb_logger = WandbLogger(project="ai2")
trainer = pl.Trainer(limit_train_batches=100, max_epochs=1, logger=wandb_logger)

model = CorrosionUNet()

with open(Path(__file__).parent.parent / "configs/configs.json", "r", encoding="utf-8") as c:
    config = json.load(c)
datamodule = CorrosionDataModule(config=config)

trainer.fit(model, datamodule=datamodule)