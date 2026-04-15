from datamodule.datamodule import CorrosionDataModule
from model.lit_model import CorrosionUNet
from utils.get_data import get_config

from pathlib import Path
import os

import torch.nn as nn
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import grad_norm

seed_everything(0, workers=True)


class LogGradNormCallback(Callback):
    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        norm_order = 2.0
        norms = grad_norm(pl_module, norm_type=norm_order)
        pl_module.log(
            'grad_norm', norms[f'grad_{norm_order}_norm_total'],
            on_step=True, on_epoch=False)


base_path = Path(__file__).parent.parent

# Load and unite configs
config_file = base_path / "configs" / "debug-configs.json"
config = get_config(config_file)

model = CorrosionUNet(model_config=config["unet"])
model.apply(model.weights_init)

data_dir = base_path / "data"
datamodule = CorrosionDataModule(data_dir=data_dir, datamodule_config=config["datamodule"])

# initialise the wandb logger and name your wandb project
wandb_logger = WandbLogger(project="ai2", group="Debug", log_model=False, offline=True)
wandb_logger.experiment.config.update(config)

# Checkpoint callback
checkpoint_dir = Path(os.path.dirname(__file__)).parent / "checkpoints" / wandb_logger.experiment.id
checkpoint_callback = ModelCheckpoint(
    monitor="val-loss", dirpath=checkpoint_dir, save_last=True, save_top_k=1, every_n_epochs=1,
    filename='{epoch}-{val-loss:.2f}')

trainer = Trainer(
    devices="auto",
    accelerator="auto",
    gradient_clip_val=1.0,
    logger=wandb_logger,
    callbacks=[checkpoint_callback, LogGradNormCallback()],
    max_epochs=10,
    num_nodes=1,
    log_every_n_steps=1,
    precision='bf16-mixed',
    num_sanity_val_steps=0,
    # max_steps=2
    # limit_train_batches=1,
    # limit_val_batches=1,
)

trainer.fit(model, datamodule=datamodule)
