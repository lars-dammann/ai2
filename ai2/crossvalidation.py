from datamodule.datamodule import CorrosionDataModule
from model.lit_model import CorrosionUNet
from utils.get_data import get_config
from utils.data_splitter import CrossValidationDataSplitter
from utils.normalizer import Normalizer

import os
from pathlib import Path

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
import wandb

seed_everything(0, workers=True)
# Base directory for the model checkpoints
checkpoints = Path(os.getenv("CHECKPOINT_DIR"))
# Directory containing the complete dataset to be split into folds
complete_data_dir = Path(os.getenv("COMPLETE_DATA_DIR"))
# Directory where the split data will be stored
split_data_dir = Path(os.getenv("SPLIT_DATA_DIR"))
# Wandb project name
project = os.getenv("WANDB_PROJECT")
# Wandb group name
group = os.getenv("WANDB_GROUP")
# File containing the list of modulators to be used for splitting the data into folds
modulator_list_file = Path(os.getenv("MODULATOR_LIST_FILE"))
# Number of nodes to be used for training, obtained from the SLURM environment variable
nnodes = int(os.getenv("SLURM_NNODES"))
# Config file path for the model and datamodule
config_file = Path(os.getenv("CONFIG_FILE"))

# Load the configuration from the specified config file
config = get_config(config_file)

# Create a CrossValidationDataSplitter instance to handle the data splitting into folds
splitter = CrossValidationDataSplitter(
    target_dir=split_data_dir, source_dir=complete_data_dir,
    modulator_list_file=modulator_list_file, val_size=10, test_size=10, random_seed=1)
splitter.materialize_fold(clean_target_dir=True)
splitter.validate_splits()

for fold_index in range(splitter.num_folds):
    name = f"Fold-{fold_index}"
    data_dir = split_data_dir / f"{fold_index}"

    # Calculate the normaliztion values from the training data and add them to the config
    normalization_values = Normalizer.calculate_normalization_stats(data_dir)
    updated_config = config.copy()
    updated_config["datamodule"]["normalization"] = normalization_values

    # Checkpoint callback
    checkpoint_dir = checkpoints / group
    checkpoint_callback = ModelCheckpoint(
        monitor="val-loss", dirpath=checkpoint_dir, save_last=True, save_top_k=1, every_n_epochs=1,
        filename=f'{group}-{name}-' + '{epoch}-{val-loss:.2f}')

    # Initialise the wandb logger
    wandb_logger = WandbLogger(
        project=project,
        group=group, name=name, log_model=True, config=updated_config)

    # Initialize the PyTorch Lightning Trainer with the specified configuration and callbacks
    trainer = Trainer(
        devices="auto",
        accelerator="auto",
        gradient_clip_val=updated_config["trainer"]["gradient_clip_val"],
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        max_epochs=updated_config["trainer"]["max_epochs"],
        num_nodes=nnodes,
        log_every_n_steps=20,
        precision=updated_config["trainer"]["precision"],
        accumulate_grad_batches=updated_config["trainer"]["accumulate_grad_batches"]
    )

    # Initialize the model with the configuration and apply weight initialization
    model = CorrosionUNet(model_config=updated_config["unet"])
    model.apply(model.weights_init)

    # Initialize the datamodule with the data directory and configuration
    datamodule = CorrosionDataModule(
        data_dir=data_dir, datamodule_config=updated_config["datamodule"])

    # Train the model using the trainer and datamodule
    trainer.fit(model, datamodule=datamodule)
    # Explicitly finish the wandb run to ensure that the next fold is logged as a separate run
    wandb.finish()
