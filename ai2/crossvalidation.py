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
group = os.getenv("WANDB_GROUP")
complete_data_dir = Path(os.getenv("COMPLETE_DATA_DIR"))
split_data_dir = Path(os.getenv("SPLIT_DATA_DIR"))
inhibitor_list_file = Path(os.getenv("INHIBITOR_LIST_FILE"))
nnodes = int(os.getenv("SLURM_NNODES"))
# Load config
config_file = Path(os.getenv("CONFIG_FILE"))
config = get_config(config_file)

splitter = CrossValidationDataSplitter(
    target_dir=split_data_dir, source_dir=complete_data_dir,
    inhibitor_list_file=inhibitor_list_file, val_size=10, test_size=10, random_seed=1)
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
    checkpoint_dir = Path(os.getenv("CHECKPOINT_DIR")) / group
    checkpoint_callback = ModelCheckpoint(
        monitor="val-loss", dirpath=checkpoint_dir, save_last=True, save_top_k=1, every_n_epochs=1,
        filename=f'{group}-{name}-' + '{epoch}-{val-loss:.2f}')

    # initialise the wandb logger and name your wandb project
    wandb_logger = WandbLogger(
        project=os.getenv("WANDB_PROJECT"),
        group=group, name=name, log_model=True, config=updated_config)

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

    model = CorrosionUNet(model_config=updated_config["unet"])
    model.apply(model.weights_init)

    datamodule = CorrosionDataModule(
        data_dir=data_dir, datamodule_config=updated_config["datamodule"])

    trainer.fit(model, datamodule=datamodule)
    wandb.finish()
