from datamodule.datamodule import CorrosionDataModule
from model.lit_model import CorrosionUNet

import os
import json

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback
from pathlib import Path
import numpy as np

seed_everything(0, workers=True)


class SaveHeightPrediction(Callback):
    """Save predicted height profiles to disk."""

    def __init__(self, save_path):
        super().__init__()
        self.save_path = Path(save_path)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        profiles, sample_ids = outputs
        self.save_path.mkdir(parents=True, exist_ok=True)
        for index, sample_id in enumerate(sample_ids):
            np.save(self.save_path / f"{sample_id}.npy", torch.squeeze(profiles[index]))


# Determine the datasets predicitons should be generated for
predict_datasets = ["test", "val", "train"]

# Map the model ids to the data fold numbers. This is used to load the correct model for each fold and predict on the specified datasets.
model_ids = {
    "ypm9dmrj": 0, "noejb2ky": 1, "704xp5bj": 2, "xaza9yzp": 3, "7pl2ib66": 4, "0nxx3mqq": 5,
    "5id3tm5i": 6, "nrz3vy9w": 7, "pbi4a0jf": 8, "hly0w115": 9, "kplvubkg": 10, "8do9ubhb": 11,
    "1ph4ulgv": 12, "rjnirtm6": 13, "dhvcfhj8": 14, "n9mp487q": 15, "ueekyv1u": 16, "ti8o2i3n": 17}


# Read environment variables
# Parent directory containing the data folds
parent_data_dir = Path(os.getenv("DATA_DIR"))
# Directory where the predictions will be saved
prediction_save_dir = Path(os.getenv("PREDICTION_DIR"))
# Directory containing the configuration files
config_dir = Path(os.getenv("CONFIG_DIR"))
# Directory containing the trained models
model_dir = Path(os.getenv("MODEL_DIR"))

for run_id, fold in model_ids.items():
    data_dir = parent_data_dir / f"{fold}"

    # Load model config
    with open(config_dir / f"config-{run_id}.json", "r") as f:
        config = json.load(f)

    # Load the model
    model_name = f"model-{run_id}-best"
    model = CorrosionUNet.load_from_checkpoint(
        Path(model_dir) / f"{model_name}.ckpt", model_config=config["unet"],
        reconstruction_overlap=100)

    # For each model, predict the specified datasets
    for predict_dataset in predict_datasets:
        print(f"Predicting height profiles of fold {fold} on {predict_dataset} dataset...")

        # Create the datamodule for the current fold and dataset
        datamodule = CorrosionDataModule(
            data_dir=data_dir, datamodule_config=config["datamodule"],
            reconstruction_overlap=100, predict_dataset=predict_dataset)

        # Determine where to save the prediction results to
        save_path = prediction_save_dir / f"{model_name} / {predict_dataset}"

        # Create a callback to save the predictions and run the prediction
        save_predictions = SaveHeightPrediction(save_path=save_path)
        trainer = Trainer(callbacks=[save_predictions])
        trainer.predict(model, datamodule=datamodule)
