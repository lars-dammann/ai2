from datamodule.datamodule import CorrosionDataModule
from model.lit_model import CorrosionUNet
from utils.get_data import get_config

import os

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import Callback
from pathlib import Path
import numpy as np
import wandb

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

# Load environment variables
data_dir = Path(os.getenv("AI2_DATA_DIR"))
prediciton_save_path = Path(os.getenv("AI2_PREDICTION_SAVE_PATH"))
config_file = Path(os.getenv("AI2_CONFIG_FILE"))
wandb_account = os.getenv("AI2_WANDB_ACCOUNT")
wandb_project = os.getenv("AI2_WANDB_PROJECT")
run_id = os.getenv("AI2_WANDB_RUN_ID")

predict_datasets = ["test", "val", "train"]
version = "best"

model_name = f"model-{run_id}"
model_dir = model_name + f"-{version}"

# Get the config of the producing run to be able to load the model and datamodule with the same config
api = wandb.Api()
config = api.run(f"{wandb_account}/{wandb_project}/{run_id}").config

# Get model artifact and download it to a local directory
artifact = api.artifact(f"{wandb_account}/{wandb_project}/model-{run_id}:{version}")
artifact_dir = artifact.download()

model = CorrosionUNet.load_from_checkpoint(
    Path(artifact_dir) / "model.ckpt", model_config=config["unet"], reconstruction_overlap=100)

for predict_dataset in predict_datasets:
    datamodule = CorrosionDataModule(data_dir=data_dir,
                                     datamodule_config=config["datamodule"],
                                     reconstruction_overlap=100, predict_dataset=predict_dataset)

    # Determine where to save the prediciton results to
    save_path = prediciton_save_path / f"{model_dir}/{predict_dataset}/numpy"

    save_height_predictions = SaveHeightPrediction(save_path=save_path)
    trainer = Trainer(callbacks=[save_height_predictions])

    # tuner = Tuner(trainer)
    # tuner.scale_batch_size(model, datamodule=datamodule, method='predict')

    trainer.predict(model, datamodule=datamodule)
