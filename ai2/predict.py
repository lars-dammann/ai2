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


model_ids = {
    "ypm9dmrj": 0, "noejb2ky": 1, "704xp5bj": 2, "xaza9yzp": 3, "7pl2ib66": 4, "0nxx3mqq": 5,
    "5id3tm5i": 6, "nrz3vy9w": 7, "pbi4a0jf": 8, "hly0w115": 9, "kplvubkg": 10, "8do9ubhb": 11,
    "1ph4ulgv": 12, "rjnirtm6": 13, "dhvcfhj8": 14, "n9mp487q": 15, "ueekyv1u": 16, "ti8o2i3n": 17}


# Load environment variables
parent_data_dir = Path(os.getenv("DATA_DIR"))
prediciton_save_path = Path(os.getenv("PREDICTION_SAVE_PATH"))
config_file = Path(os.getenv("CONFIG_FILE"))
wandb_account = os.getenv("WANDB_ACCOUNT")
wandb_project = os.getenv("WANDB_PROJECT")
version = os.getenv("MODEL_VERSION")

for run_id, fold in model_ids.items():
    data_dir = parent_data_dir / f"{fold}"

    predict_datasets = ["test", "val", "train"]
    # version = "last"

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
        print(f"Predicting height profiles of fold {fold} on {predict_dataset} dataset...")
        datamodule = CorrosionDataModule(
            data_dir=data_dir, datamodule_config=config["datamodule"],
            reconstruction_overlap=100, predict_dataset=predict_dataset)

        # Determine where to save the prediciton results to
        save_path = prediciton_save_path / f"{model_dir}/{predict_dataset}"

        save_height_predictions = SaveHeightPrediction(save_path=save_path)
        trainer = Trainer(callbacks=[save_height_predictions])

        # tuner = Tuner(trainer)
        # tuner.scale_batch_size(model, datamodule=datamodule, method='predict')

        trainer.predict(model, datamodule=datamodule)
