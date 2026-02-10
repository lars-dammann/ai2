from datamodule.datamodule import CorrosionDataModule
from model.lit_model import CorrosionUNet
from utils.get_data import get_config

from pathlib import Path

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import Callback
import wandb
import numpy as np

seed_everything(0, workers=True)


class SaveHeightPrediction(Callback):
    """
    Callback that saves every batch of the predictions to file
    """

    def __init__(self, save_path):
        super().__init__()
        self.save_path = Path(save_path)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        profiles, sample_ids = outputs
        self.save_path.mkdir(parents=True, exist_ok=True)
        for index, id in enumerate(sample_ids):
            np.save(self.save_path / f"{id}.npy", torch.squeeze(profiles[index]))


# Load model from checkpoint
run_id = "d2rltbw6"
model = f"model-{run_id}"
version = "best"
checkpoint_reference = f"lars-dammann-phd/ai2/{model}:{version}"

# Determine where to save the prediciton results to
predict_dataset = "test"
save_path = Path(
    __file__).parent.parent / f"results/predictions/{model}-{version}/{predict_dataset}/numpy"

run = wandb.init(project="ai2", id=run_id)

artifact = run.use_artifact(checkpoint_reference, type="model")
artifact_dir = artifact.download()

producing_run = artifact.logged_by()
config = producing_run.config

model = CorrosionUNet.load_from_checkpoint(
    Path(artifact_dir) / "model.ckpt", model_config=config["unet"], reconstruction_overlap=100)

datamodule = CorrosionDataModule(
    datamodule_config=config["datamodule"],
    reconstruction_overlap=100, predict_dataset=predict_dataset)

save_height_predictions = SaveHeightPrediction(save_path=save_path)
trainer = Trainer(callbacks=[save_height_predictions])

# tuner = Tuner(trainer)
# tuner.scale_batch_size(model, datamodule=datamodule, method='predict')

trainer.predict(model, datamodule=datamodule)
