from datamodule.datamodule import CorrosionDataModule
from model.lit_model import CorrosionUNet
from utils.get_data import get_config

from pathlib import Path

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback
import wandb
import numpy as np


class SaveHeightPrediction(Callback):
    """
    Callback that saves every batch of the predictions to file
    """

    def __init__(self, save_path):
        super().__init__()
        self.save_path = Path(save_path)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        profiles, sample_ids = outputs
        for index, id in enumerate(sample_ids):
            np.save(self.save_path / f"{id}.npy", torch.squeeze(profiles[index]))


save_path = Path(__file__).parent.parent / "results/predictions"
# Load model from checkpoint
checkpoint_reference = "lars-dammann-phd/ai2/model-5tg5u9kj:best"

run = wandb.init(project="ai2", group="predict")

artifact = run.use_artifact(checkpoint_reference, type="model")
artifact_dir = artifact.download()

config = get_config({"batchsize": 2})

model = CorrosionUNet.load_from_checkpoint(
    Path(artifact_dir) / "model.ckpt", model_config=config["unet"])
datamodule = CorrosionDataModule(datamodule_config=config["datamodule"])

save_height_prectiions = SaveHeightPrediction(save_path=save_path)
trainer = Trainer(callbacks=[save_height_prectiions])

trainer.predict(model, datamodule=datamodule)
