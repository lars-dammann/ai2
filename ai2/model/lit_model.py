from model.unet import UNet

import lightning as pl
from torchmetrics.regression import R2Score
import torch
import torch.nn as nn


class CorrosionUNet(pl.LightningModule):
    def __init__(self, model_config):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet(model_config, in_channels=4, out_channels=1)
        self.loss_fn = nn.functional.mse_loss
        self.learning_rate = model_config["lr"]
        self.weight_decay = model_config["weight_decay"]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # x: (B,3,H,W) Imgage + (B,1,H,W) Height profile, y: (B,1,H,W) Height profile
        x, y_target = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y_target)
        self.log('train_loss', loss)
        r2score = R2Score()
        self.log('train_r2_loss', r2score(torch.flatten(y_pred), torch.flatten(y_target)))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_target = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y_target)
        self.log('val_loss', loss)
        r2score = R2Score()
        self.log('val_r2_loss', r2score(torch.flatten(y_pred), torch.flatten(y_target)))
        return loss

    def test_step(self, batch, batch_idx):
        x, y_target = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y_target)
        self.log('test_loss', loss)
        r2score = R2Score()
        self.log('test_r2_loss', r2score(torch.flatten(y_pred), torch.flatten(y_target)))
        return loss

    def predict_step(self, batch, batch_idx):
        """
        Make a prediction for every
        """
        x, sample_info = batch
        y_pred = self(x)

        return self._reconstruct_height_profiles(y_pred, sample_info)

    def _reconstruct_height_profiles(self, y_pred, sample_info):

        # Get list of sample ids
        sample_ids = list(sample_info.keys())

        # Reconstruct every height profile from the patches
        predicted_height_profiles = []
        current_patch_number = 0
        for id in sample_ids:
            total_number_patches = sample_info[id]["total_number_patches"]
            predicted_height_profiles.append(
                self._reconstruct_single_height_profile(
                    data=y_pred
                    [current_patch_number: current_patch_number + total_number_patches],
                    patch_positions=sample_info[id]["positions"],
                    imageshape=sample_info[id]["imageshape"]))
            current_patch_number += total_number_patches

        return predicted_height_profiles, sample_ids

    def _reconstruct_single_height_profile(self, data, patch_positions, imageshape):
        reconstructed_profile = torch.full((1, *imageshape), torch.nan)
        patchshape = data.shape[-2:]
        # Loop over every patch and the correspondin position in the full image
        for index, pos in enumerate(patch_positions):
            # Get patch in reconstructed image
            reconstructed_patch = reconstructed_profile[:,
                                                        pos[0]:pos[0]+patchshape[0],
                                                        pos[1]:pos[1]+patchshape[1]]
            # Determine NaN values
            mask = torch.isnan(reconstructed_patch)
            # If NaN value, overwrite with patch
            reconstructed_patch[mask] = data[index][mask]
            # If not NaN value overwrite with mean
            reconstructed_patch[~mask] = torch.mean(torch.stack(
                (reconstructed_patch[~mask], data[index][~mask])), dim=0)
        return reconstructed_profile

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate, weight_decay=self.weight_decay)
