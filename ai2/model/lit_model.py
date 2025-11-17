from model.unet import UNet

import lightning as pl
from torchmetrics.regression import R2Score, PearsonCorrCoef
import torch
import torch.nn as nn


class CorrosionUNet(pl.LightningModule):
    def __init__(self, model_config):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet(model_config, in_channels=4, out_channels=1)
        self.learning_rate = model_config["lr"]
        self.weight_decay = model_config["weight_decay"]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # x: (B,3,H,W) Imgage + (B,1,H,W) Height profile, y: (B,1,H,W) Height profile
        x, y_target = batch
        y_pred = self(x)
        loss = nn.functional.mse_loss(y_pred, y_target)
        self.log('train-mse-loss', loss)
        self.log('train-r2score', self._batch_r2score(y_pred, y_target))
        self.log('train-corr', self._batch_pearson_corr(y_pred, y_target))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_target = batch
        y_pred = self(x)
        loss = nn.functional.mse_loss(y_pred, y_target)
        self.log('val-mse-loss', loss)
        self.log('val-r2score', self._batch_r2score(y_pred, y_target))
        self.log('train-corr', self._batch_pearson_corr(y_pred, y_target))
        return loss

    def test_step(self, batch, batch_idx):
        x, y_target = batch
        y_pred = self(x)
        loss = nn.functional.mse_loss(y_pred, y_target)
        self.log('test-mse-loss', loss)
        self.log('test-r2score', self._batch_r2score(y_pred, y_target))
        self.log('train-corr', self._batch_pearson_corr(y_pred, y_target))
        return loss

    def _batch_r2score(self, y_pred, y_target):
        r2score = R2Score(multioutput="raw_values")
        return torch.mean(r2score(y_pred.view(y_pred.shape[0], -1).t(), y_target.view(y_target.shape[0], -1).t()))

    def _batch_pearson_corr(self, y_pred, y_target):
        pearson = PearsonCorrCoef(num_outputs=y_pred.shape[0])
        return torch.mean(pearson(y_pred.view(y_pred.shape[0], -1).t(), y_target.view(y_target.shape[0], -1).t()))

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
