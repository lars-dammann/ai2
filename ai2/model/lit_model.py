from model.unet import UNet

import lightning as pl
from torchmetrics.regression import R2Score
import torch
import torch.nn as nn


class CorrosionUNet(pl.LightningModule):
    def __init__(self, model_config, reconstruction_overlap=0):
        super().__init__()
        self.save_hyperparameters(ignore=["reconstruction_overlap"])
        self.model = UNet(model_config, in_channels=4, out_channels=1)
        self.learning_rate = model_config["lr"]
        self.weight_decay = model_config["weight_decay"]
        self.reconstruction_overlap = reconstruction_overlap

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
        self.log('val-corr', self._batch_pearson_corr(y_pred, y_target))
        return loss

    def test_step(self, batch, batch_idx):
        x, y_target = batch
        y_pred = self(x)
        loss = nn.functional.mse_loss(y_pred, y_target)
        self.log('test-mse-loss', loss)
        self.log('test-r2score', self._batch_r2score(y_pred, y_target))
        self.log('test-corr', self._batch_pearson_corr(y_pred, y_target))
        return loss

    def _batch_r2score(self, y_pred, y_target):
        r2score = R2Score(multioutput="raw_values")
        return torch.mean(r2score(y_pred.view(y_pred.shape[0], -1).t(), y_target.view(y_target.shape[0], -1).t()))

    def _batch_pearson_corr(self, y_pred, y_target):
        reshaped_y_pred = y_pred.view(y_pred.shape[0], -1)
        reshaped_y_target = y_target.view(y_target.shape[0], -1)
        norm = 1/(torch.std(reshaped_y_pred, dim=1)
                  * torch.std(reshaped_y_target, dim=1) * (reshaped_y_pred.shape[1] - 1))
        y_pred_mean = torch.mean(reshaped_y_pred, dim=1, keepdim=True)
        y_target_mean = torch.mean(reshaped_y_pred, dim=1, keepdim=True)
        return torch.mean(norm * torch.sum(
            (reshaped_y_pred - y_pred_mean) *
            (reshaped_y_target - y_target_mean),
            dim=1))

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

        # Determine if patch is on the first or last position
        flattened_pos = [i for ii in patch_positions for i in ii]
        max_pos = max(flattened_pos)
        min_pos = min(flattened_pos)

        # Loop over every patch and the corresponding position in the full image
        for index, pos in enumerate(patch_positions):
            # Determine how much the borders of the patches have to be cropped
            x_start_crop, x_end_crop = self._determine_crop(
                pos=pos[0],
                min_pos=min_pos, max_pos=max_pos)
            y_start_crop, y_end_crop = self._determine_crop(
                pos=pos[1],
                min_pos=min_pos, max_pos=max_pos)

            # Get patch in reconstructed image
            reconstructed_patch = reconstructed_profile[:,
                                                        pos[0] + x_start_crop:pos[0] + patchshape[0] - x_end_crop,
                                                        pos[1] + y_start_crop:pos[1] + patchshape[1] - y_end_crop]
            cropped_data = data[index, :, x_start_crop: patchshape[0] - x_end_crop,
                                y_start_crop: patchshape[1] - y_end_crop]
            # Determine NaN values
            mask = torch.isnan(reconstructed_patch)
            # If NaN value, overwrite with patch
            reconstructed_patch[mask] = cropped_data[mask]
            # If not NaN value overwrite with mean
            reconstructed_patch[~mask] = torch.mean(torch.stack(
                (reconstructed_patch[~mask], cropped_data[~mask])), dim=0)
        return reconstructed_profile

    def _determine_crop(self, pos, min_pos, max_pos):
        """Determine if the patch has to be cropped at the borders"""
        start_crop = self.reconstruction_overlap
        end_crop = self.reconstruction_overlap

        if pos == min_pos:
            start_crop = 0
        if pos == max_pos:
            end_crop = 0

        return start_crop, end_crop

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate, weight_decay=self.weight_decay)
