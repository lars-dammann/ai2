from utils.nomalizer import Normalizer
from model.unet import UNet

import lightning as pl
from torchmetrics.regression import R2Score
import torch
import torch.nn as nn
import wandb


class CorrosionUNet(pl.LightningModule):
    def __init__(self, model_config, reconstruction_overlap=0):
        super().__init__()
        self.save_hyperparameters(ignore=["reconstruction_overlap"])
        self.model = UNet(model_config, in_channels=4, out_channels=1)
        self.learning_rate = model_config["lr"]
        self.weight_decay = model_config["weight_decay"]
        self.volume_error_weight = model_config["volume_error_weight"]
        self.reconstruction_overlap = reconstruction_overlap
        self.train_summary = False
        self.val_summary = False
        self.test_summary = False

    def forward(self, x):
        return self.model(x)

    def _log_loss(self, loss_dict, prefix):
        """
        Log the losses for a given prefix (train, val or test)
        """
        for key, value in loss_dict.items():
            self.log(prefix + '-' + key, value)
            # self.log('best-' + prefix + '-' + key, value)

    def _calc_losses(self, y_pred, y_target):
        """
        Calculate all losses and metrics for a batch
        """
        mae_loss = nn.functional.l1_loss(y_pred, y_target)
        mse_loss = nn.functional.mse_loss(y_pred, y_target)
        r2score = self._batch_r2score(y_pred, y_target)
        corr = self._batch_pearson_corr(y_pred, y_target)
        target_sum = torch.sum(y_target, dim=(1, 2, 3))
        pred_sum = torch.sum(y_pred, dim=(1, 2, 3))
        volume_loss_abs = torch.mean(torch.abs(pred_sum - target_sum) / torch.abs(target_sum))
        volume_loss_sq = torch.mean(torch.square((pred_sum - target_sum) / torch.abs(target_sum)))
        return {"mae-loss": mae_loss, "mse-loss": mse_loss, "r2score": r2score,
                "corr": corr, "volume-loss-abs": volume_loss_abs,
                "volume-loss-sq": volume_loss_sq}

    def on_fit_start(self):
        """
        Define the metrics to log the best values of the losses and scores
        """
        wandb_run = self.logger.experiment

        scores = ["loss", "mae-loss", "mse-loss", "r2score",
                  "corr", "volume-loss-abs", "volume-loss-sq"]
        for score in scores:
            for prefix in ["train", "val", "test"]:
                summary = "min" if "loss" in score else "max"
                # wandb_run.define_metric('best-' + prefix + '-' + score, summary=summary)
                wandb_run.define_metric(prefix + '-' + score, summary=summary)

    def _step(self, batch, prefix):
        x, y_target = batch
        y_pred = self(x)
        losses = self._calc_losses(y_pred, y_target)
        losses["loss"] = losses["mae-loss"] + self.volume_error_weight * losses["volume-loss-sq"]
        self._log_loss(losses, prefix)
        return losses["loss"]

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._step(batch, 'test')

    def _batch_r2score(self, y_pred, y_target):
        """Calculate R2 score for every sample in the batch and return the mean R2 score over the batch"""
        r2score = R2Score(multioutput="raw_values")
        return torch.mean(r2score(y_pred.view(y_pred.shape[0], -1).t(), y_target.view(y_target.shape[0], -1).t()))

    def _batch_pearson_corr(self, y_pred, y_target):
        """
        Calculate Pearson correlation for every sample in the batch and return the mean correlation over the batch
        """
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
        Make a prediction for every sample in the batch
        """
        x, sample_info = batch
        y_pred = self(x)

        return self._reconstruct_height_profiles(y_pred, sample_info)

    def _reconstruct_height_profiles(self, y_pred, sample_info):
        """
        Reconstructs the height profiles from the predicted patches and denormalizes them
        """
        # Get the normalization values to denormalize after reconstruction
        normalization = sample_info.pop("normalization")
        normalizer = Normalizer(
            torch.tensor(normalization["mean"]),
            torch.tensor(normalization["std"]))

        # Get list of sample ids
        sample_ids = list(sample_info.keys())

        # Reconstruct every height profile from the patches
        predicted_height_profiles = []
        current_patch_number = 0
        for id in sample_ids:
            total_number_patches = sample_info[id]["total_number_patches"]

            reconstructed_height_profile = self._reconstruct_single_height_profile(
                data=y_pred
                [current_patch_number: current_patch_number + total_number_patches],
                patch_positions=sample_info[id]["positions"],
                imageshape=sample_info[id]["imageshape"])

            predicted_height_profiles.append(normalizer.denormalize(reconstructed_height_profile))
            current_patch_number += total_number_patches

        return predicted_height_profiles, sample_ids

    def _reconstruct_single_height_profile(self, data, patch_positions, imageshape):
        """
        Reconstruct a single height profile from its patches
        """
        reconstructed_profile = torch.full((1, *imageshape), torch.nan)
        patchshape = data.shape[-2:]

        # Determine if patch is on the first or last position
        max_pos = torch.tensor(patch_positions).max(dim=0).values
        min_pos = torch.tensor(patch_positions).min(dim=0).values

        # Loop over every patch and the corresponding position in the full image
        for index, pos in enumerate(patch_positions):
            # Determine how much the borders of the patches have to be cropped
            x_start_crop, x_end_crop = self._determine_crop(
                pos=pos[0],
                min_pos=min_pos[0], max_pos=max_pos[0])
            y_start_crop, y_end_crop = self._determine_crop(
                pos=pos[1],
                min_pos=min_pos[1], max_pos=max_pos[1])

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
        """
        Determine if the patch has to be cropped at the borders
        """
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
