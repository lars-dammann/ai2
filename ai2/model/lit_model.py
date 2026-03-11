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

    def _calc_losses(self, y_pred, y_target, mask):
        """
        Calculate all losses and metrics for a batch
        """
        mae_loss = nn.functional.l1_loss(y_pred[~mask], y_target[~mask])
        mse_loss = nn.functional.mse_loss(y_pred[~mask], y_target[~mask])

        n_pixels = torch.sum((~mask), dim=(1, 2, 3))
        r2score = self._batch_r2score(y_pred[~mask], y_target[~mask], n_pixels)
        corr = self._batch_pearson_corr(y_pred[~mask], y_target[~mask], n_pixels)
        volume_loss_abs, volume_loss_sq = self._calc_volume_loss(y_pred[~mask], y_target[~mask], n_pixels)

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
        x, y_target, mask = batch
        y_pred = self(x)
        losses = self._calc_losses(y_pred, y_target, mask)
        losses["loss"] = losses["mae-loss"] + self.volume_error_weight * losses["volume-loss-sq"]
        self._log_loss(losses, prefix)
        return losses["loss"]

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._step(batch, 'test')

    def _batch_r2score(self, y_pred, y_target, lengths):
        """Calculate R2 score for every sample in the batch and return the mean R2 score over the batch"""
        res_sum_squared = self._segment_sum((y_target - y_pred) ** 2, lengths=lengths)
        total_sum_squared = self._segment_sum((y_target - torch.repeat_interleave(self._calc_segment_mean(y_target, lengths), lengths)) ** 2, lengths=lengths)
        return torch.mean(1 - res_sum_squared / total_sum_squared)

    def _batch_pearson_corr(self, y_pred, y_target, lengths):
        """
        Calculate Pearson correlation for every sample in the batch and return the mean correlation over the batch
        """
        norm = 1/(self._calc_segment_std(y_pred, lengths) * self._calc_segment_std(y_target, lengths) * (lengths - 1))
        y_pred_mean = self._calc_segment_mean(y_pred, lengths)
        y_target_mean = self._calc_segment_mean(y_target, lengths)
        return torch.mean(norm * self._segment_sum((y_pred - torch.repeat_interleave(y_pred_mean, lengths)) * (y_target - torch.repeat_interleave(y_target_mean, lengths)), lengths))

    def _calc_segment_std(self, data, lengths):
        """
        Calculate the standard deviation of the predicted height values for every sample in the batch
        """
        mean = torch.segment_reduce(data, reduce="mean", lengths=lengths)
        mean_of_squares = torch.segment_reduce(data**2, reduce="mean", lengths=lengths)
        return torch.sqrt(mean_of_squares - mean**2)

    def _segment_sum(self, data, lengths):
        """
        Calculate the sum of the predicted height values for every sample in the batch
        """
        return torch.segment_reduce(data, reduce="sum", lengths=lengths)

    def _calc_segment_mean(self, data, lengths):
        """
        Calculate the mean of the predicted height values for every sample in the batch
        """
        return torch.segment_reduce(data, reduce="mean", lengths=lengths)

    def _calc_volume_loss(self, y_pred, y_target, lengths):
        """
        Calculate the absolute and squared error of the predicted volume loss compared to the target volume loss
        """
        pred_sum = self._segment_sum(y_pred, lengths)
        target_sum = self._segment_sum(y_target, lengths)
        return torch.mean(torch.abs((pred_sum - target_sum) / target_sum)), torch.mean(torch.square((pred_sum - target_sum) / target_sum))

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
