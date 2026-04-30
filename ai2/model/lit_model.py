from utils.normalizer import Normalizer
from model.unet import UNet

import lightning as pl
import torch
import torch.nn as nn
import wandb


class CorrosionUNet(pl.LightningModule):
    """Lightning module that wraps the UNet and computes training metrics.

    Args:
        model_config: UNet and optimizer configuration dictionary.
        reconstruction_overlap: Overlap in pixels used for patch reconstruction.
    """

    def __init__(self, model_config, reconstruction_overlap=0):
        super().__init__()
        self.save_hyperparameters(ignore=["reconstruction_overlap"])
        self.model = UNet(model_config, in_channels=4, out_channels=1)
        self.learning_rate = model_config["lr"]
        self.weight_decay = model_config["weight_decay"]
        self.reconstruction_overlap = reconstruction_overlap
        self.train_summary = False
        self.val_summary = False
        self.test_summary = False

    def forward(self, x):
        """Run a forward pass through the underlying UNet.

        Args:
            x: Input tensor with shape (B, 4, H, W).

        Returns:
            Predicted height tensor with shape (B, 1, H, W).
        """
        return self.model(x)

    def _log_loss(self, loss_dict, prefix):
        """Log all losses for the given stage prefix.

        Args:
            loss_dict: Mapping of metric names to tensors.
            prefix: Stage prefix such as train, val, or test.

        Returns:
            None.
        """
        for key, value in loss_dict.items():
            self.log(f"{prefix}-{key}", value)
            # self.log('best-' + prefix + '-' + key, value)

    def _calc_losses(self, y_pred, y_target, mask):
        """Calculate losses and metrics for a masked batch.

        Args:
            y_pred: Predicted tensor.
            y_target: Target tensor.
            mask: Boolean mask where True marks ignored values.

        Returns:
            Dictionary containing scalar loss and metric tensors.
        """
        mae_loss = nn.functional.l1_loss(y_pred[~mask], y_target[~mask])
        mse_loss = nn.functional.mse_loss(y_pred[~mask], y_target[~mask])

        n_pixels = torch.sum((~mask), dim=(1, 2, 3))
        r2score = self._batch_r2score(y_pred[~mask], y_target[~mask], n_pixels)
        corr = self._batch_pearson_corr(y_pred[~mask], y_target[~mask], n_pixels)
        volume_loss_abs = self._calc_volume_loss(
            y_pred[~mask], y_target[~mask], n_pixels
        )

        return {
            "mae-loss": mae_loss,
            "mse-loss": mse_loss,
            "r2score": r2score,
            "corr": corr,
            "volume-loss-abs": volume_loss_abs,
        }

    def on_fit_start(self):
        """Configure W&B summaries for the tracked metrics.

        Returns:
            None.
        """
        wandb_run = self.logger.experiment

        scores = ["loss", "mae-loss", "mse-loss", "r2score", "corr", "volume-loss-abs"]
        for score in scores:
            for prefix in ["train", "val", "test"]:
                summary = "min" if "loss" in score else "max"
                # wandb_run.define_metric('best-' + prefix + '-' + score, summary=summary)
                wandb_run.define_metric(f"{prefix}-{score}", summary=summary)

    def _step(self, batch, prefix):
        """Run a shared train/val/test step.

        Args:
            batch: Tuple of inputs, targets, and mask.
            prefix: Stage prefix such as train, val, or test.

        Returns:
            Main optimization loss tensor.
        """
        x, y_target, mask = batch
        y_pred = self(x)
        losses = self._calc_losses(y_pred, y_target, mask)
        losses["loss"] = losses["mae-loss"]
        self._log_loss(losses, prefix)
        return losses["loss"]

    def training_step(self, batch, batch_idx):
        """Run one training step.

        Args:
            batch: Training batch.
            batch_idx: Batch index.

        Returns:
            Training loss tensor.
        """
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        """Run one validation step.

        Args:
            batch: Validation batch.
            batch_idx: Batch index.

        Returns:
            Validation loss tensor.
        """
        return self._step(batch, 'val')

    def test_step(self, batch, batch_idx):
        """Run one test step.

        Args:
            batch: Test batch.
            batch_idx: Batch index.

        Returns:
            Test loss tensor.
        """
        return self._step(batch, 'test')

    def _batch_r2score(self, y_pred, y_target, lengths):
        """Return the mean R2 score across all samples in the batch.

        Args:
            y_pred: Flattened predicted values from unmasked pixels.
            y_target: Flattened target values from unmasked pixels.
            lengths: Number of unmasked pixels per sample.

        Returns:
            Scalar tensor containing the mean batch R2 score.
        """
        res_sum_squared = self._calc_segment_sum((y_target - y_pred) ** 2, lengths=lengths)
        total_sum_squared = self._calc_segment_sum(
            (y_target - torch.repeat_interleave(self._calc_segment_mean(y_target, lengths), lengths)) ** 2,
            lengths=lengths,
        )
        return torch.mean(1 - res_sum_squared / total_sum_squared)

    def _batch_pearson_corr(self, y_pred, y_target, lengths):
        """Return the mean Pearson correlation across all batch samples.

        Args:
            y_pred: Flattened predicted values from unmasked pixels.
            y_target: Flattened target values from unmasked pixels.
            lengths: Number of unmasked pixels per sample.

        Returns:
            Scalar tensor containing the mean Pearson correlation.
        """
        norm = 1 / (
            self._calc_segment_std(y_pred, lengths)
            * self._calc_segment_std(y_target, lengths)
            * (lengths - 1)
        )
        y_pred_mean = self._calc_segment_mean(y_pred, lengths)
        y_target_mean = self._calc_segment_mean(y_target, lengths)
        return torch.mean(
            norm
            * self._calc_segment_sum(
                (y_pred - torch.repeat_interleave(y_pred_mean, lengths))
                * (y_target - torch.repeat_interleave(y_target_mean, lengths)),
                lengths,
            )
        )

    def _calc_segment_std(self, data, lengths):
        """Calculate the per-sample standard deviation.

        Args:
            data: Flattened values.
            lengths: Segment sizes.

        Returns:
            Tensor with one standard deviation per segment.
        """
        mean = torch.segment_reduce(data, reduce="mean", lengths=lengths)
        mean_of_squares = torch.segment_reduce(data ** 2, reduce="mean", lengths=lengths)
        return torch.sqrt(mean_of_squares - mean**2)

    def _calc_segment_sum(self, data, lengths):
        """Calculate the per-sample sum.

        Args:
            data: Flattened values.
            lengths: Segment sizes.

        Returns:
            Tensor with one sum per segment.
        """
        return torch.segment_reduce(data, reduce="sum", lengths=lengths)

    def _calc_segment_mean(self, data, lengths):
        """Calculate the per-sample mean.

        Args:
            data: Flattened values.
            lengths: Segment sizes.

        Returns:
            Tensor with one mean per segment.
        """
        return torch.segment_reduce(data, reduce="mean", lengths=lengths)

    def _calc_volume_loss(self, y_pred, y_target, lengths):
        """Calculate the absolute error between predicted and target volume.

        Args:
            y_pred: Flattened predicted values from unmasked pixels.
            y_target: Flattened target values from unmasked pixels.
            lengths: Number of unmasked pixels per sample.

        Returns:
            Scalar tensor containing mean absolute volume error.
        """
        pred_sum = self._calc_segment_sum(y_pred, lengths)
        target_sum = self._calc_segment_sum(y_target, lengths)
        return torch.mean(torch.abs((pred_sum - target_sum)))

    def predict_step(self, batch, batch_idx):
        """Predict reconstructed height profiles for a batch.

        Args:
            batch: Tuple of patch tensors and reconstruction metadata.
            batch_idx: Batch index.

        Returns:
            Tuple of reconstructed profiles and corresponding sample IDs.
        """
        x, sample_info = batch
        y_pred = self(x)

        return self._reconstruct_height_profiles(y_pred, sample_info)

    def _reconstruct_height_profiles(self, y_pred, sample_info):
        """Reconstruct and denormalize height profiles from patch predictions.

        Args:
            y_pred: Predicted patch tensor.
            sample_info: Metadata with patch positions and normalization values.

        Returns:
            Tuple of denormalized reconstructed profiles and sample IDs.
        """
        normalization = sample_info.pop("normalization")
        normalizer = Normalizer(
            torch.tensor(normalization["mean"]),
            torch.tensor(normalization["std"]),
        )

        sample_ids = list(sample_info.keys())

        predicted_height_profiles = []
        current_patch_number = 0
        for sample_id in sample_ids:
            total_number_patches = sample_info[sample_id]["total_number_patches"]

            reconstructed_height_profile = self._reconstruct_single_height_profile(
                data=y_pred[current_patch_number: current_patch_number + total_number_patches],
                patch_positions=sample_info[sample_id]["positions"],
                imageshape=sample_info[sample_id]["imageshape"],
            )

            predicted_height_profiles.append(normalizer.denormalize(reconstructed_height_profile))
            # Advance over the flat batch of patches that belongs to this sample.
            current_patch_number += total_number_patches

        return predicted_height_profiles, sample_ids

    def _reconstruct_single_height_profile(self, data, patch_positions, imageshape):
        """Reconstruct one height profile from overlapping patches.

        Args:
            data: Tensor containing predicted patches for one sample.
            patch_positions: List of (top, left) locations for each patch.
            imageshape: Target output height and width.

        Returns:
            Reconstructed height profile tensor with shape (1, H, W).
        """
        reconstructed_profile = torch.full((1, *imageshape), torch.nan)
        patchshape = data.shape[-2:]

        max_pos = torch.tensor(patch_positions).max(dim=0).values
        min_pos = torch.tensor(patch_positions).min(dim=0).values

        for index, pos in enumerate(patch_positions):
            x_start_crop, x_end_crop = self._determine_crop(
                pos=pos[0], min_pos=min_pos[0], max_pos=max_pos[0]
            )
            y_start_crop, y_end_crop = self._determine_crop(
                pos=pos[1], min_pos=min_pos[1], max_pos=max_pos[1]
            )

            reconstructed_patch = reconstructed_profile[
                :,
                pos[0] + x_start_crop:pos[0] + patchshape[0] - x_end_crop,
                pos[1] + y_start_crop:pos[1] + patchshape[1] - y_end_crop,
            ]
            cropped_data = data[
                index,
                :,
                x_start_crop:patchshape[0] - x_end_crop,
                y_start_crop:patchshape[1] - y_end_crop,
            ]
            mask = torch.isnan(reconstructed_patch)
            # First write fills empty regions; later writes are blended in overlaps.
            reconstructed_patch[mask] = cropped_data[mask]
            reconstructed_patch[~mask] = torch.mean(
                torch.stack((reconstructed_patch[~mask], cropped_data[~mask])), dim=0
            )
        return reconstructed_profile

    def _determine_crop(self, pos, min_pos, max_pos):
        """Determine how much overlap to trim at the patch borders.

        Args:
            pos: Current patch position along one axis.
            min_pos: Minimum patch position for the sample.
            max_pos: Maximum patch position for the sample.

        Returns:
            Tuple of start and end crop sizes.
        """
        start_crop = self.reconstruction_overlap
        end_crop = self.reconstruction_overlap

        if pos == min_pos:
            start_crop = 0
        if pos == max_pos:
            end_crop = 0

        return start_crop, end_crop

    def configure_optimizers(self):
        """Create the optimizer used during training.

        Returns:
            Configured AdamW optimizer.
        """
        return torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

    @staticmethod
    def weights_init(model):
        if isinstance(model, nn.Conv2d):
            nn.init.kaiming_normal_(model.weight, mode='fan_out', nonlinearity='relu')
            if model.bias is not None:
                nn.init.constant_(model.bias, 0)
        elif isinstance(model, nn.BatchNorm2d):
            nn.init.constant_(model.weight, 1)
            nn.init.constant_(model.bias, 0)
