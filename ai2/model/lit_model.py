from utils.normalizer import Normalizer
from model.unet import UNet

from typing import Dict, Tuple, Any, List, Optional

import lightning as pl
import torch
import torch.nn as nn


class CorrosionUNet(pl.LightningModule):
    """Lightning module that wraps the UNet and computes training metrics.

    Args:
        model_config (dict): UNet and optimizer configuration dictionary.
        reconstruction_overlap (int): Overlap in pixels used for patch reconstruction.
    """

    def __init__(self, model_config: dict, reconstruction_overlap: int = 0) -> None:
        super().__init__()
        self.model = UNet(model_config, in_channels=4, out_channels=1)
        self.learning_rate = model_config["lr"]
        self.weight_decay = model_config["weight_decay"]
        self.reconstruction_overlap = reconstruction_overlap
        self.train_summary = False
        self.val_summary = False
        self.test_summary = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the underlying UNet.

        Args:
            x (torch.Tensor): Input tensor with shape (B, 4, H, W).

        Returns:
            torch.Tensor: Predicted height tensor with shape (B, 1, H, W).
        """
        return self.model(x)

    def _log_loss(self, loss_dict: Dict[str, torch.Tensor], prefix: str) -> None:
        """Log all losses for the given stage prefix.

        Args:
            loss_dict (dict): Mapping of metric names to tensors.
            prefix (str): Stage prefix such as 'train', 'val', or 'test'.

        Returns:
            None
        """
        for key, value in loss_dict.items():
            self.log(f"{prefix}-{key}", value)
            # self.log('best-' + prefix + '-' + key, value)

    def _calc_losses(self, y_pred: torch.Tensor, y_target: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calculate losses and metrics for a masked batch.

        Args:
            y_pred (torch.Tensor): Predicted tensor.
            y_target (torch.Tensor): Target tensor.
            mask (torch.Tensor): Boolean mask where True marks ignored values.

        Returns:
            dict: Dictionary containing scalar loss and metric tensors.
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

    def on_fit_start(self) -> None:
        """Configure W&B summaries for the tracked metrics.

        Returns:
            None
        """
        wandb_run = self.logger.experiment

        scores = ["loss", "mae-loss", "mse-loss", "r2score", "corr", "volume-loss-abs"]
        for score in scores:
            for prefix in ["train", "val", "test"]:
                summary = "min" if "loss" in score else "max"
                wandb_run.define_metric(f"{prefix}-{score}", summary=summary)

    def _step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], prefix: str) -> torch.Tensor:
        """Run a shared train/val/test step.

        Args:
            batch (tuple): Tuple of inputs, targets, and mask.
            prefix (str): Stage prefix such as 'train', 'val', or 'test'.

        Returns:
            torch.Tensor: Main optimization loss tensor.
        """
        x, y_target, mask = batch
        y_pred = self(x)
        losses = self._calc_losses(y_pred, y_target, mask)
        losses["loss"] = losses["mae-loss"]
        self._log_loss(losses, prefix)
        return losses["loss"]

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run one training step.

        Args:
            batch (tuple): Training batch.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Training loss tensor.
        """
        return self._step(batch, 'train')

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run one validation step.

        Args:
            batch (tuple): Validation batch.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Validation loss tensor.
        """
        return self._step(batch, 'val')

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run one test step.

        Args:
            batch (tuple): Test batch.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Test loss tensor.
        """
        return self._step(batch, 'test')

    def _batch_r2score(self, y_pred: torch.Tensor, y_target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Return the mean R2 score across all samples in the batch.

        Args:
            y_pred (torch.Tensor): Flattened predicted values from unmasked pixels.
            y_target (torch.Tensor): Flattened target values from unmasked pixels.
            lengths (torch.Tensor): Number of unmasked pixels per sample.

        Returns:
            torch.Tensor: Scalar tensor containing the mean batch R2 score.
        """
        res_sum_squared = self._calc_segment_sum((y_target - y_pred) ** 2, lengths=lengths)
        total_sum_squared = self._calc_segment_sum(
            (y_target - torch.repeat_interleave(self._calc_segment_mean(y_target, lengths), lengths)) ** 2,
            lengths=lengths,
        )
        return torch.mean(1 - res_sum_squared / total_sum_squared)

    def _batch_pearson_corr(self, y_pred: torch.Tensor, y_target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Return the mean Pearson correlation across all batch samples.

        Args:
            y_pred (torch.Tensor): Flattened predicted values from unmasked pixels.
            y_target (torch.Tensor): Flattened target values from unmasked pixels.
            lengths (torch.Tensor): Number of unmasked pixels per sample.

        Returns:
            torch.Tensor: Scalar tensor containing the mean Pearson correlation.
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

    def _calc_segment_std(self, data: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Calculate the per-sample standard deviation.

        Args:
            data (torch.Tensor): Flattened values.
            lengths (torch.Tensor): Segment sizes.

        Returns:
            torch.Tensor: Tensor with one standard deviation per segment.
        """
        mean = torch.segment_reduce(data, reduce="mean", lengths=lengths)
        mean_of_squares = torch.segment_reduce(data ** 2, reduce="mean", lengths=lengths)
        return torch.sqrt(mean_of_squares - mean**2)

    def _calc_segment_sum(self, data: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Calculate the per-sample sum.

        Args:
            data (torch.Tensor): Flattened values.
            lengths (torch.Tensor): Segment sizes.

        Returns:
            torch.Tensor: Tensor with one sum per segment.
        """
        return torch.segment_reduce(data, reduce="sum", lengths=lengths)

    def _calc_segment_mean(self, data: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Calculate the per-sample mean.

        Args:
            data (torch.Tensor): Flattened values.
            lengths (torch.Tensor): Segment sizes.

        Returns:
            torch.Tensor: Tensor with one mean per segment.
        """
        return torch.segment_reduce(data, reduce="mean", lengths=lengths)

    def _calc_volume_loss(self, y_pred: torch.Tensor, y_target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Calculate the absolute error between predicted and target volume.

        Args:
            y_pred (torch.Tensor): Flattened predicted values from unmasked pixels.
            y_target (torch.Tensor): Flattened target values from unmasked pixels.
            lengths (torch.Tensor): Number of unmasked pixels per sample.

        Returns:
            torch.Tensor: Scalar tensor containing mean absolute volume error.
        """
        pred_sum = self._calc_segment_sum(y_pred, lengths)
        target_sum = self._calc_segment_sum(y_target, lengths)
        return torch.mean(torch.abs((pred_sum - target_sum)))

    def predict_step(self, batch: Tuple[torch.Tensor, dict], batch_idx: int) -> Tuple[List[torch.Tensor], List[str]]:
        """Predict reconstructed height profiles for a batch.

        Args:
            batch (tuple): Tuple of patch tensors and reconstruction metadata.
            batch_idx (int): Batch index.

        Returns:
            tuple[list[torch.Tensor], list[str]]: Tuple of reconstructed profiles and corresponding sample IDs.
        """
        x, sample_info = batch
        y_pred = self(x)

        return self._reconstruct_height_profiles(y_pred, sample_info)

    def _reconstruct_height_profiles(self, y_pred: torch.Tensor, sample_info: dict) -> Tuple[List[torch.Tensor], List[str]]:
        """Reconstruct and denormalize height profiles from patch predictions.

        Args:
            y_pred (torch.Tensor): Predicted patch tensor.
            sample_info (dict): Metadata with patch positions and normalization values.

        Returns:
            tuple[list[torch.Tensor], list[str]]: Tuple of denormalized reconstructed profiles and sample IDs.
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

    def _reconstruct_single_height_profile(self, data: torch.Tensor, patch_positions: List[Tuple[int, int]], imageshape: Tuple[int, int]) -> torch.Tensor:
        """Reconstruct one height profile from overlapping patches.

        Args:
            data (torch.Tensor): Tensor containing predicted patches for one sample.
            patch_positions (list[tuple[int, int]]): List of (top, left) locations for each patch.
            imageshape (tuple[int, int]): Target output height and width.

        Returns:
            torch.Tensor: Reconstructed height profile tensor with shape (1, H, W).
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

    def _determine_crop(self, pos: int, min_pos: int, max_pos: int) -> Tuple[int, int]:
        """Determine how much overlap to trim at the patch borders.

        Args:
            pos (int): Current patch position along one axis.
            min_pos (int): Minimum patch position for the sample.
            max_pos (int): Maximum patch position for the sample.

        Returns:
            tuple[int, int]: Tuple of start and end crop sizes.
        """
        start_crop = self.reconstruction_overlap
        end_crop = self.reconstruction_overlap

        if pos == min_pos:
            start_crop = 0
        if pos == max_pos:
            end_crop = 0

        return start_crop, end_crop

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Create the optimizer used during training.

        Returns:
            torch.optim.Optimizer: Configured AdamW optimizer.
        """
        return torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

    @staticmethod
    def weights_init(model: torch.nn.Module) -> None:
        """Initialize model weights.

        Uses Kaiming normal initialization for Conv2d layers and constant initialization
        for BatchNorm2d layers.

        Args:
            model (torch.nn.Module): The model to initialize.
        """
        if isinstance(model, nn.Conv2d):
            nn.init.kaiming_normal_(model.weight, mode='fan_out', nonlinearity='relu')
            if model.bias is not None:
                nn.init.constant_(model.bias, 0)
        elif isinstance(model, nn.BatchNorm2d):
            nn.init.constant_(model.weight, 1)
            nn.init.constant_(model.bias, 0)
