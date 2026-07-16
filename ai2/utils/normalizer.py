"""Normalization helpers for channel-wise and masked tensor transforms.

This module provides utilities to compute per-channel statistics and apply
normalization/denormalization to dense and masked data. Masked operations
operate on PyTorch tensors.
"""

import utils.image_dataset

from pathlib import Path

import numpy as np
import torch


class Normalizer:
    """Apply channel-wise normalization and denormalization.

    This helper encapsulates per-channel mean/std statistics and provides
    convenience methods to compute statistics from training data, apply
    normalization/denormalization to dense tensors and to masked tensors
    (only operating on unmasked values).

    Attributes:
        mean (numpy.ndarray): Per-channel mean values.
        std (numpy.ndarray): Per-channel standard deviation values.
        channels (int): Number of channels covered by the statistics.
    """

    def __init__(self, mean, std):
        """Initialize a Normalizer with precomputed per-channel stats.

        Parameters
        ----------
        mean : sequence or numpy.ndarray
            Per-channel mean values.
        std : sequence or numpy.ndarray
            Per-channel standard deviation values.
        """
        self.mean = mean
        self.std = std
        self.channels = len(mean)

    @classmethod
    def get_concatenated_data(cls, before_image, before_height, after_height, mask):
        """Concatenate channel data and apply an expanded mask.

        Args:
            before_image (numpy.ndarray): Image channels for the "before" time (shape ``(3, H, W)``).
            before_height (numpy.ndarray): Height channel for the "before" time (shape ``(H, W)``).
            after_height (numpy.ndarray): Height channel for the "after" time (shape ``(H, W)``).
            mask (numpy.ndarray): Boolean mask with shape ``(H, W)`` where True indicates masked/invalid pixels.

        Returns:
            numpy.ma.MaskedArray: Masked array of shape ``(5, H, W)`` with invalid pixels masked out.
        """

        # Adapt mask to num channels (expand to channel dimension)
        image_mask = np.repeat(mask[np.newaxis, :, :], 5, axis=0)

        concatenated_data = np.concatenate(
            [before_image, before_height[None], after_height[None]], axis=0)

        return cls.get_masked_data(concatenated_data, image_mask)

    @classmethod
    def calculate_normalization_stats(cls, data_dir):
        """Compute per-channel mean and standard deviation from training data.

        Scans the training split under ``data_dir`` and computes masked per-channel
        mean and standard deviation statistics using the per-pixel mask to ignore
        invalid pixels. The returned structure provides mean/std for before-image
        (3 channels), before-height, and after-height channels.

        Args:
            data_dir (str | pathlib.Path): Root directory containing the split folders (must include a ``train`` split).

        Returns:
            dict: Nested dictionary with keys ``"before"`` and ``"after"`` mapping to per-data-type mean/std arrays.
        """
        train_dataset_generator = utils.image_dataset.ImageDataset(
            data_dir=data_dir, split_type=["train"], image_loader="torchvision")

        channels = 5

        total_count = np.zeros(channels, dtype=np.int64)
        total_sum = np.zeros(channels, dtype=np.float64)
        total_sum_sqrd = np.zeros(channels, dtype=np.float64)

        for before_image, before_height, after_height, mask, sample_id in train_dataset_generator.iterate_files(["train/before/image", "train/before/height", "train/after/height", "train/mask"]):
            masked_data = cls.get_concatenated_data(
                before_image, before_height, after_height, mask)
            count, sum, sum_sqrd = cls.calc_data_stats(masked_data)
            total_count += count
            total_sum += sum
            total_sum_sqrd += sum_sqrd

        # Calculate final statistics
        mean = cls.calc_mean(total_count, total_sum)
        std = cls.calc_std(total_count, total_sum, total_sum_sqrd)

        return {"before": {"image": {"mean": mean[:3], "std": std[:3]}, "height": {"mean": mean[3:4], "std": std[3:4]}}, "after": {"height": {"mean": mean[4:5], "std": std[4:5]}}}

    @staticmethod
    def get_masked_data(data: np.ndarray, mask: np.ndarray) -> np.ma.MaskedArray:
        """Apply a boolean mask to a data array and return a masked array.

        Args:
            data (numpy.ndarray): Data array with shape ``(C, H, W)``.
            mask (numpy.ndarray): Boolean mask with shape ``(C, H, W)`` where ``True`` indicates masked/invalid pixels.

        Returns:
            numpy.ma.MaskedArray: Masked array with the same shape as ``data`` where masked pixels are excluded from subsequent statistics computations.
        """

        return np.ma.array(data, mask=mask)

    @staticmethod
    def calc_data_stats(img: np.ma.MaskedArray) -> tuple:
        """Calculate per-channel sample statistics from a masked image.

        Args:
            img (numpy.ma.MaskedArray): Masked image with shape ``(C, H, W)``.

        Returns:
            tuple: A tuple ``(count, sum, sum_sq)`` where each element is a 1-D array with one value per channel.
        """
        return (
            img.count(axis=(1, 2)),
            img.sum(axis=(1, 2)),
            (img * img).sum(axis=(1, 2))
        )

    @staticmethod
    def calc_std(count: np.ndarray, sum_val: np.ndarray, sum_squared: np.ndarray) -> np.ndarray:
        """Compute standard deviation per channel from accumulated statistics.

        Uses the numerically stable relation derived from second moment sums.

        Args:
            count (numpy.ndarray): Number of valid samples per channel.
            sum_val (numpy.ndarray): Sum of sample values per channel.
            sum_squared (numpy.ndarray): Sum of squared sample values per channel.

        Returns:
            numpy.ndarray: Standard deviation per channel.
        """
        return np.sqrt(count * sum_squared - sum_val ** 2) / count

    @staticmethod
    def calc_mean(count: np.ndarray, sum_val: np.ndarray) -> np.ndarray:
        """Compute mean per channel from accumulated statistics.

        Args:
            count (numpy.ndarray): Number of valid samples per channel.
            sum_val (numpy.ndarray): Sum of sample values per channel.

        Returns:
            numpy.ndarray: Mean value per channel.
        """
        return sum_val / count

    def normalize(self, data):
        """Normalize a dense (unmasked) tensor per channel.

        Args:
            data (numpy.ndarray | torch.Tensor): Array of shape ``(C, H, W)`` containing channel data.

        Returns:
            numpy.ndarray | torch.Tensor: Channel-wise normalized array with the same shape as ``data``.
        """
        return (data - self.mean[:, None, None]) / self.std[:, None, None]

    def _normalize(self, data):
        """Normalize flattened unmasked values for all channels.

        Args:
            data (numpy.ndarray): Flattened array of unmasked values for all channels concatenated.

        Returns:
            numpy.ndarray: Flattened normalized values.
        """
        return ((data.reshape(self.channels, -1) - self.mean[:, None]) / self.std[:, None]).flatten()

    def normalize_masked(self, data, mask):
        """Normalize only the unmasked (valid) entries of a tensor.

        Args:
            data (torch.Tensor): Tensor of shape ``(C, H, W)`` to normalize. This method operates on PyTorch tensors.
            mask (torch.Tensor): Boolean mask with shape ``(1, H, W)`` where ``True`` indicates masked/invalid pixels.

        Returns:
            torch.Tensor: Tensor with the same shape as ``data`` where only the unmasked positions have been normalized.
        """
        cdata = torch.clone(data)
        mask = mask.to(bool)
        inv_mask = self._prepare_mask(mask)
        cdata[inv_mask] = self._normalize(data[inv_mask])
        return cdata

    def denormalize(self, data):
        """Denormalize a dense (unmasked) tensor per channel.

        Args:
            data (numpy.ndarray | torch.Tensor): Normalized array of shape ``(C, H, W)``.

        Returns:
            numpy.ndarray | torch.Tensor: Denormalized array with the same shape.
        """
        return (data * self.std[:, None, None] + self.mean[:, None, None])

    def _denormalize(self, data):
        """Denormalize flattened normalized values for all channels.

        Args:
            data (numpy.ndarray): Flattened normalized values for all channels concatenated.

        Returns:
            numpy.ndarray: Flattened denormalized values.
        """
        return ((data.reshape(self.channels, -1) * self.std[:, None] + self.mean[:, None])).flatten()

    def denormalize_masked(self, data, mask):
        """Denormalize only the unmasked (valid) entries of a tensor.

        Args:
            data (torch.Tensor): Tensor of shape ``(C, H, W)`` containing normalized values.
            mask (torch.Tensor): Boolean mask with shape ``(1, H, W)`` where ``True`` indicates masked/invalid pixels.

        Returns:
            torch.Tensor: Tensor with the same shape as ``data`` where only the unmasked positions have been denormalized.
        """
        cdata = torch.clone(data)
        inv_mask = self._prepare_mask(mask)
        cdata[inv_mask] = self._denormalize(data[inv_mask])
        return cdata

    def _prepare_mask(self, mask):
        """Prepare a boolean mask indexed by channel.

        The input mask is expected to have shape ``(1, H, W)`` with ``True`` marking
        masked pixels. This method inverts the mask to yield a boolean selector for
        the valid (unmasked) pixels and repeats it across the channel dimension
        producing an array of shape ``(C, H, W)``.

        Args:
            mask (torch.Tensor): Boolean mask with shape ``(1, H, W)`` where ``True`` indicates masked/invalid pixels. This method is implemented to work with PyTorch tensors for masked operations.

        Returns:
            torch.Tensor: Boolean tensor with shape ``(C, H, W)`` where ``True`` denotes valid/unmasked positions.
        """

        return np.repeat(~mask, self.channels, axis=0).to(bool)
