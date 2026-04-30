"""Normalization helpers for channel-wise and masked tensor transforms."""

import utils.image_dataset

from pathlib import Path

import numpy as np
import torch


class Normalizer:
    """Apply channel-wise normalization and denormalization.

    Args:
        mean: Per-channel mean values.
        std: Per-channel standard deviation values.
    """

    def __init__(self, mean, std):
        """Calculate mean and std on the training data."""
        self.mean = mean
        self.std = std
        self.channels = len(mean)

    @classmethod
    def get_concatenated_data(cls, before_image, before_height, after_height, mask):
        # Adapt mask to num channels
        image_mask = np.repeat(mask[np.newaxis, :, :], 5, axis=0)

        concatenated_data = np.concatenate(
            [before_image, before_height[None], after_height[None]], axis=0)

        return cls.get_masked_data(concatenated_data, image_mask)

    @classmethod
    def calculate_normalization_stats(cls, data_dir):
        """Calculate normalization statistics for a given data type and time point."""
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
        """
        Load image and apply mask.

        Args:
            data: Data array to be masked
            mask: Boolean mask (True for invalid pixels)

        Returns:
            Masked array with invalid pixels excluded
        """
        return np.ma.array(data, mask=mask)

    @staticmethod
    def calc_data_stats(img: np.ma.MaskedArray) -> tuple:
        """
        Calculate statistics for an image (per channel).

        Args:
            img: Image array with shape [C, H, W]

        Returns:
            Tuple of (count per channel, sum per channel, sum of squares per channel)
        """
        return (
            img.count(axis=(1, 2)),
            img.sum(axis=(1, 2)),
            (img * img).sum(axis=(1, 2))
        )

    @staticmethod
    def calc_std(count: np.ndarray, sum_val: np.ndarray, sum_squared: np.ndarray) -> np.ndarray:
        """
        Calculate standard deviation from sample statistics.

        Formula: std = sqrt((count * sum_squared - sum^2) / count^2)

        Args:
            count: Number of samples per channel
            sum_val: Sum of values per channel
            sum_squared: Sum of squared values per channel

        Returns:
            Standard deviation per channel
        """
        return np.sqrt(count * sum_squared - sum_val ** 2) / count

    @staticmethod
    def calc_mean(count: np.ndarray, sum_val: np.ndarray) -> np.ndarray:
        """
        Calculate mean from sample statistics.

        Args:
            count: Number of samples per channel
            sum_val: Sum of values per channel

        Returns:
            Mean value per channel
        """
        return sum_val / count

    def normalize(self, data):
        """Normalize all channels in a dense tensor.

        Args:
            data: Tensor with shape (C, H, W).

        Returns:
            Normalized tensor with the same shape.
        """
        return (data - self.mean[:, None, None]) / self.std[:, None, None]

    def _normalize(self, data):
        """Normalize a flattened masked tensor segment.

        Args:
            data: Flattened data from unmasked values.

        Returns:
            Flattened normalized values.
        """
        return ((data.reshape(self.channels, -1) - self.mean[:, None]) / self.std[:, None]).flatten()

    def normalize_masked(self, data, mask):
        """Normalize only the unmasked values in a tensor.

        Args:
            data: Tensor with shape (C, H, W).
            mask: Boolean mask with shape (1, H, W), where True means masked.

        Returns:
            Tensor with only unmasked positions normalized.
        """
        cdata = torch.clone(data)
        mask = mask.to(bool)
        inv_mask = self._prepare_mask(mask)
        cdata[inv_mask] = self._normalize(data[inv_mask])
        return cdata

    def denormalize(self, data):
        """Invert :meth:`normalize` on a dense tensor.

        Args:
            data: Normalized tensor with shape (C, H, W).

        Returns:
            Denormalized tensor with the same shape.
        """
        return (data * self.std[:, None, None] + self.mean[:, None, None])

    def _denormalize(self, data):
        """Denormalize a flattened masked tensor segment.

        Args:
            data: Flattened normalized values.

        Returns:
            Flattened denormalized values.
        """
        return ((data.reshape(self.channels, -1) * self.std[:, None] + self.mean[:, None])).flatten()

    def denormalize_masked(self, data, mask):
        """Denormalize only the unmasked values in a tensor.

        Args:
            data: Tensor with shape (C, H, W).
            mask: Boolean mask with shape (1, H, W), where True means masked.

        Returns:
            Tensor with only unmasked positions denormalized.
        """
        cdata = torch.clone(data)
        inv_mask = self._prepare_mask(mask)
        cdata[inv_mask] = self._denormalize(data[inv_mask])
        return cdata

    def _prepare_mask(self, mask):
        """Expand the mask across channels and convert it to boolean.

        Args:
            mask: Boolean mask with shape (1, H, W).

        Returns:
            Expanded boolean mask with shape (C, H, W).
        """
        return np.repeat(~mask, self.channels, axis=0).to(bool)


if __name__ == "__main__":
    data_dir = "/home/cld9301/WORK/repos/ai2/data"
    normalizer = Normalizer(data_dir=data_dir)
    print("Mean:", normalizer.mean)
    print("Std:", normalizer.std)
