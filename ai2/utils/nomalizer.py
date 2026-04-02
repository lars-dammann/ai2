"""Normalization helpers for channel-wise and masked tensor transforms."""

import numpy as np
import torch


class Normalizer:
    """Apply channel-wise normalization and denormalization.

    Args:
        mean: Per-channel mean values.
        std: Per-channel standard deviation values.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.channels = len(self.mean)

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
