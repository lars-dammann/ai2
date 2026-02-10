import numpy as np
import torch

class Normalizer:

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.channels = len(self.mean)

    def normalize(self, data):
        return (data - self.mean[:, None, None]) / self.std[:, None, None]

    def _normalize(self, data):
        return ((data.reshape(self.channels, -1) - self.mean[:, None]) / self.std[:, None]).flatten()

    def normalize_masked(self, data, mask):
        cdata = torch.clone(data)
        mask = mask.to(bool)
        inv_mask = self._prepare_mask(mask)
        cdata[inv_mask] = self._normalize(data[inv_mask])
        return cdata

    def denormalize(self, data):
        return (data * self.std[:, None, None] + self.mean[:, None, None])

    def _denormalize(self, data):
        return ((data.reshape(self.channels, -1) * self.std[:, None] + self.mean[:, None])).flatten()

    def denormalize_masked(self, data, mask):
        cdata = torch.clone(data)
        inv_mask = self._prepare_mask(mask)
        cdata[inv_mask] = self._denormalize(data[inv_mask])
        return cdata

    def _prepare_mask(self, mask):
        return np.repeat(~mask, self.channels, axis=0).to(bool)