import numpy as np
import torch

class Normalizer:

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.channels = len(self.mean)

    def normalize(self, data):
        return ((data.reshape(self.channels, -1) - self.mean[:, None]) / self.std[:, None]).flatten()

    def normalize_masked(self, data, mask):
        cdata = torch.clone(data)
        inv_mask = self._prepare_mask(mask)
        cdata[inv_mask] = self.normalize(data[inv_mask])
        return cdata

    def denormalize(self, data):
        return ((data.reshape(self.channels, -1) * self.std[:, None] + self.mean[:, None])).flatten()

    def denormalize_masked(self, data, mask):
        cdata = torch.clone(data)
        inv_mask = self._prepare_mask(mask)
        cdata[inv_mask] = self.denormalize(data[inv_mask])
        return cdata

    def _prepare_mask(self, mask):
        return np.repeat(~mask, self.channels, axis=0).to(bool)