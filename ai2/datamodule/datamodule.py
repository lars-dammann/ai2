import os
from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision.io import read_image
import lightning as pl

class CorrosionDataset(Dataset):
    def __init__(self, data_dir, config, transform=None):
        self.data_dir = Path(data_dir)
        self.before_image_dir = self.data_dir / "before/image"
        self.before_height_dir = self.data_dir / "before/height"
        self.after_height_dir = self.data_dir / "after/height"
        self.mask_dir = self.data_dir / "mask"
        self.sample_ids = [name[:-4] for name in os.listdir(self.before_image_dir)]
        self.transform = transform
        self.config = config

        normalization_dict = self.config["datamodule"]["normalization"]
        self.data_mean = np.concatenate((normalization_dict["before"]["image"]["mean"], normalization_dict["before"]["height"]["mean"], normalization_dict["after"]["height"]["mean"]), axis=0).astype(np.float32)
        self.data_std = np.concatenate((normalization_dict["before"]["image"]["std"], normalization_dict["before"]["height"]["std"], normalization_dict["after"]["height"]["std"]), axis=0).astype(np.float32)

        self.before_dim = len(normalization_dict["before"]["image"]["mean"]) + len(normalization_dict["before"]["height"]["mean"])
        self.data_dim = self.data_mean.shape[0]

        # self.target_transform = target_transform

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        # Get the sample id
        sample_id = self.sample_ids[idx]

        # Get the image and before and after height profile and stack them
        before_image = self._load_image(self.before_image_dir, sample_id)
        before_height = self._load_height(self.before_height_dir, sample_id)
        after_height = self._load_height(self.after_height_dir, sample_id)

        data = torch.cat((before_image, before_height, after_height), dim=0)

        # Load the mask
        mask = self._load_mask(self.mask_dir, sample_id)

        # Concatenate mask to data so it is transformed equally as data
        data_mask = torch.cat((data, mask), dim=0)

        # Conduct optional transforms of the data
        if self.transform is not None:
            data_mask = self.transform(data_mask)

        data = data_mask[:-1, :, :]
        mask = data_mask[-1:, :, :]
        mask = np.repeat(mask, self.data_dim, axis=0).to(bool)

        # Normalize the data
        data[~mask] = self._normalize_data(data[~mask])

        # Split the data back to before and after and return
        return data[:self.before_dim, :, :], data[self.before_dim:, :, :]

    def _load_height(self, path, sample_id):
        return torch.from_numpy((np.load(path / f"{sample_id}.npy")[np.newaxis, :, :]).astype(np.float32))

    def _load_mask(self, path, sample_id):
        mask = np.load(path / f"{sample_id}.npy").astype(bool)
        # mask = np.repeat(mask[np.newaxis, :, :], self.data_dim, axis=0)
        return torch.from_numpy(mask[np.newaxis, :, :])

    def _load_numpy(self, path, sample_id):
        return torch.from_numpy(np.load(path / f"{sample_id}.npy")[np.newaxis, :, :])

    def _load_image(self, path, sample_id):
        scale = v2.ToDtype(torch.float32, scale=True)
        return scale(read_image(path / f"{sample_id}.png"))

    def _normalize_data(self, data):
        return ((data.reshape(self.data_dim, -1) - self.data_mean[:, None]) / self.data_std[:, None]).flatten()


class CorrosionDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_dir = Path(config["datamodule"]["datadir"])
        self.data_size = config["datamodule"]["datasize"]
        self.batch_size = config["datamodule"]["batchsize"]
        self.train_transform = v2.Compose([v2.RandomCrop(self.data_size, pad_if_needed=True), v2.RandomHorizontalFlip(), v2.RandomVerticalFlip()])
        self.val_transform = v2.Compose([v2.RandomCrop(self.data_size, pad_if_needed=True)])

    def setup(self, stage: str):
        if stage == "fit":
            self.train_data = CorrosionDataset(self.data_dir / "train", self.config, transform=self.train_transform)
            self.val_data = CorrosionDataset(self.data_dir / "val", self.config, transform=self.val_transform)
        if stage == "test":
            self.test_data = CorrosionDataset(self.data_dir / "test", self.config, transform=self.val_transform)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)

# if __name__=="__main__":
#     # Example usage
#     print(Path(__file__).parent)
#     with open(Path(__file__).parent.parent / "configs/configs.json", "r", encoding="utf-8") as c:
#         config = json.load(c)
#     dataset = CorrosionDataset(data_dir=(Path(__file__).parent.parent / "data/train/"), config=config)
#     print(f"Dataset length: {len(dataset)}")
#     sample = dataset[0]
#     print(f"Before data shape: {sample[0].shape}")
#     print(f"After height shape: {sample[1].shape}")
