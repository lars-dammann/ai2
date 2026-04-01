from utils.nomalizer import Normalizer
from utils.get_data import get_mask, get_image, get_height

import os
from pathlib import Path
from collections import defaultdict, ChainMap

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision.io import read_image
import lightning as pl


class CorrosionDataset(Dataset):
    """
    Dataset class that provides the data for the train, val and test steps.
    The feature data are (4, H, W) datasets,  consisting of 3 channles RGB image
    and 1 channel height profile.
    The target data is a (1, H, W) height profile.
    """

    def __init__(self, data_dir, config, transform=None, photometric_transform=None):
        self.data_dir = Path(data_dir)
        self.before_image_dir = self.data_dir / "before/image"
        self.before_height_dir = self.data_dir / "before/height"
        self.after_height_dir = self.data_dir / "after/height"
        self.mask_dir = self.data_dir / "mask"
        self.sample_ids = [name[:-4]
                           for name in os.listdir(self.before_image_dir)]
        self.transform = transform
        self.photometric_transform = photometric_transform
        self.config = config

        self.normalizer = self._get_normalizer()

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        # Get the sample id
        sample_id = self.sample_ids[idx]

        # Get the image and before and after height profile and stack them
        before_image = get_image(self.before_image_dir, sample_id)
        before_height = get_height(self.before_height_dir, sample_id)
        after_height = get_height(self.after_height_dir, sample_id)

        before_dim = before_image.shape[0] + before_height.shape[0]

        data = torch.cat((before_image, before_height, after_height), dim=0)

        # Load the mask
        mask = get_mask(self.mask_dir, sample_id)
        # Invert mask to make more suitable for the transforms
        mask = ~mask

        # Concatenate inverse mask to data so it is transformed equally as data
        data_mask = torch.cat((data, mask), dim=0)

        # Conduct optional transforms of the data
        if self.transform is not None:
            data_mask = self.transform(data_mask)

        data = data_mask[:-1, :, :]
        mask = data_mask[-1:, :, :]

        # Floor the mask (any value between 0 and 1 should be masked) and convert to bool
        mask = torch.floor(mask).to(bool)
        # Reinvert the mask
        mask = ~mask

        # Apply photometric transforms if specified
        if self.photometric_transform is not None:
            data[:3, :, :] = self.photometric_transform(data[:3, :, :])

        # Normalize the data
        data = self.normalizer.normalize_masked(data, mask)

        # Split the data back to before and after and return
        return data[:before_dim, :, :], data[before_dim:, :, :], mask

    def _get_normalizer(self):
        """
        Utility function to get the right normalizer object for the data
        """
        normalization_dict = self.config["normalization"]
        mean = np.concatenate(
            (normalization_dict["before"]["image"]["mean"],
             normalization_dict["before"]["height"]["mean"],
             normalization_dict["after"]["height"]["mean"]),
            axis=0).astype(
            np.float32)
        std = np.concatenate(
            (normalization_dict["before"]["image"]["std"],
             normalization_dict["before"]["height"]["std"],
             normalization_dict["after"]["height"]["std"]),
            axis=0).astype(
            np.float32)
        return Normalizer(mean, std)


class PredictCorrosionDataset(CorrosionDataset):
    """
    Dataset Class providing the rightly formatted data
    for predicting the height profiles from the test data
    """

    def __init__(self, data_dir, config, datasize, reconstruction_overlap=0):
        super().__init__(data_dir, config, transform=None)
        self.datasize = datasize
        self.reconstruction_overlap = reconstruction_overlap

    def __getitem__(self, idx):
        # Get the sample id
        sample_id = self.sample_ids[idx]

        # Get the image and before and after height profile and stack them
        data = torch.cat((get_image(self.before_image_dir, sample_id),
                         get_height(self.before_height_dir, sample_id)), dim=0)
        imageshape = data.shape[-2:]

        # Normalize the data
        mask = get_mask(self.mask_dir, sample_id)
        data = self.normalizer.normalize_masked(data, mask)

        # Determine the sliding lenght of each patch
        slide_length = (self.datasize - 2 * self.reconstruction_overlap)

        # Create tensor to store the results of the sliding window
        patch_positions = []
        n_height, n_width = data.shape[-2]//slide_length, data.shape[-1]//slide_length
        patch_tensor = torch.full(
            ((n_height + 1),
             (n_width + 1),
             data.shape[0],
             self.datasize, self.datasize),
            torch.nan,
            dtype=data.dtype)

        # Sliding window to extract patches
        for n_h in range(n_height):
            height_position = n_h * (self.datasize - 2 * self.reconstruction_overlap)
            self._slide_width(data, n_width, n_h, height_position, patch_positions, patch_tensor)
        height_position = data.shape[-2] - self.datasize
        self._slide_width(data, n_width, n_height, height_position, patch_positions, patch_tensor)

        # Store information about sample patches
        sample_info = defaultdict(dict)
        sample_info[sample_id]["positions"] = patch_positions
        sample_info[sample_id]["total_number_patches"] = ((n_height+1) * (n_width+1))
        sample_info[sample_id]["imageshape"] = imageshape
        sample_info["normalization"] = self.config["normalization"]["after"]["height"]

        # Split the data back to before and after and return
        return torch.flatten(patch_tensor, start_dim=0, end_dim=1), sample_info

    def _slide_width(self, data, n_width, n_h, height_position, patch_positions, patch_tensor):
        # Crop the data at multiple positions along the image width
        top = height_position
        for n_w in range(n_width):
            left = n_w * (self.datasize - 2 * self.reconstruction_overlap)
            patch_positions.append((top, left))
            patch_tensor[n_h, n_w, :, :, :] = v2.functional.crop(
                data, top=top, left=left, height=self.datasize, width=self.datasize)

        # Add the last overlapping image for this height
        left = data.shape[-1] - self.datasize
        patch_positions.append((top, left))
        patch_tensor[n_h, n_width, :, :, :] = v2.functional.crop(
            data, top=top, left=left, height=self.datasize, width=self.datasize)

    def _get_normalizer(self):
        """
        Utility function to get the right normalizer object for the data
        """
        normalization_dict = self.config["normalization"]
        mean = np.concatenate(
            (normalization_dict["before"]["image"]["mean"],
             normalization_dict["before"]["height"]["mean"]),
            axis=0).astype(
            np.float32)
        std = np.concatenate(
            (normalization_dict["before"]["image"]["std"],
             normalization_dict["before"]["height"]["std"]),
            axis=0).astype(
            np.float32)
        return Normalizer(mean, std)


class CorrosionDataModule(pl.LightningDataModule):
    """
    Lightning data module that provides the data for the train, val, test and predict steps.
    """

    def __init__(self, datamodule_config, reconstruction_overlap=0, predict_dataset="test"):
        super().__init__()
        self.save_hyperparameters(ignore=["reconstruction_overlap", "predict_dataset"])
        self.config = datamodule_config
        self.data_dir = Path(datamodule_config["datadir"])
        self.predict_dataset = predict_dataset
        self.data_size = datamodule_config["datasize"]
        self.reconstruction_overlap = reconstruction_overlap
        self.batch_size = datamodule_config["batch_size"]
        self.train_transform = v2.Compose(
            [v2.RandomRotation(degrees=(0, 90), interpolation=v2.InterpolationMode.BILINEAR),
             v2.RandomCrop(self.data_size, pad_if_needed=True),
             v2.Lambda(CorrosionDataModule.random_rot90),
             v2.RandomHorizontalFlip(),
             v2.RandomVerticalFlip()])
        self.photometric_transform = v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05, hue=0.01)
        self.val_transform = v2.Compose(
            [v2.RandomCrop(self.data_size, pad_if_needed=True)])

        try:
            self.num_workers = int(os.environ['SLURM_CPUS_PER_TASK'])
        except KeyError:
            self.num_workers = 2

    @staticmethod
    def random_rot90(img):
        # img: torch tensor, shape [C,H,W]
        k = torch.randint(0, 4, (1,)).item()  # 0,1,2,3
        return torch.rot90(img, k, dims=[1,2])

    @staticmethod
    def predict_coallate_function(batch):
        """
        Transform the different patches from the same image as subsequent images in the batch
        """
        data, position_dicts = zip(*batch)
        return torch.cat(data, 0), dict(pair for d in position_dicts for pair in d.items())

    def setup(self, stage: str):
        if stage == "fit":
            self.train_data = CorrosionDataset(
                self.data_dir / "train", self.config, transform=self.train_transform, photometric_transform=self.photometric_transform)
            self.val_data = CorrosionDataset(
                self.data_dir / "val", self.config, transform=self.val_transform)
        if stage == "test":
            self.test_data = CorrosionDataset(
                self.data_dir / "test", self.config, transform=self.val_transform)
        if stage == "predict":
            self.predict_data = PredictCorrosionDataset(
                self.data_dir / self.predict_dataset, self.config, self.data_size,
                reconstruction_overlap=self.reconstruction_overlap)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(
            self.predict_data, batch_size=self.batch_size,
            collate_fn=CorrosionDataModule.predict_coallate_function, shuffle=False,
            num_workers=self.num_workers)

# Possible data augmentations, interpolation methods could be bilinear
# Shift image and height against each other by small margins
# RandomRotation, RandomAffine, ElasticTransform, RandomResizedCrop, GaussianBlur
# Only Image: ColorJitter
