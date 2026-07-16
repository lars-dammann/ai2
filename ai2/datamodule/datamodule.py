from utils.normalizer import Normalizer
from utils.get_data import get_mask, get_image, get_height

import os
from pathlib import Path
from collections import defaultdict
from typing import Callable, Iterable, List, Tuple, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import lightning as pl


class CorrosionDataset(Dataset):
    """Dataset for supervised training, validation, and testing.

    Args:
        data_dir (str | pathlib.Path): Root directory for one split containing before/after/mask folders.
        config (dict): Configuration dictionary with normalization statistics.
        transform (callable | None): Optional spatial transform applied jointly to data and mask.
        photometric_transform (callable | None): Optional transform applied only to RGB channels.
    """

    def __init__(self, data_dir: Union[str, Path], config: dict, transform: Optional[Callable] = None, photometric_transform: Optional[Callable] = None):
        self.data_dir = Path(data_dir)
        self.before_image_dir = self.data_dir / "before/image"
        self.before_height_dir = self.data_dir / "before/height"
        self.after_height_dir = self.data_dir / "after/height"
        self.mask_dir = self.data_dir / "mask"
        self.sample_ids = [name[:-4] for name in os.listdir(self.before_image_dir)]
        self.transform = transform
        self.photometric_transform = photometric_transform
        self.config = config

        self.normalizer = self._get_normalizer()

    def __len__(self) -> int:
        """Return the number of available samples.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load and preprocess a single sample.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing model input, target height map, and boolean mask.
        """
        sample_id = self.sample_ids[idx]

        before_image = get_image(self.before_image_dir, sample_id)
        before_height = get_height(self.before_height_dir, sample_id)
        after_height = get_height(self.after_height_dir, sample_id)

        before_dim = before_image.shape[0] + before_height.shape[0]

        data = torch.cat((before_image, before_height, after_height), dim=0)

        mask = get_mask(self.mask_dir, sample_id)
        # Transforms operate on image-like tensors, so invert once before augmentation.
        mask = ~mask

        data_mask = torch.cat((data, mask), dim=0)

        if self.transform is not None:
            data_mask = self.transform(data_mask)

        data = data_mask[:-1, :, :]
        mask = data_mask[-1:, :, :]

        # Interpolating transforms may produce fractional mask values.
        mask = torch.floor(mask).to(bool)
        # Restore original convention: True means masked pixel.
        mask = ~mask

        if self.photometric_transform is not None:
            data[:3, :, :] = self.photometric_transform(data[:3, :, :])

        data = self.normalizer.normalize_masked(data, mask)

        return data[:before_dim, :, :], data[before_dim:, :, :], mask

    def _get_normalizer(self) -> Normalizer:
        """Build the normalizer from the configuration values.

        Returns:
            Normalizer: Configured for RGB, before-height, and after-height channels.
        """
        normalization_dict = self.config["normalization"]
        mean = np.concatenate(
            (np.array(normalization_dict["before"]["image"]["mean"], dtype=np.float32).reshape(3),
             np.array(normalization_dict["before"]["height"]["mean"], dtype=np.float32).reshape(1),
             np.array(normalization_dict["after"]["height"]["mean"], dtype=np.float32).reshape(1)),
            axis=0)
        std = np.concatenate(
            (np.array(normalization_dict["before"]["image"]["std"], dtype=np.float32).reshape(3),
             np.array(normalization_dict["before"]["height"]["std"], dtype=np.float32).reshape(1),
             np.array(normalization_dict["after"]["height"]["std"], dtype=np.float32).reshape(1)),
            axis=0)
        return Normalizer(mean, std)


class PredictCorrosionDataset(CorrosionDataset):
    """Dataset that yields overlapping patches for prediction.

    Args:
        data_dir (str | pathlib.Path): Root directory for the prediction split.
        config (dict): Configuration dictionary with normalization statistics.
        datasize (int): Patch size used for model inference.
        reconstruction_overlap (int): Number of overlapping pixels on each patch border.
    """

    def __init__(self, data_dir: Union[str, Path], config: dict, datasize: int, reconstruction_overlap: int = 0):
        super().__init__(data_dir, config, transform=None)
        self.datasize = datasize
        self.reconstruction_overlap = reconstruction_overlap

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        """Load one sample and split it into normalized patches.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple[torch.Tensor, dict]: Tuple with flattened patch tensor and reconstruction metadata.
        """
        sample_id = self.sample_ids[idx]

        data = torch.cat((get_image(self.before_image_dir, sample_id),
                          get_height(self.before_height_dir, sample_id)), dim=0, )
        imageshape = data.shape[-2:]

        mask = get_mask(self.mask_dir, sample_id)
        data = self.normalizer.normalize_masked(data, mask)

        slide_length = self.datasize - 2 * self.reconstruction_overlap

        patch_positions = []
        n_height, n_width = data.shape[-2] // slide_length, data.shape[-1] // slide_length
        patch_tensor = torch.full(
            (n_height + 1, n_width + 1, data.shape[0], self.datasize, self.datasize),
            torch.nan,
            dtype=data.dtype)

        for n_h in range(n_height):
            height_position = n_h * slide_length
            self._slide_width(data, n_width, n_h, height_position, patch_positions, patch_tensor)
        # Always include the terminal row to cover the bottom border.
        height_position = data.shape[-2] - self.datasize
        self._slide_width(data, n_width, n_height, height_position, patch_positions, patch_tensor)

        sample_info = defaultdict(dict)
        sample_info[sample_id]["positions"] = patch_positions
        sample_info[sample_id]["total_number_patches"] = (n_height + 1) * (n_width + 1)
        sample_info[sample_id]["imageshape"] = imageshape
        sample_info["normalization"] = {
            "mean": np.array(
                self.config["normalization"]["after"]["height"]["mean"],
                dtype=np.float32).reshape(1),
            "std": np.array(
                self.config["normalization"]["after"]["height"]["std"],
                dtype=np.float32).reshape(1)}

        return torch.flatten(patch_tensor, start_dim=0, end_dim=1), sample_info

    def _slide_width(self, data: torch.Tensor, n_width: int, n_h: int, height_position: int, patch_positions: List[Tuple[int, int]], patch_tensor: torch.Tensor) -> None:
        """Crop a row of patches across the image width.

        Args:
            data (torch.Tensor): Full normalized sample tensor.
            n_width (int): Number of non-terminal horizontal patch steps.
            n_h (int): Row index in the patch tensor.
            height_position (int): Vertical start index for the current row.
            patch_positions (list): Output list of (top, left) patch positions.
            patch_tensor (torch.Tensor): Output tensor storing all cropped patches.

        Returns:
            None: Results are written into patch_positions and patch_tensor.
        """
        top = height_position
        for n_w in range(n_width):
            left = n_w * (self.datasize - 2 * self.reconstruction_overlap)
            patch_positions.append((top, left))
            patch_tensor[n_h, n_w, :, :, :] = v2.functional.crop(
                data, top=top, left=left, height=self.datasize, width=self.datasize)

        # Always include the terminal column to cover the right border.
        left = data.shape[-1] - self.datasize
        patch_positions.append((top, left))
        patch_tensor[n_h, n_width, :, :, :] = v2.functional.crop(
            data, top=top, left=left, height=self.datasize, width=self.datasize)

    def _get_normalizer(self) -> Normalizer:
        """Build the prediction normalizer from the configuration values.

        Returns:
            Normalizer: Configured for RGB and before-height channels.
        """
        normalization_dict = self.config["normalization"]
        mean = np.concatenate(
            (np.array(normalization_dict["before"]["image"]["mean"], dtype=np.float32).reshape(3),
             np.array(normalization_dict["before"]["height"]["mean"], dtype=np.float32).reshape(1)),
            axis=0)
        std = np.concatenate(
            (np.array(normalization_dict["before"]["image"]["std"], dtype=np.float32).reshape(3),
             np.array(normalization_dict["before"]["height"]["std"], dtype=np.float32).reshape(1)),
            axis=0)
        return Normalizer(mean, std)


class CorrosionDataModule(pl.LightningDataModule):
    """Lightning DataModule for train, validation, test, and predict steps.

    Args:
        datamodule_config (dict): Data-related configuration including paths and batch size.
        reconstruction_overlap (int): Overlap in pixels used by prediction reconstruction.
        predict_dataset (str): Split name used during prediction.
    """

    def __init__(self, data_dir: Union[str, Path], datamodule_config: dict, reconstruction_overlap: int = 0,
                 predict_dataset: str = "test") -> None:
        super().__init__()
        self.config = datamodule_config
        self.data_dir = Path(data_dir)
        self.predict_dataset = predict_dataset
        self.data_size = datamodule_config["datasize"]
        self.reconstruction_overlap = reconstruction_overlap
        self.batch_size = datamodule_config["batch_size"]
        self.train_transform = v2.Compose(
            [v2.RandomCrop(self.data_size, pad_if_needed=True),
             v2.Lambda(CorrosionDataModule.random_rot90),
             v2.RandomHorizontalFlip(),
             v2.RandomVerticalFlip()])
        color_jitter_config = datamodule_config["color_jitter"]
        self.photometric_transform = v2.ColorJitter(
            brightness=color_jitter_config["brightness"],
            contrast=color_jitter_config["contrast"],
            saturation=color_jitter_config["saturation"],
            hue=color_jitter_config["hue"])
        self.val_transform = v2.Compose(
            [v2.RandomCrop(self.data_size, pad_if_needed=True)])

        try:
            self.num_workers = int(os.environ['SLURM_CPUS_PER_TASK'])
        except KeyError:
            self.num_workers = 2

    @staticmethod
    def random_rot90(img: torch.Tensor) -> torch.Tensor:
        """Rotate a tensor by a random multiple of 90 degrees.

        Args:
            img (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Rotated tensor with the same shape as input.
        """
        k = torch.randint(0, 4, (1,)).item()
        return torch.rot90(img, k, dims=[1, 2])

    @staticmethod
    def predict_coallate_function(batch: Iterable[Tuple[torch.Tensor, dict]]) -> Tuple[torch.Tensor, dict]:
        """Flatten patch batches so all patches from one sample stay adjacent.

        Args:
            batch (Iterable[tuple[torch.Tensor, dict]]): Iterable of (patch_tensor, sample_info) items.

        Returns:
            tuple[torch.Tensor, dict]: Tuple of concatenated patch tensors and merged metadata dictionary.
        """
        data, position_dicts = zip(*batch)
        return torch.cat(data, 0), dict(pair for d in position_dicts for pair in d.items())

    def setup(self, stage: Optional[str]) -> None:
        """Create datasets for the requested Lightning stage.

        Args:
            stage (str): One of 'fit', 'test', or 'predict'.

        Returns:
            None
        """
        if stage == "fit":
            self.train_data = CorrosionDataset(
                self.data_dir / "train", self.config, transform=self.train_transform,
                photometric_transform=self.photometric_transform)
            self.val_data = CorrosionDataset(
                self.data_dir / "val", self.config, transform=self.val_transform)
        if stage == "test":
            self.test_data = CorrosionDataset(
                self.data_dir / "test", self.config, transform=self.val_transform)
        if stage == "predict":
            self.predict_data = PredictCorrosionDataset(
                self.data_dir / self.predict_dataset, self.config, self.data_size,
                reconstruction_overlap=self.reconstruction_overlap)

    def train_dataloader(self) -> DataLoader:
        """Build the training DataLoader.

        Returns:
            torch.utils.data.DataLoader: DataLoader for training samples.
        """
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Build the validation DataLoader.

        Returns:
            torch.utils.data.DataLoader: DataLoader for validation samples.
        """
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Build the test DataLoader.

        Returns:
            torch.utils.data.DataLoader: DataLoader for test samples.
        """
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        """Build the prediction DataLoader.

        Returns:
            torch.utils.data.DataLoader: DataLoader using a custom collate function for patch metadata.
        """
        return DataLoader(
            self.predict_data,
            batch_size=self.batch_size,
            collate_fn=CorrosionDataModule.predict_coallate_function,
            shuffle=False,
            num_workers=self.num_workers,
        )
