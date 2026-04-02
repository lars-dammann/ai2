import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import lightning as pl
from PIL import Image

from datamodule.datamodule import (
    CorrosionDataset,
    PredictCorrosionDataset,
    CorrosionDataModule,
)


@pytest.fixture
def minimal_config():
    """Minimal configuration for dataset testing"""
    return {
        "datadir": "data",
        "datasize": 64,
        "batch_size": 2,
        "normalization": {
            "before": {
                "image": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
                "height": {"mean": [0.0], "std": [1.0]},
            },
            "after": {
                "height": {"mean": [0.0], "std": [1.0]},
            },
        },
    }


@pytest.fixture
def sample_data_dir(tmp_path):
    """Create temporary directory with sample data"""
    def write_sample_tree(base_dir: Path):
        before_image_dir = base_dir / "before" / "image"
        before_height_dir = base_dir / "before" / "height"
        after_height_dir = base_dir / "after" / "height"
        mask_dir = base_dir / "mask"

        for directory in [before_image_dir, before_height_dir, after_height_dir, mask_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        for sample_id in range(3):
            image = Image.new("RGB", (128, 128), color=(100, 150, 200))
            image.save(before_image_dir / f"sample_{sample_id:03d}.png")

            before_height = np.random.randn(128, 128).astype(np.float32) * 0.1
            after_height = np.random.randn(128, 128).astype(np.float32) * 0.1
            np.save(before_height_dir / f"sample_{sample_id:03d}.npy", before_height)
            np.save(after_height_dir / f"sample_{sample_id:03d}.npy", after_height)

            mask = np.random.rand(128, 128) > 0.3
            np.save(mask_dir / f"sample_{sample_id:03d}.npy", mask)

    write_sample_tree(tmp_path)
    for split in ["train", "val", "test"]:
        write_sample_tree(tmp_path / split)

    return tmp_path


class TestCorrosionDataset:
    """Test suite for CorrosionDataset"""

    def test_dataset_initialization(self, sample_data_dir, minimal_config):
        """Test CorrosionDataset initialization"""
        dataset = CorrosionDataset(sample_data_dir, minimal_config)

        assert isinstance(dataset, torch.utils.data.Dataset)
        assert dataset.data_dir == sample_data_dir
        assert len(dataset.sample_ids) == 3

    def test_dataset_length(self, sample_data_dir, minimal_config):
        """Test that dataset length matches number of samples"""
        dataset = CorrosionDataset(sample_data_dir, minimal_config)
        assert len(dataset) == 3

    def test_dataset_getitem_returns_tuple(self, sample_data_dir, minimal_config):
        """Test that __getitem__ returns tuple of (before, after, mask)"""
        dataset = CorrosionDataset(sample_data_dir, minimal_config)
        item = dataset[0]

        assert isinstance(item, tuple)
        assert len(item) == 3
        before, after, mask = item
        assert isinstance(before, torch.Tensor)
        assert isinstance(after, torch.Tensor)
        assert isinstance(mask, torch.Tensor)

    def test_dataset_getitem_shapes(self, sample_data_dir, minimal_config):
        """Test that dataset items have correct shapes"""
        dataset = CorrosionDataset(sample_data_dir, minimal_config)
        before, after, mask = dataset[0]

        # before should be (4, H, W): 3 RGB + 1 height
        assert before.shape[0] == 4
        assert before.shape[1:] == after.shape[1:]

        # after should be (1, H, W): 1 height
        assert after.shape[0] == 1

        # mask should be (1, H, W)
        assert mask.shape[0] == 1

    def test_dataset_getitem_dtypes(self, sample_data_dir, minimal_config):
        """Test that dataset items have correct dtypes"""
        dataset = CorrosionDataset(sample_data_dir, minimal_config)
        before, after, mask = dataset[0]

        assert before.dtype == torch.float32
        assert after.dtype == torch.float32
        assert mask.dtype == torch.bool

    def test_dataset_with_transform(self, sample_data_dir, minimal_config):
        """Test dataset with transform applied"""
        from torchvision.transforms import v2

        transform = v2.Compose(
            [
                v2.RandomCrop(64, pad_if_needed=True),
                v2.RandomHorizontalFlip(),
            ]
        )

        dataset = CorrosionDataset(sample_data_dir, minimal_config, transform=transform)
        before, after, mask = dataset[0]

        # Shapes should match due to RandomCrop
        assert before.shape[1:] == (64, 64)
        assert after.shape[1:] == (64, 64)
        assert mask.shape[1:] == (64, 64)

    def test_dataset_with_photometric_transform(self, sample_data_dir, minimal_config):
        """Test dataset with photometric transform"""
        from torchvision.transforms import v2

        photometric_transform = v2.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01
        )

        dataset = CorrosionDataset(
            sample_data_dir,
            minimal_config,
            photometric_transform=photometric_transform,
        )
        before, after, mask = dataset[0]

        # Should still have valid values
        assert not torch.isnan(before).any()
        assert before.dtype == torch.float32

    def test_dataset_normalizer_created(self, sample_data_dir, minimal_config):
        """Test that normalizer is created from config"""
        dataset = CorrosionDataset(sample_data_dir, minimal_config)

        assert dataset.normalizer is not None
        expected_channels = (
            len(minimal_config["normalization"]["before"]["image"]["mean"])
            + len(minimal_config["normalization"]["before"]["height"]["mean"])
            + len(minimal_config["normalization"]["after"]["height"]["mean"])
        )
        assert dataset.normalizer.channels == expected_channels

    def test_dataset_all_samples_accessible(self, sample_data_dir, minimal_config):
        """Test that all samples in dataset are accessible"""
        dataset = CorrosionDataset(sample_data_dir, minimal_config)

        for idx in range(len(dataset)):
            before, after, mask = dataset[idx]

            assert before.shape[0] == 4
            assert after.shape[0] == 1
            assert mask.shape[0] == 1

    def test_dataset_masked_areas_consistent(self, sample_data_dir, minimal_config):
        """Test that masked areas are consistently handled"""
        dataset = CorrosionDataset(sample_data_dir, minimal_config)

        # Get same sample twice
        before1, after1, mask1 = dataset[0]
        before2, after2, mask2 = dataset[0]

        # Should be identical (no randomness without transform)
        torch.testing.assert_close(before1, before2)
        torch.testing.assert_close(mask1, mask2)


class TestPredictCorrosionDataset:
    """Test suite for PredictCorrosionDataset"""

    def test_predict_dataset_initialization(self, sample_data_dir, minimal_config):
        """Test PredictCorrosionDataset initialization"""
        dataset = PredictCorrosionDataset(
            sample_data_dir, minimal_config, datasize=64, reconstruction_overlap=0
        )

        assert isinstance(dataset, CorrosionDataset)
        assert dataset.datasize == 64
        assert dataset.reconstruction_overlap == 0

    def test_predict_dataset_getitem_returns_tuple(
        self, sample_data_dir, minimal_config
    ):
        """Test that __getitem__ returns tuple of (patches, info)"""
        dataset = PredictCorrosionDataset(
            sample_data_dir, minimal_config, datasize=64, reconstruction_overlap=0
        )
        item = dataset[0]

        assert isinstance(item, tuple)
        assert len(item) == 2
        patches, info = item
        assert isinstance(patches, torch.Tensor)
        assert isinstance(info, dict)

    def test_predict_dataset_patch_shape(self, sample_data_dir, minimal_config):
        """Test that predicted patches have correct shape"""
        dataset = PredictCorrosionDataset(
            sample_data_dir, minimal_config, datasize=64, reconstruction_overlap=0
        )
        patches, info = dataset[0]

        # patches should be (n_patches, 4, 64, 64)
        assert patches.dim() == 4
        assert patches.shape[1] == 4  # 3 RGB + 1 height
        assert patches.shape[2:] == (64, 64)

    def test_predict_dataset_info_dict(self, sample_data_dir, minimal_config):
        """Test that info dict contains expected keys"""
        dataset = PredictCorrosionDataset(
            sample_data_dir, minimal_config, datasize=64, reconstruction_overlap=0
        )
        patches, info = dataset[0]

        sample_id = dataset.sample_ids[0]
        assert sample_id in info
        assert "positions" in info[sample_id]
        assert "total_number_patches" in info[sample_id]
        assert "imageshape" in info[sample_id]
        assert "normalization" in info

    def test_predict_dataset_with_overlap(self, sample_data_dir, minimal_config):
        """Test PredictCorrosionDataset with reconstruction overlap"""
        dataset = PredictCorrosionDataset(
            sample_data_dir, minimal_config, datasize=64, reconstruction_overlap=16
        )
        patches, info = dataset[0]

        # Should work with overlap
        assert patches.shape[1] == 4
        assert patches.shape[2:] == (64, 64)

    def test_predict_dataset_patch_positions(self, sample_data_dir, minimal_config):
        """Test that patch positions are stored correctly"""
        dataset = PredictCorrosionDataset(
            sample_data_dir, minimal_config, datasize=64, reconstruction_overlap=0
        )
        patches, info = dataset[0]

        sample_id = dataset.sample_ids[0]
        positions = info[sample_id]["positions"]

        assert isinstance(positions, list)
        assert len(positions) > 0
        # Each position should be a tuple of (top, left)
        for pos in positions:
            assert isinstance(pos, tuple)
            assert len(pos) == 2

    def test_predict_dataset_fewer_patches_than_corrosion_dataset(
        self, sample_data_dir, minimal_config
    ):
        """Test that PredictCorrosionDataset returns different format than CorrosionDataset"""
        corrosion_dataset = CorrosionDataset(sample_data_dir, minimal_config)
        predict_dataset = PredictCorrosionDataset(
            sample_data_dir, minimal_config, datasize=64, reconstruction_overlap=0
        )

        corrosion_item = corrosion_dataset[0]
        predict_item = predict_dataset[0]

        # Different return types
        assert len(corrosion_item) == 3  # (before, after, mask)
        assert len(predict_item) == 2  # (patches, info)

    def test_predict_dataset_normalizer_only_before_after(
        self, sample_data_dir, minimal_config
    ):
        """Test that PredictCorrosionDataset normalizer doesn't include after height"""
        dataset = PredictCorrosionDataset(
            sample_data_dir, minimal_config, datasize=64, reconstruction_overlap=0
        )

        # Predict dataset should have 4 channels (3 RGB + 1 before height)
        assert dataset.normalizer.channels == 4


class TestCorrosionDataModule:
    """Test suite for CorrosionDataModule (Lightning DataModule)"""

    def test_datamodule_initialization(self, minimal_config):
        """Test CorrosionDataModule initialization"""
        dm = CorrosionDataModule(minimal_config)

        assert isinstance(dm, pl.LightningDataModule)
        assert dm.batch_size == minimal_config["batch_size"]
        assert dm.data_size == minimal_config["datasize"]

    def test_datamodule_num_workers_default(self, minimal_config, monkeypatch):
        """Test that num_workers defaults to 2 when SLURM env var missing"""
        monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
        dm = CorrosionDataModule(minimal_config)
        assert dm.num_workers == 2

    def test_datamodule_num_workers_from_slurm(self, minimal_config, monkeypatch):
        """Test that num_workers is set from SLURM_CPUS_PER_TASK"""
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
        dm = CorrosionDataModule(minimal_config)
        assert dm.num_workers == 8

    def test_datamodule_random_rot90(self):
        """Test random_rot90 augmentation"""
        img = torch.arange(9).reshape(1, 3, 3).float()

        # Should produce one of 4 rotations
        rotated = CorrosionDataModule.random_rot90(img)

        assert rotated.shape == img.shape

    def test_datamodule_predict_collate_function(self):
        """Test predict_coallate_function combines batches correctly"""
        first_tensor = torch.ones((2, 4, 16, 16), dtype=torch.float32)
        second_tensor = torch.zeros((1, 4, 16, 16), dtype=torch.float32)

        batch = [
            (first_tensor, {"sample_a": {"positions": [(0, 0)]}}),
            (second_tensor, {"sample_b": {"positions": [(8, 8)]}}),
        ]

        data, merged_info = CorrosionDataModule.predict_coallate_function(batch)

        assert data.shape == (3, 4, 16, 16)
        assert "sample_a" in merged_info
        assert "sample_b" in merged_info

    def test_datamodule_setup_fit_stage(self, sample_data_dir, minimal_config):
        """Test datamodule setup for fit stage"""
        config = minimal_config.copy()
        config["datadir"] = str(sample_data_dir)

        dm = CorrosionDataModule(config)
        dm.setup(stage="fit")

        assert hasattr(dm, "train_data")
        assert hasattr(dm, "val_data")

    def test_datamodule_setup_test_stage(self, sample_data_dir, minimal_config):
        """Test datamodule setup for test stage"""
        config = minimal_config.copy()
        config["datadir"] = str(sample_data_dir)

        dm = CorrosionDataModule(config)
        dm.setup(stage="test")

        assert hasattr(dm, "test_data")

    def test_datamodule_setup_predict_stage(self, sample_data_dir, minimal_config):
        """Test datamodule setup for predict stage"""
        config = minimal_config.copy()
        config["datadir"] = str(sample_data_dir)

        dm = CorrosionDataModule(config, predict_dataset="test")
        dm.setup(stage="predict")

        assert hasattr(dm, "predict_data")

    def test_datamodule_train_dataloader(self, sample_data_dir, minimal_config):
        """Test train_dataloader returns DataLoader"""
        config = minimal_config.copy()
        config["datadir"] = str(sample_data_dir)
        config["batch_size"] = 1

        dm = CorrosionDataModule(config)
        dm.setup(stage="fit")

        train_loader = dm.train_dataloader()

        assert isinstance(train_loader, torch.utils.data.DataLoader)
        assert train_loader.batch_size == 1

    def test_datamodule_val_dataloader(self, sample_data_dir, minimal_config):
        """Test val_dataloader returns DataLoader"""
        config = minimal_config.copy()
        config["datadir"] = str(sample_data_dir)
        config["batch_size"] = 1

        dm = CorrosionDataModule(config)
        dm.setup(stage="fit")

        val_loader = dm.val_dataloader()

        assert isinstance(val_loader, torch.utils.data.DataLoader)

    def test_datamodule_test_dataloader(self, sample_data_dir, minimal_config):
        """Test test_dataloader returns DataLoader"""
        config = minimal_config.copy()
        config["datadir"] = str(sample_data_dir)
        config["batch_size"] = 1

        dm = CorrosionDataModule(config)
        dm.setup(stage="test")

        test_loader = dm.test_dataloader()

        assert isinstance(test_loader, torch.utils.data.DataLoader)

    def test_datamodule_predict_dataloader(self, sample_data_dir, minimal_config):
        """Test predict_dataloader returns DataLoader with custom collate"""
        config = minimal_config.copy()
        config["datadir"] = str(sample_data_dir)
        config["batch_size"] = 1

        dm = CorrosionDataModule(config, predict_dataset="test")
        dm.setup(stage="predict")

        predict_loader = dm.predict_dataloader()

        assert isinstance(predict_loader, torch.utils.data.DataLoader)
        # Should have custom collate function
        assert (
            predict_loader.collate_fn
            == CorrosionDataModule.predict_coallate_function
        )

    def test_datamodule_initialization_saves_hyperparameters(self, minimal_config):
        """Test that datamodule saves hyperparameters (Lightning feature)"""
        dm = CorrosionDataModule(minimal_config)
        # Lightning datamodule should have hparams
        assert hasattr(dm, "hparams") or hasattr(dm, "datamodule_config")


class TestDataModuleIntegration:
    """Integration tests for datamodule"""

    def test_full_training_loop_simulation(self, sample_data_dir, minimal_config):
        """Test simulated training loop with datamodule"""
        config = minimal_config.copy()
        config["datadir"] = str(sample_data_dir)
        config["batch_size"] = 2

        dm = CorrosionDataModule(config)
        dm.num_workers = 0
        dm.setup(stage="fit")

        train_loader = dm.train_dataloader()

        # Simulate a few batches
        for batch_idx, (before, after, mask) in enumerate(train_loader):
            assert before.shape[0] <= 2  # batch size
            assert before.shape[1] == 4
            assert after.shape[1] == 1
            assert mask.shape[1] == 1

            if batch_idx >= 2:
                break

    def test_multiple_stages_setup(self, sample_data_dir, minimal_config):
        """Test setting up multiple stages sequentially"""
        config = minimal_config.copy()
        config["datadir"] = str(sample_data_dir)

        dm = CorrosionDataModule(config)

        # Setup multiple stages
        dm.setup(stage="fit")
        assert hasattr(dm, "train_data")

        dm.setup(stage="test")
        assert hasattr(dm, "test_data")

        dm.setup(stage="predict")
        assert hasattr(dm, "predict_data")
