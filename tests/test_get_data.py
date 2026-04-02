import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from utils.get_data import get_mask, get_image, get_height, load_config, get_config


class TestLoadConfig:
    """Test suite for load_config function"""

    def test_load_default_config(self):
        """Test loading the repository default config file"""
        config = load_config(None)

        assert isinstance(config, dict)
        assert "datamodule" in config
        assert "unet" in config

    def test_load_config_with_specific_file(self):
        """Test loading a named config file from the configs directory"""
        default_config = load_config(None)
        named_config = load_config("configs.json")

        assert named_config == default_config


class TestGetConfig:
    """Test suite for get_config function"""

    def test_get_config_returns_dict(self):
        """Test that get_config returns a dictionary"""
        test_config = {
            "model": {"lr": 0.001},
            "data": {"batch_size": 32}
        }
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr("utils.get_data.load_config", lambda file=None: test_config)
            result = get_config()

        assert result == test_config

    def test_get_config_with_none_new_config(self):
        """Test get_config returns original config when new_config is None"""
        test_config = {"param1": 1, "param2": 2}
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr("utils.get_data.load_config", lambda file=None: test_config)
            result = get_config(new_config=None)

        assert result == test_config

    def test_get_config_overwrites_nested_values(self):
        """Test that new_config can overwrite nested values"""
        test_config = {
            "model": {"lr": 0.001, "depth": 3},
            "data": {"batch_size": 32}
        }
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr("utils.get_data.load_config", lambda file=None: test_config)
            result = get_config(new_config={"lr": 0.01})

        assert result == {
            "model": {"lr": 0.01, "depth": 3},
            "data": {"batch_size": 32},
        }

    def test_get_config_adds_new_keys(self):
        """Test that get_config can add new keys"""
        test_config = {"existing": "value"}
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr("utils.get_data.load_config", lambda file=None: test_config)
            result = get_config(new_config={"new_key": "new_value"})

        assert result == {"existing": "value", "new_key": "new_value"}


class TestGetMask:
    """Test suite for get_mask function"""

    def test_get_mask_loads_npy_and_returns_tensor(self):
        """Test that get_mask loads numpy file and returns tensor"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create a sample mask file
            mask_data = np.array([[True, False], [False, True]], dtype=bool)
            np.save(tmpdir_path / "sample_001.npy", mask_data)

            result = get_mask(tmpdir_path, "sample_001")

            assert isinstance(result, torch.Tensor)
            assert result.shape == (1, 2, 2)
            assert result.dtype == torch.bool

    def test_get_mask_adds_single_channel_dimension(self):
        """Test that get_mask adds a channel dimension using np.newaxis"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create a 2D mask
            mask_data = np.random.rand(64, 64) > 0.5
            np.save(tmpdir_path / "test_mask.npy", mask_data)

            result = get_mask(tmpdir_path, "test_mask")

            # Should have shape (1, 64, 64)
            assert result.shape == (1, 64, 64)
            assert result.dim() == 3

    def test_get_mask_preserves_values(self):
        """Test that mask values are preserved correctly"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create a specific pattern
            mask_data = np.array(
                [[True, False, True], [False, True, False]], dtype=bool
            )
            np.save(tmpdir_path / "pattern.npy", mask_data)

            result = get_mask(tmpdir_path, "pattern")

            assert torch.all(result[0, 0] == torch.tensor([True, False, True]))
            assert torch.all(result[0, 1] == torch.tensor([False, True, False]))


class TestGetImage:
    """Test suite for get_image function"""

    def test_get_image_reads_and_converts_to_tensor(self):
        """Test that get_image reads image and converts to tensor"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create a simple test image
            from PIL import Image
            img = Image.new("RGB", (32, 32), color=(100, 150, 200))
            img.save(tmpdir_path / "test_image.png")

            result = get_image(tmpdir_path, "test_image")

            assert isinstance(result, torch.Tensor)
            assert result.dtype == torch.float32
            assert result.dim() == 3  # Should have 3 dimensions (channels, height, width)
            assert result.shape[0] == 3  # RGB = 3 channels

    def test_get_image_scales_to_float32_range(self):
        """Test that image values are scaled to float32"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            from PIL import Image
            # Create image with known values
            img = Image.new("RGB", (16, 16), color=(255, 128, 0))
            img.save(tmpdir_path / "scaled_image.png")

            result = get_image(tmpdir_path, "scaled_image")

            # Values should be in float range (0-1 or 0-255 depending on implementation)
            assert result.min() >= 0
            assert result.dtype == torch.float32

    def test_get_image_with_different_sizes(self):
        """Test get_image with different image dimensions"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            from PIL import Image
            for size in [(32, 32), (64, 128), (256, 256)]:
                img = Image.new("RGB", size, color=(100, 100, 100))
                img.save(tmpdir_path / f"image_{size[0]}x{size[1]}.png")

                result = get_image(tmpdir_path, f"image_{size[0]}x{size[1]}")

                assert result.shape == (3, size[1], size[0])


class TestGetHeight:
    """Test suite for get_height function"""

    def test_get_height_loads_npy_and_returns_tensor(self):
        """Test that get_height loads numpy file and returns tensor"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create height profile
            height_data = np.random.randn(32, 32).astype(np.float32)
            np.save(tmpdir_path / "height_001.npy", height_data)

            result = get_height(tmpdir_path, "height_001")

            assert isinstance(result, torch.Tensor)
            assert result.dtype == torch.float32
            assert result.shape == (1, 32, 32)

    def test_get_height_adds_channel_dimension(self):
        """Test that get_height adds channel dimension correctly"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            height_data = np.ones((64, 64), dtype=np.float32) * 1.5
            np.save(tmpdir_path / "heights.npy", height_data)

            result = get_height(tmpdir_path, "heights")

            assert result.shape == (1, 64, 64)
            assert result.dim() == 3

    def test_get_height_preserves_values(self):
        """Test that height values are preserved correctly"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            height_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            np.save(tmpdir_path / "test_height.npy", height_data)

            result = get_height(tmpdir_path, "test_height")

            expected = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
            assert torch.allclose(result, expected)

    def test_get_height_converts_to_float32(self):
        """Test that height data is converted to float32"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Save as float64
            height_data = np.random.randn(16, 16).astype(np.float64)
            np.save(tmpdir_path / "float64_height.npy", height_data)

            result = get_height(tmpdir_path, "float64_height")

            assert result.dtype == torch.float32


class TestDataLoadingIntegration:
    """Integration tests for data loading functions"""

    def test_load_image_and_height_together(self):
        """Test loading image and height together as a corrosion sample"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            from PIL import Image
            img = Image.new("RGB", (64, 64), color=(128, 128, 128))
            img.save(tmpdir_path / "sample_001.png")

            height_data = np.random.randn(64, 64).astype(np.float32)
            np.save(tmpdir_path / "sample_001.npy", height_data)

            image = get_image(tmpdir_path, "sample_001")
            height = get_height(tmpdir_path, "sample_001")

            # Should be able to concatenate them
            combined = torch.cat((image, height), dim=0)
            assert combined.shape == (4, 64, 64)  # 3 RGB + 1 height

    def test_load_complete_sample_with_mask(self):
        """Test loading a complete sample with image, height, and mask"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create image
            from PIL import Image
            img = Image.new("RGB", (32, 32), color=(100, 100, 100))
            img.save(tmpdir_path / "complete.png")

            # Create height
            height_data = np.random.randn(32, 32).astype(np.float32)
            np.save(tmpdir_path / "complete.npy", height_data)

            # Create mask
            mask_data = np.random.rand(32, 32) > 0.5
            np.save(tmpdir_path / "complete_mask.npy", mask_data)

            image = get_image(tmpdir_path, "complete")
            height = get_height(tmpdir_path, "complete")
            mask = get_mask(tmpdir_path, "complete_mask")

            assert image.shape[1:] == height.shape[1:] == mask.shape[1:]
            assert image.shape == (3, 32, 32)
            assert height.shape == (1, 32, 32)
            assert mask.shape == (1, 32, 32)
