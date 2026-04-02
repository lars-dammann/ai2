import numpy as np
import pytest
import torch

from utils.nomalizer import Normalizer


class TestNormalizer:
    """Test suite for Normalizer class"""

    @pytest.fixture
    def normalizer(self):
        """Create a normalizer with known mean and std"""
        mean = np.array([0.5, 0.5, 0.5, 0.0], dtype=np.float32)
        std = np.array([0.1, 0.2, 0.15, 1.0], dtype=np.float32)
        return Normalizer(mean, std)

    @pytest.fixture
    def sample_data(self):
        """Create sample data tensor (4 channels, 32x32)"""
        return torch.randn((4, 32, 32), dtype=torch.float32)

    @pytest.fixture
    def sample_mask(self):
        """Create sample mask tensor (1 channel, 32x32)"""
        mask = torch.ones((1, 32, 32), dtype=torch.bool)
        # Set some True values (masked areas)
        mask[:, :10, :10] = True
        return mask

    def test_initializer_creates_normalizer(self, normalizer):
        """Test that Normalizer initializes correctly"""
        assert normalizer.channels == 4
        assert len(normalizer.mean) == 4
        assert len(normalizer.std) == 4
        assert np.allclose(normalizer.mean, [0.5, 0.5, 0.5, 0.0])

    def test_normalize_returns_correct_shape(self, normalizer, sample_data):
        """Test that normalize returns tensor with same shape"""
        result = normalizer.normalize(sample_data)
        assert result.shape == sample_data.shape

    def test_normalize_applies_formula_correctly(self):
        """Test that normalize applies (x - mean) / std correctly"""
        mean = np.array([2.0, 2.0], dtype=np.float32)
        std = np.array([2.0, 2.0], dtype=np.float32)
        normalizer = Normalizer(mean, std)

        data = torch.tensor([[[4.0, 6.0]]], dtype=torch.float32)  # shape (1, 1, 2)
        data = data.reshape(1, 1, 2)
        result = normalizer.normalize(data.expand(2, 1, 2))  # Make it 2 channels

        # Expected: (4-2)/2 = 1, (6-2)/2 = 2
        expected = torch.tensor([[[1.0, 2.0]]], dtype=torch.float32)
        expected = expected.expand(2, 1, 2)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_denormalize_inverts_normalize(self, normalizer, sample_data):
        """Test that denormalize is the inverse of normalize"""
        normalized = normalizer.normalize(sample_data)
        denormalized = normalizer.denormalize(normalized)
        assert torch.allclose(denormalized, sample_data, atol=1e-6)

    def test_normalize_masked_preserves_masked_areas(self, normalizer, sample_data, sample_mask):
        """Test that normalize_masked keeps masked areas unchanged"""
        original_data = sample_data.clone()
        result = normalizer.normalize_masked(sample_data, sample_mask)

        # Masked areas should be unchanged
        assert torch.equal(result[:, :10, :10], original_data[:, :10, :10])

    def test_denormalize_masked_preserves_masked_areas(self, normalizer, sample_data, sample_mask):
        """Test that denormalize_masked keeps masked areas unchanged"""
        # First normalize
        normalized = normalizer.normalize_masked(sample_data, sample_mask)
        original_normalized = normalized.clone()

        # Then denormalize
        denormalized = normalizer.denormalize_masked(normalized, sample_mask)

        # Masked areas should be unchanged
        assert torch.equal(denormalized[:, :10, :10], original_normalized[:, :10, :10])

    def test_normalize_masked_then_denormalize_masked_roundtrip(self, normalizer, sample_data, sample_mask):
        """Test full roundtrip: original -> normalize -> denormalize -> original"""
        original_data = sample_data.clone()

        normalized = normalizer.normalize_masked(sample_data, sample_mask)
        denormalized = normalizer.denormalize_masked(normalized, sample_mask)

        # Unmasked areas should match
        assert torch.allclose(denormalized[:, 10:, 10:], original_data[:, 10:, 10:], atol=1e-5)

    def test_normalize_with_zero_std_raises_error(self, sample_data):
        """Test that zero std values cause issues (would divide by zero)"""
        mean = np.array([0.0, 0.0], dtype=np.float32)
        std = np.array([0.0, 0.0], dtype=np.float32)
        normalizer = Normalizer(mean, std)

        data = torch.ones((2, 32, 32), dtype=torch.float32)
        with pytest.warns(RuntimeWarning, match="divide by zero"):
            result = normalizer.normalize(data)

        # Result should contain inf values
        assert torch.isinf(result).any()

    def test_normalize_masked_with_empty_mask(self, normalizer, sample_data):
        """Test normalize_masked when entire area is masked"""
        mask = torch.ones((1, 32, 32), dtype=torch.bool)
        original = sample_data.clone()

        result = normalizer.normalize_masked(sample_data, mask)

        # Everything should be unchanged since everything is masked
        assert torch.equal(result, original)

    def test_normalize_masked_only_changes_unmasked_values(self):
        """Test that normalize_masked leaves masked values untouched and normalizes others"""
        mean = np.array([1.0, 2.0], dtype=np.float32)
        std = np.array([1.0, 2.0], dtype=np.float32)
        normalizer = Normalizer(mean, std)

        data = torch.tensor(
            [
                [[1.0, 3.0]],
                [[2.0, 6.0]],
            ],
            dtype=torch.float32,
        )
        mask = torch.tensor([[[True, False]]], dtype=torch.bool)

        result = normalizer.normalize_masked(data, mask)

        assert torch.equal(result[:, :, 0], data[:, :, 0])
        assert torch.allclose(result[:, :, 1], torch.tensor([[2.0], [2.0]]))

    def test_channels_attribute_set_correctly(self):
        """Test that channels attribute matches mean length"""
        for num_channels in [1, 3, 4, 8]:
            mean = np.zeros(num_channels, dtype=np.float32)
            std = np.ones(num_channels, dtype=np.float32)
            normalizer = Normalizer(mean, std)
            assert normalizer.channels == num_channels


class TestNormalizerEdgeCases:
    """Test edge cases for Normalizer"""

    def test_single_channel_normalizer(self):
        """Test normalizer with single channel"""
        mean = np.array([0.5], dtype=np.float32)
        std = np.array([0.2], dtype=np.float32)
        normalizer = Normalizer(mean, std)

        data = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float32)
        result = normalizer.normalize(data)

        assert result.shape == data.shape

    def test_large_tensor_normalization(self):
        """Test normalizer with large tensors"""
        mean = np.array([0.5, 0.5], dtype=np.float32)
        std = np.array([0.1, 0.1], dtype=np.float32)
        normalizer = Normalizer(mean, std)

        large_data = torch.randn((2, 512, 512), dtype=torch.float32)
        result = normalizer.normalize(large_data)

        assert result.shape == large_data.shape
        assert not torch.isnan(result).any()

    def test_normalize_with_very_small_std(self):
        """Test normalization with very small std values"""
        mean = np.array([0.0, 0.0], dtype=np.float32)
        std = np.array([1e-7, 1e-7], dtype=np.float32)
        normalizer = Normalizer(mean, std)

        data = torch.ones((2, 8, 8), dtype=torch.float32)
        result = normalizer.normalize(data)

        # Results should be very large due to division by tiny std
        assert (torch.abs(result) > 1e6).any()
