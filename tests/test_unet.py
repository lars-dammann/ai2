import pytest
import torch
import torch.nn as nn

from model.unet import DoubleConv, UNet


class TestDoubleConv:
    """Test suite for DoubleConv module"""

    @pytest.fixture
    def double_conv_layer(self):
        """Create a DoubleConv layer for testing"""
        return DoubleConv(in_channels=3, out_channels=64)

    def test_double_conv_initialization(self, double_conv_layer):
        """Test that DoubleConv initializes correctly"""
        assert isinstance(double_conv_layer, nn.Module)
        assert hasattr(double_conv_layer, "net")
        assert hasattr(double_conv_layer, "groups")

    def test_double_conv_forward_pass(self, double_conv_layer):
        """Test forward pass through DoubleConv"""
        x = torch.randn(2, 3, 64, 64)  # Batch size 2, 3 channels, 64x64
        output = double_conv_layer(x)

        assert output.shape == (2, 64, 64, 64)  # Output channels = 64
        assert output.dtype == x.dtype

    def test_double_conv_output_channels_correct(self):
        """Test that output has correct number of channels"""
        in_channels = 4
        out_channels = 128
        layer = DoubleConv(in_channels, out_channels)

        x = torch.randn(1, in_channels, 32, 32)
        output = layer(x)

        assert output.shape[1] == out_channels

    def test_double_conv_preserves_spatial_dimensions(self):
        """Test that spatial dimensions are preserved with padding='same'"""
        layer = DoubleConv(3, 64)

        for size in [32, 64, 128, 256]:
            x = torch.randn(1, 3, size, size)
            output = layer(x)
            assert output.shape[-2:] == (size, size)

    def test_double_conv_group_norm_configuration(self):
        """Test that GroupNorm is configured correctly"""
        layer = DoubleConv(3, 64)
        # Check that the network has GroupNorm layers
        has_group_norm = any(isinstance(m, nn.GroupNorm) for m in layer.modules())
        assert has_group_norm

    def test_double_conv_relu_activation(self):
        """Test that DoubleConv uses ReLU activation"""
        layer = DoubleConv(3, 64)
        has_relu = any(isinstance(m, nn.ReLU) for m in layer.modules())
        assert has_relu

    def test_double_conv_with_various_channel_sizes(self):
        """Test DoubleConv with different channel configurations"""
        test_cases = [
            (1, 32),
            (3, 64),
            (64, 128),
            (256, 512),
            (512, 1024),
        ]

        for in_ch, out_ch in test_cases:
            layer = DoubleConv(in_ch, out_ch)
            x = torch.randn(1, in_ch, 16, 16)
            output = layer(x)

            assert output.shape[1] == out_ch
            assert output.shape[-2:] == (16, 16)

    def test_double_conv_batch_processing(self):
        """Test DoubleConv with different batch sizes"""
        layer = DoubleConv(3, 64)

        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 3, 32, 32)
            output = layer(x)
            assert output.shape[0] == batch_size


class TestUNet:
    """Test suite for UNet model"""

    @pytest.fixture
    def unet_config(self):
        """Create a minimal UNet config"""
        return {
            "startfeature": 32,
            "udepth": 3,
            "lr": 0.001,
            "weight_decay": 1e-5,
        }

    @pytest.fixture
    def unet_model(self, unet_config):
        """Create a UNet model for testing"""
        return UNet(unet_config, in_channels=4, out_channels=1)

    def test_unet_initialization(self, unet_model):
        """Test that UNet initializes correctly"""
        assert isinstance(unet_model, nn.Module)
        assert hasattr(unet_model, "downconv")
        assert hasattr(unet_model, "upconv")
        assert hasattr(unet_model, "upsampling")
        assert hasattr(unet_model, "pool")
        assert hasattr(unet_model, "bottleneck")
        assert hasattr(unet_model, "final_conv")

    def test_unet_forward_pass(self, unet_model):
        """Test forward pass through UNet"""
        x = torch.randn(1, 4, 256, 256)
        output = unet_model(x)

        assert output.shape == (1, 1, 256, 256)
        assert output.dtype == x.dtype

    def test_unet_output_channels_correct(self, unet_model):
        """Test that output has correct number of channels"""
        x = torch.randn(2, 4, 128, 128)
        output = unet_model(x)

        assert output.shape[1] == 1  # out_channels = 1

    def test_unet_maintains_spatial_dimensions(self, unet_model):
        """Test that UNet maintains spatial dimensions"""
        test_sizes = [256, 512]

        for size in test_sizes:
            x = torch.randn(1, 4, size, size)
            output = unet_model(x)

            assert output.shape == (1, 1, size, size)

    def test_unet_with_different_depths(self):
        """Test UNet with different depth configurations"""
        config_base = {
            "startfeature": 32,
            "lr": 0.001,
            "weight_decay": 1e-5,
        }

        for depth in [2, 3, 4]:
            config = {**config_base, "udepth": depth}
            model = UNet(config, in_channels=4, out_channels=1)

            x = torch.randn(1, 4, 256, 256)
            output = model(x)

            assert output.shape == (1, 1, 256, 256)

    def test_unet_has_skip_connections(self, unet_model):
        """Test that UNet has skip connections by checking upconv layers"""
        # UNet should have skip connections if upconv exists
        assert len(unet_model.upconv) > 0
        assert len(unet_model.downconv) > 0

    def test_unet_residual_connection(self, unet_model):
        """Test that UNet applies residual connection from input height"""
        x = torch.randn(1, 4, 128, 128)
        output_with_residual = unet_model(x)
        output_without_residual = unet_model(torch.cat([x[:, :3], torch.zeros_like(x[:, -1:])], dim=1))

        assert output_with_residual.shape == output_without_residual.shape
        assert not torch.allclose(output_with_residual, output_without_residual)

    def test_unet_encoder_creates_skip_connections(self, unet_model):
        """Test that encoder path creates skip connections"""
        x = torch.randn(1, 4, 256, 256)

        # Track that downconv operations create features
        skip_count = len(unet_model.downconv)
        assert skip_count > 0

    def test_unet_decoder_uses_skip_connections(self, unet_model):
        """Test that decoder has upsampling and concatenation"""
        # Check that upsampling and upconv have same length
        assert len(unet_model.upsampling) == len(unet_model.upconv)

    def test_unet_with_batch_processing(self, unet_model):
        """Test UNet with different batch sizes"""
        for batch_size in [1, 2, 4]:
            x = torch.randn(batch_size, 4, 256, 256)
            output = unet_model(x)

            assert output.shape == (batch_size, 1, 256, 256)

    def test_unet_gradient_flow(self, unet_model):
        """Test that gradients flow through UNet"""
        x = torch.randn(1, 4, 128, 128, requires_grad=True)
        output = unet_model(x)
        loss = output.sum()
        loss.backward()

        # Check that input has gradients
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_unet_no_nan_output(self, unet_model):
        """Test that UNet doesn't produce NaN values"""
        x = torch.randn(2, 4, 256, 256)
        output = unet_model(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_unet_output_range(self, unet_model):
        """Test that UNet outputs are in reasonable range"""
        x = torch.randn(1, 4, 128, 128)
        output = unet_model(x)

        assert torch.isfinite(output).all()


class TestUNetConfiguration:
    """Test UNet configuration and architecture"""

    def test_unet_start_feature_affects_model_size(self):
        """Test that startfeature parameter affects model capacity"""
        config_small = {"startfeature": 16, "udepth": 2, "lr": 0.001, "weight_decay": 1e-5}
        config_large = {"startfeature": 64, "udepth": 2, "lr": 0.001, "weight_decay": 1e-5}

        model_small = UNet(config_small, in_channels=4, out_channels=1)
        model_large = UNet(config_large, in_channels=4, out_channels=1)

        # Count parameters
        params_small = sum(p.numel() for p in model_small.parameters())
        params_large = sum(p.numel() for p in model_large.parameters())

        # Larger config should have more parameters
        assert params_large > params_small

    def test_unet_depth_affects_layers(self):
        """Test that udepth affects number of layers"""
        config_shallow = {"startfeature": 32, "udepth": 2, "lr": 0.001, "weight_decay": 1e-5}
        config_deep = {"startfeature": 32, "udepth": 4, "lr": 0.001, "weight_decay": 1e-5}

        model_shallow = UNet(config_shallow, in_channels=4, out_channels=1)
        model_deep = UNet(config_deep, in_channels=4, out_channels=1)

        # Deeper model should have more layers
        shallow_layers = len(model_shallow.downconv)
        deep_layers = len(model_deep.downconv)

        assert deep_layers > shallow_layers

    def test_unet_input_output_channels_configurable(self):
        """Test that input and output channels are configurable"""
        config = {"startfeature": 32, "udepth": 2, "lr": 0.001, "weight_decay": 1e-5}

        # Test different input channels
        for in_ch in [1, 3, 4, 8]:
            model = UNet(config, in_channels=in_ch, out_channels=1)
            x = torch.randn(1, in_ch, 128, 128)
            output = model(x)
            assert output.shape[1] == 1

        # Test different output channels
        for out_ch in [1, 2, 4]:
            model = UNet(config, in_channels=4, out_channels=out_ch)
            x = torch.randn(1, 4, 128, 128)
            output = model(x)
            assert output.shape[1] == out_ch


class TestUNetEdgeCases:
    """Test edge cases for UNet"""

    def test_unet_with_minimum_size_input(self):
        """Test UNet with very small input"""
        config = {"startfeature": 32, "udepth": 2, "lr": 0.001, "weight_decay": 1e-5}
        model = UNet(config, in_channels=4, out_channels=1)

        # Very small input might not work well, but shouldn't crash
        x = torch.randn(1, 4, 64, 64)
        output = model(x)
        assert output.shape == (1, 1, 64, 64)

    def test_unet_with_rectangular_input(self):
        """Test UNet with non-square input"""
        config = {"startfeature": 32, "udepth": 2, "lr": 0.001, "weight_decay": 1e-5}
        model = UNet(config, in_channels=4, out_channels=1)

        x = torch.randn(1, 4, 128, 256)
        output = model(x)
        assert output.shape == (1, 1, 128, 256)

    def test_unet_eval_mode(self):
        """Test UNet in eval mode"""
        config = {"startfeature": 32, "udepth": 2, "lr": 0.001, "weight_decay": 1e-5}
        model = UNet(config, in_channels=4, out_channels=1)
        model.eval()

        x = torch.randn(1, 4, 128, 128)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 1, 128, 128)

    def test_unet_reproducibility(self):
        """Test that UNet produces reproducible outputs with fixed seed"""
        config = {"startfeature": 32, "udepth": 2, "lr": 0.001, "weight_decay": 1e-5}

        torch.manual_seed(42)
        model1 = UNet(config, in_channels=4, out_channels=1)
        x = torch.randn(1, 4, 128, 128)
        output1 = model1(x)

        torch.manual_seed(42)
        model2 = UNet(config, in_channels=4, out_channels=1)
        output2 = model2(x)

        assert torch.allclose(output1, output2)
