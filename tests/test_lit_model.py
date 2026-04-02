import pytest
import torch
import torch.nn as nn

from model.lit_model import CorrosionUNet


@pytest.fixture
def model_config():
    """Configuration for CorrosionUNet"""
    return {
        "startfeature": 32,
        "udepth": 2,
        "lr": 0.001,
        "weight_decay": 1e-5,
    }


@pytest.fixture
def corrosion_unet(model_config):
    """Create a CorrosionUNet model for testing"""
    model = CorrosionUNet(model_config)
    model.log = lambda *args, **kwargs: None
    return model


@pytest.fixture
def sample_batch():
    """Create a sample batch for testing"""
    x = torch.randn(2, 4, 128, 128)
    y = torch.randn(2, 1, 128, 128)
    mask = torch.randint(0, 2, (2, 1, 128, 128), dtype=torch.bool)
    return (x, y, mask)


class TestCorrosionUNetBasics:
    """Test basic functionality of CorrosionUNet"""

    def test_initialization(self, corrosion_unet, model_config):
        """Test CorrosionUNet initialization"""
        assert isinstance(corrosion_unet, torch.nn.Module)
        assert hasattr(corrosion_unet, "model")
        assert corrosion_unet.learning_rate == model_config["lr"]
        assert corrosion_unet.weight_decay == model_config["weight_decay"]

    def test_forward_pass(self, corrosion_unet):
        """Test forward pass through CorrosionUNet"""
        x = torch.randn(2, 4, 128, 128)
        output = corrosion_unet(x)

        assert output.shape == (2, 1, 128, 128)
        assert not torch.isnan(output).any()

    def test_forward_maintains_dtype(self, corrosion_unet):
        """Test that forward pass maintains dtype"""
        x = torch.randn(2, 4, 128, 128, dtype=torch.float32)
        output = corrosion_unet(x)

        assert output.dtype == torch.float32

    def test_has_unet_inside(self, corrosion_unet):
        """Test that CorrosionUNet contains a UNet model"""
        from model.unet import UNet

        assert isinstance(corrosion_unet.model, UNet)

    def test_save_hyperparameters(self, corrosion_unet):
        """Test that hyperparameters are saved"""
        assert hasattr(corrosion_unet, "hparams")


class TestCorrosionUNetLosses:
    """Test loss calculation methods"""

    def test_calc_losses_returns_dict(self, corrosion_unet, sample_batch):
        """Test that _calc_losses returns a dictionary"""
        x, y_target, mask = sample_batch
        y_pred = corrosion_unet(x)

        losses = corrosion_unet._calc_losses(y_pred, y_target, mask)

        assert isinstance(losses, dict)
        assert "mae-loss" in losses
        assert "mse-loss" in losses
        assert "r2score" in losses
        assert "corr" in losses
        assert "volume-loss-abs" in losses

    def test_mae_loss_is_scalar(self, corrosion_unet, sample_batch):
        """Test that MAE loss is a scalar"""
        x, y_target, mask = sample_batch
        y_pred = corrosion_unet(x)

        losses = corrosion_unet._calc_losses(y_pred, y_target, mask)

        assert losses["mae-loss"].dim() == 0
        assert isinstance(losses["mae-loss"].item(), float)

    def test_mse_loss_is_scalar(self, corrosion_unet, sample_batch):
        """Test that MSE loss is a scalar"""
        x, y_target, mask = sample_batch
        y_pred = corrosion_unet(x)

        losses = corrosion_unet._calc_losses(y_pred, y_target, mask)

        assert losses["mse-loss"].dim() == 0

    def test_r2score_bounded(self, corrosion_unet, sample_batch):
        """Test that R2 score is bounded between -inf and 1"""
        x, y_target, mask = sample_batch
        y_pred = corrosion_unet(x)

        losses = corrosion_unet._calc_losses(y_pred, y_target, mask)

        r2 = losses["r2score"].item()
        assert r2 <= 1.0

    def test_correlation_bounded(self, corrosion_unet, sample_batch):
        """Test that correlation is bounded between -1 and 1"""
        x, y_target, mask = sample_batch
        y_pred = corrosion_unet(x)

        losses = corrosion_unet._calc_losses(y_pred, y_target, mask)

        corr = losses["corr"].item()
        assert -1.0 <= corr <= 1.0

    def test_losses_with_all_masked(self, corrosion_unet):
        """Test losses when all values are masked"""
        x = torch.randn(2, 4, 64, 64)
        y_target = torch.randn(2, 1, 64, 64)
        mask = torch.ones((2, 1, 64, 64), dtype=torch.bool)  # All masked

        y_pred = corrosion_unet(x)
        losses = corrosion_unet._calc_losses(y_pred, y_target, mask)

        # Should handle all masked case gracefully
        assert isinstance(losses, dict)

    def test_losses_with_no_masked(self, corrosion_unet):
        """Test losses when nothing is masked"""
        x = torch.randn(2, 4, 64, 64)
        y_target = torch.randn(2, 1, 64, 64)
        mask = torch.zeros((2, 1, 64, 64), dtype=torch.bool)  # Nothing masked

        y_pred = corrosion_unet(x)
        losses = corrosion_unet._calc_losses(y_pred, y_target, mask)

        assert isinstance(losses, dict)
        for loss in losses.values():
            assert not torch.isnan(loss)


class TestCorrosionUNetSegmentOperations:
    """Test segment calculation methods"""

    def test_calc_segment_sum(self, corrosion_unet):
        """Test _calc_segment_sum calculation"""
        data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        lengths = torch.tensor([2, 3])

        result = corrosion_unet._calc_segment_sum(data, lengths)

        assert result.shape == (2,)
        assert result[0].item() == 3.0  # 1+2
        assert result[1].item() == 12.0  # 3+4+5

    def test_calc_segment_mean(self, corrosion_unet):
        """Test _calc_segment_mean calculation"""
        data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        lengths = torch.tensor([2, 3])

        result = corrosion_unet._calc_segment_mean(data, lengths)

        assert result.shape == (2,)
        assert result[0].item() == 1.5  # (1+2)/2
        assert result[1].item() == 4.0  # (3+4+5)/3

    def test_calc_segment_std(self, corrosion_unet):
        """Test _calc_segment_std calculation"""
        data = torch.tensor([1.0, 1.0, 3.0, 4.0, 5.0])
        lengths = torch.tensor([2, 3])

        result = corrosion_unet._calc_segment_std(data, lengths)

        assert result.shape == (2,)
        assert result[0].item() >= 0
        assert result[1].item() >= 0

    def test_calc_segment_operations_with_batch(self, corrosion_unet):
        """Test segment operations with multiple samples"""
        data = torch.randn(100)  # Flattened data from multiple samples
        lengths = torch.tensor([25, 25, 25, 25])  # 4 samples of 25 points each

        sum_result = corrosion_unet._calc_segment_sum(data, lengths)
        mean_result = corrosion_unet._calc_segment_mean(data, lengths)
        std_result = corrosion_unet._calc_segment_std(data, lengths)

        assert sum_result.shape == (4,)
        assert mean_result.shape == (4,)
        assert std_result.shape == (4,)


class TestCorrosionUNetVolumeCalculation:
    """Test volume loss calculation"""

    def test_calc_volume_loss(self, corrosion_unet):
        """Test volume loss calculation"""
        y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y_target = torch.tensor([1.1, 2.1, 3.1, 4.1, 5.1])
        lengths = torch.tensor([2, 3])

        loss = corrosion_unet._calc_volume_loss(y_pred, y_target, lengths)

        assert loss.shape == torch.Size([])
        assert loss.item() >= 0

    def test_calc_volume_loss_perfect_prediction(self, corrosion_unet):
        """Test volume loss when prediction is perfect"""
        y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y_target = y_pred.clone()
        lengths = torch.tensor([2, 3])

        loss = corrosion_unet._calc_volume_loss(y_pred, y_target, lengths)

        assert loss.item() == 0.0


class TestCorrosionUNetBatchMetrics:
    """Test batch metric calculations"""

    def test_batch_r2score(self, corrosion_unet):
        """Test batch R2 score calculation"""
        # Create data with known R2
        y_pred = torch.randn(100)
        y_target = torch.randn(100)
        lengths = torch.tensor([25, 25, 25, 25])

        r2 = corrosion_unet._batch_r2score(y_pred, y_target, lengths)

        assert r2.shape == torch.Size([])
        assert r2.item() <= 1.0

    def test_batch_pearson_correlation(self, corrosion_unet):
        """Test batch Pearson correlation calculation"""
        y_pred = torch.randn(100)
        y_target = torch.randn(100)
        lengths = torch.tensor([25, 25, 25, 25])

        corr = corrosion_unet._batch_pearson_corr(y_pred, y_target, lengths)

        assert corr.shape == torch.Size([])
        assert -1.0 <= corr.item() <= 1.0

    def test_batch_r2score_perfect_prediction(self, corrosion_unet):
        """Test R2 score when prediction is perfect"""
        y_pred = torch.randn(100)
        y_target = y_pred.clone()
        lengths = torch.tensor([25, 25, 25, 25])

        r2 = corrosion_unet._batch_r2score(y_pred, y_target, lengths)

        assert r2.item() == pytest.approx(1.0, rel=1e-5)


class TestCorrosionUNetSteps:
    """Test training/validation/test steps"""

    def test_training_step(self, corrosion_unet, sample_batch):
        """Test training step"""
        loss = corrosion_unet.training_step(sample_batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_validation_step(self, corrosion_unet, sample_batch):
        """Test validation step"""
        loss = corrosion_unet.validation_step(sample_batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_test_step(self, corrosion_unet, sample_batch):
        """Test test step"""
        loss = corrosion_unet.test_step(sample_batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_all_steps_return_loss(self, corrosion_unet, sample_batch):
        """Test that all steps return a loss tensor"""
        train_loss = corrosion_unet.training_step(sample_batch, batch_idx=0)
        val_loss = corrosion_unet.validation_step(sample_batch, batch_idx=0)
        test_loss = corrosion_unet.test_step(sample_batch, batch_idx=0)

        for loss in [train_loss, val_loss, test_loss]:
            assert isinstance(loss, torch.Tensor)
            assert not torch.isnan(loss)
            assert not torch.isinf(loss)


class TestCorrosionUNetOptimizer:
    """Test optimizer configuration"""

    def test_configure_optimizers(self, corrosion_unet):
        """Test that configure_optimizers returns an optimizer"""
        optimizer = corrosion_unet.configure_optimizers()

        assert isinstance(optimizer, torch.optim.Optimizer)
        assert isinstance(optimizer, torch.optim.AdamW)

    def test_optimizer_learning_rate(self, corrosion_unet, model_config):
        """Test that optimizer has correct learning rate"""
        optimizer = corrosion_unet.configure_optimizers()

        lr = optimizer.param_groups[0]["lr"]
        assert lr == model_config["lr"]

    def test_optimizer_weight_decay(self, corrosion_unet, model_config):
        """Test that optimizer has correct weight decay"""
        optimizer = corrosion_unet.configure_optimizers()

        wd = optimizer.param_groups[0]["weight_decay"]
        assert wd == model_config["weight_decay"]


class TestCorrosionUNetDetermineCrop:
    """Test crop determination for reconstruction"""

    def test_determine_crop_min_position(self, corrosion_unet):
        """Test crop determination at minimum position"""
        corrosion_unet.reconstruction_overlap = 8

        start_crop, end_crop = corrosion_unet._determine_crop(pos=0, min_pos=0, max_pos=100)

        assert start_crop == 0
        assert end_crop == 8

    def test_determine_crop_max_position(self, corrosion_unet):
        """Test crop determination at maximum position"""
        corrosion_unet.reconstruction_overlap = 8

        start_crop, end_crop = corrosion_unet._determine_crop(pos=100, min_pos=0, max_pos=100)

        assert start_crop == 8
        assert end_crop == 0

    def test_determine_crop_middle_position(self, corrosion_unet):
        """Test crop determination at middle position"""
        corrosion_unet.reconstruction_overlap = 8

        start_crop, end_crop = corrosion_unet._determine_crop(pos=50, min_pos=0, max_pos=100)

        assert start_crop == 8
        assert end_crop == 8

    def test_determine_crop_no_overlap(self, corrosion_unet):
        """Test crop determination with no overlap"""
        corrosion_unet.reconstruction_overlap = 0

        start_crop, end_crop = corrosion_unet._determine_crop(pos=0, min_pos=0, max_pos=100)

        assert start_crop == 0
        assert end_crop == 0


class TestCorrosionUNetGradientFlow:
    """Test gradient flow through the model"""

    def test_gradients_flow_through_forward(self, corrosion_unet):
        """Test that gradients flow through forward pass"""
        x = torch.randn(2, 4, 128, 128, requires_grad=True)
        output = corrosion_unet(x)

        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_gradients_flow_through_loss(self, corrosion_unet, sample_batch):
        """Test that gradients flow through loss computation"""
        x, y_target, mask = sample_batch
        x.requires_grad = True

        y_pred = corrosion_unet(x)
        losses = corrosion_unet._calc_losses(y_pred, y_target, mask)
        loss = losses["mae-loss"]

        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_optimizer_step_works(self, corrosion_unet, sample_batch):
        """Test that optimizer step works"""
        optimizer = corrosion_unet.configure_optimizers()

        initial_params = [p.clone() for p in corrosion_unet.parameters()]

        x, y_target, mask = sample_batch
        y_pred = corrosion_unet(x)
        losses = corrosion_unet._calc_losses(y_pred, y_target, mask)
        loss = losses["mae-loss"]

        loss.backward()
        optimizer.step()

        # Check that at least some parameters changed
        params_changed = False
        for initial_p, current_p in zip(initial_params, corrosion_unet.parameters()):
            if not torch.allclose(initial_p, current_p):
                params_changed = True
                break

        assert params_changed


class TestCorrosionUNetEdgeCases:
    """Test edge cases"""

    def test_with_single_sample_batch(self, corrosion_unet):
        """Test with single sample in batch"""
        x = torch.randn(1, 4, 128, 128)
        y = torch.randn(1, 1, 128, 128)
        mask = torch.zeros((1, 1, 128, 128), dtype=torch.bool)

        y_pred = corrosion_unet(x)
        losses = corrosion_unet._calc_losses(y_pred, y, mask)

        assert isinstance(losses, dict)

    def test_with_large_batch(self, corrosion_unet):
        """Test with large batch size"""
        x = torch.randn(16, 4, 128, 128)
        y = torch.randn(16, 1, 128, 128)
        mask = torch.zeros((16, 1, 128, 128), dtype=torch.bool)

        y_pred = corrosion_unet(x)
        losses = corrosion_unet._calc_losses(y_pred, y, mask)

        assert isinstance(losses, dict)

    def test_with_eval_mode(self, corrosion_unet):
        """Test model in eval mode"""
        corrosion_unet.eval()

        x = torch.randn(2, 4, 128, 128)
        with torch.no_grad():
            output = corrosion_unet(x)

        assert output.shape == (2, 1, 128, 128)
