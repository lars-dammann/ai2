from model.unet import UNet

import lightning as pl
from torchmetrics.regression import R2Score
import torch
import torch.nn as nn

class CorrosionUNet(pl.LightningModule):
    def __init__(self, config, learning_rate=1e-3, weight_decay=1e-5):
        super().__init__()
        self.model = UNet(config, in_channels=4, out_channels=1)
        self.loss_fn = nn.functional.mse_loss
        self.learning_rate = config["lr"]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # x: (B,3,H,W) Imgage + (B,1,H,W) Height profile, y: (B,1,H,W) Height profile
        x, y_target = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y_target)
        self.log('train_loss', loss)
        r2score = R2Score()
        self.log('train_r2_loss', r2score(y_pred, y_target))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_target = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y_target)
        self.log('val_loss', loss)
        r2score = R2Score()
        self.log('val_r2_loss', r2score(y_pred, y_target))
        return loss

    def test_step(self, batch, batch_idx):
        x, y_target = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y_target)
        self.log('test_loss', loss)
        r2score = R2Score()
        self.log('test_r2_loss', r2score(y_pred, y_target))
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
