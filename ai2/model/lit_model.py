from model.unet import UNet

import lightning as pl
import torch
import torch.nn as nn

class CorrosionUNet(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.model = UNet(in_channels=4, out_channels=1)
        self.loss_fn = nn.functional.mse_loss  # Regression
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch  # x: (B,4,H,W), y: (B,1,H,W)
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
