import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two convolution blocks used throughout the UNet encoder and decoder.

    Args:
        in_channels: Number of input feature channels.
        out_channels: Number of output feature channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groups = max(1, out_channels // 16)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
            nn.GroupNorm(num_groups=self.groups, num_channels=out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same'),
            nn.GroupNorm(num_groups=self.groups, num_channels=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Apply two convolution blocks.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, out_channels, H, W).
        """
        return self.net(x)


class UNet(nn.Module):
    """UNet backbone used for corrosion height prediction.

    Args:
        config: Model configuration dictionary.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        features: Unused default feature list kept for backward compatibility.
    """

    def __init__(self, config, in_channels=4, out_channels=1, features=[64, 128, 256, 512, 1024]):
        super().__init__()

        features = [config["startfeature"] * 2 ** i for i in range(config["udepth"])]

        torch.set_float32_matmul_precision("medium")

        self.downconv = nn.ModuleList()
        self.upconv = nn.ModuleList()
        self.upsampling = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        tail_features = features.copy()
        tail_features = tail_features[::-1]
        first_feature = tail_features.pop()

        head_features = features.copy()
        last_feature = head_features.pop()

        for feature in head_features:
            self.downconv.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for index, feature in enumerate(reversed(head_features)):
            self.upsampling.append(
                nn.ConvTranspose2d(tail_features[index], feature, kernel_size=2, stride=2)
            )
            self.upconv.append(DoubleConv(tail_features[index], feature))

        self.bottleneck = DoubleConv(head_features[-1], last_feature)
        self.final_conv = nn.Conv2d(first_feature, out_channels, kernel_size=1)

    def forward(self, x):
        """Run the encoder-decoder pass and add the residual height channel.

        Args:
            x: Input tensor with shape (B, in_channels, H, W).

        Returns:
            Output tensor with shape (B, out_channels, H, W).
        """
        residual = x[:, -1:].clone()

        skip_connections = []

        for down in self.downconv:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for index, current_upsampling in enumerate(self.upsampling):
            x = current_upsampling(x)
            x = torch.cat((skip_connections[index], x), dim=1)
            x = self.upconv[index](x)

        return self.final_conv(x) + residual


if __name__ == "__main__":
    x = torch.randn((1, 4, 512, 512))
    model = UNet(in_channels=4, out_channels=1)
    preds = model(x)
    print(preds.shape)
