import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """UNet specific double convolution. That is two blocks of 2D convolution, Batch normalization and Relu"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, config, in_channels=4, out_channels=1, features=[64, 128, 256, 512, 1024]):
        super().__init__()

        features = [config["startfeature"]*2**i for i in range(config["udepth"])]

        torch.set_float32_matmul_precision('medium')

        self.downconv = nn.ModuleList()
        self.upconv = nn.ModuleList()
        self.upsampling = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Create reveresed feature list with the first feature number missing
        tail_features = features.copy()
        tail_features = tail_features[::-1]
        first_feature = tail_features.pop()

        # Create feature list with the last feature number missing
        head_features = features.copy()
        last_feature = head_features.pop()

        # Down part (Encoder)
        for feature in head_features:
            self.downconv.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part (Decoder)
        for index, feature in enumerate(reversed(head_features)):
            self.upsampling.append(
                nn.ConvTranspose2d(tail_features[index], feature, kernel_size=2, stride=2)
            )
            self.upconv.append(DoubleConv(tail_features[index], feature))

        self.bottleneck = DoubleConv(head_features[-1], last_feature)
        self.final_conv = nn.Conv2d(first_feature, out_channels, kernel_size=1)
        # self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.downconv:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Calculate bottom bottleneck of the unter
        x = self.bottleneck(x)

        # Reverse skip connections to easily iterate over them
        skip_connections = skip_connections[::-1]

        # Decoder
        for index, current_upsampling in enumerate(self.upsampling):
            # Upsampling with ConvTranspose2d
            x = current_upsampling(x)
            # Concat across results across unet
            x = torch.cat((skip_connections[index], x), dim=1)
            # DoubleConv
            x = self.upconv[index](x)

        # Final convolution to required output channel
        # return self.final_conv(x) * self.scale
        return self.final_conv(x)


if __name__ == "__main__":
    # Beispielinput: Batchgröße 1, 4 Kanäle (RGB+Heatmap zuvor), 256x256 Pixel
    x = torch.randn((1, 4, 512, 512))
    model = UNet(in_channels=4, out_channels=1)
    preds = model(x)
    print(preds.shape)  # sollte torch.Size([1,1,256,256]) ausgeben
