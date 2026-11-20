import torch
import torch.nn as nn

class UpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=4,
        stride=1,
        padding=0,
        bias=False,
        use_batch_norm=True,
    ) -> None:
        super().__init__()
        layers = [
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels, eps=0.8))
        layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.layers(input)
        return out

class DownsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=4,
        stride=1,
        padding=0,
        bias=False,
        use_batch_norm=True,
    ) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels, eps=0.8))
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.layers(input)
        return out
