import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int, int],
        in_channels: list[int],
        is_conditional: bool = True,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.is_conditional = is_conditional

        layers = []
        for idx, out_channels in enumerate(self.in_channels):
            layers.append(
                nn.Conv2d(
                    in_channels=(
                        self.img_size[0] * (2 if self.is_conditional else 1)
                        if idx == 0
                        else self.in_channels[idx - 1]
                    ),
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2 if idx < len(self.in_channels) - 1 else 1,
                    padding=1,
                    bias=False,
                )
            )
            if idx != 0:
                layers.append(nn.BatchNorm2d(out_channels, eps=0.8))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        layers.append(
            nn.Conv2d(
                in_channels=self.in_channels[-1],
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=1,
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.model(input)
        return out
