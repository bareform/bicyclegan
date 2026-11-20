import torch
import torch.nn as nn

from .blocks import DownsampleBlock, UpsampleBlock

class Generator(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int, int],
        in_channels: list[int],
        latent_dim: int,
    ):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.fc = nn.Linear(latent_dim, img_size[1] * img_size[2])

        down_blocks = []
        down_in_channels = in_channels
        for idx in range(len(down_in_channels)):
            down_blocks.append(
                DownsampleBlock(
                    in_channels=(self.img_size[0] + 1) if idx == 0 else down_in_channels[idx - 1],
                    out_channels=down_in_channels[idx],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                    use_batch_norm=idx != 0,
                )
            )

        self.down_blocks = nn.ModuleList(down_blocks)

        self.mid_blocks = DownsampleBlock(
            in_channels=down_in_channels[-1],
            out_channels=down_in_channels[-1],
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
            use_batch_norm=False,
        )

        up_blocks = []
        up_in_channels = list(reversed(self.in_channels))
        for idx in range(len(up_in_channels)):
            up_blocks.append(
                UpsampleBlock(
                    in_channels=up_in_channels[idx] if idx == 0 else up_in_channels[idx - 1] * 2,
                    out_channels=up_in_channels[idx],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                    use_batch_norm=True,
                )
            )
        self.up_blocks = nn.ModuleList(up_blocks)

        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                2 * self.in_channels[0],
                self.img_size[0],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Tanh(),
        )

    def forward(self, input: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        skips = []
        latent_map = self.fc(latent).view(-1, 1, self.img_size[1], self.img_size[2])
        out = torch.cat((input, latent_map), dim=1)
        for down_block in self.down_blocks:
            out = down_block(out)
            skips.append(out)
        out = self.mid_blocks(out)
        for idx, up_block in enumerate(self.up_blocks):
            out = up_block(out)
            out = torch.cat((out, skips[-(idx + 1)]), dim=1)
        return self.conv(out)
