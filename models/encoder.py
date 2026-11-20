import torch
import torch.nn as nn
from torchvision import models

class Encoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
    ):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        return mu, logvar