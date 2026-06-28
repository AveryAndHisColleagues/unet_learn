import torch
import torch.nn as nn


class SSMBlock(nn.Module):
    """
    Mamba-like SSM surrogate (NO CUDA, NO mamba-ssm)
    """

    def __init__(self, dim):
        super().__init__()

        # input projection
        self.in_proj = nn.Linear(dim, dim)

        # "state transition" (fake A matrix behavior)
        self.state_update = nn.GRUCell(dim, dim)

        # output projection
        self.out_proj = nn.Linear(dim, dim)

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x: (B, L, C)
        """

        B, L, C = x.shape

        x = self.in_proj(x)
        x = self.norm(x)

        h = torch.zeros(B, C, device=x.device)

        outputs = []

        # sequential state propagation
        for t in range(L):
            h = self.state_update(x[:, t, :], h)
            outputs.append(h.unsqueeze(1))

        x = torch.cat(outputs, dim=1)

        x = self.out_proj(x)

        return x


class SSMBottleneck(nn.Module):
    """
    Drop-in replacement for Mamba bottleneck
    """

    def __init__(self, dim):
        super().__init__()

        self.conv_in = nn.Conv2d(dim, dim, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

        self.ssm = SSMBlock(dim)

        self.conv_out = nn.Conv2d(dim, dim, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        residual = x

        x = self.conv_in(x)
        x = self.bn1(x)
        x = self.act(x)

        B, C, H, W = x.shape

        # (B, C, H, W) -> (B, HW, C)
        x = x.flatten(2).transpose(1, 2)

        # SSM sequence modeling
        x = self.ssm(x)

        # back to image
        x = x.transpose(1, 2).reshape(B, C, H, W)

        x = self.conv_out(x)
        x = self.bn2(x)

        return x + residual