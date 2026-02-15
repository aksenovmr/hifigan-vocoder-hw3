import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

LRELU_SLOPE = 0.1


class PeriodicDiscriminator(nn.Module):

    def __init__(self, period: int):
        super().__init__()
        self.period = period

        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0))),
        ])

        self.post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor):
        fmap = []

        B, C, T = x.shape
        if T % self.period != 0:
            pad_len = self.period - (T % self.period)
            x = F.pad(x, (0, pad_len), mode="reflect")
            T = T + pad_len

        x = x.view(B, C, T // self.period, self.period)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)

        x = self.post(x)
        fmap.append(x)

        logits = torch.flatten(x, 1, -1)
        return logits, fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [PeriodicDiscriminator(p) for p in periods]
        )

    def forward(self, x: torch.Tensor):
        outputs = []
        for d in self.discriminators:
            logits, fmaps = d(x)
            outputs.append({
                "logits": logits,
                "features": fmaps,
            })
        return outputs

class ScaleDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        
        norm = nn.utils.spectral_norm if use_spectral_norm else weight_norm

        self.convs = nn.ModuleList([
            norm(nn.Conv1d(1, 128, kernel_size=15, stride=1, padding=7)),
            norm(nn.Conv1d(128, 128, kernel_size=41, stride=2, groups=4, padding=20)),
            norm(nn.Conv1d(128, 256, kernel_size=41, stride=2, groups=16, padding=20)),
            norm(nn.Conv1d(256, 512, kernel_size=41, stride=4, groups=16, padding=20)),
            norm(nn.Conv1d(512, 1024, kernel_size=41, stride=4, groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, kernel_size=41, stride=1, groups=16, padding=20)),
        ])

        self.post = weight_norm(
            nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x: torch.Tensor):
        fmap = []

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)

        x = self.post(x)
        fmap.append(x)

        logits = torch.flatten(x, 1, -1)
        return logits, fmap

class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(use_spectral_norm=True),
            ScaleDiscriminator(use_spectral_norm=False),
            ScaleDiscriminator(use_spectral_norm=False),
        ])

        self.avgpools = nn.ModuleList([
            nn.AvgPool1d(kernel_size=4, stride=2, padding=1),
            nn.AvgPool1d(kernel_size=4, stride=2, padding=1),
        ])

    def forward(self, x: torch.Tensor):
        outputs = []

        for i, d in enumerate(self.discriminators):
            if i == 0:
                xi = x
            elif i == 1:
                xi = self.avgpools[0](x)
            else:
                xi = self.avgpools[1](self.avgpools[0](x))

            logits, fmaps = d(xi)
            outputs.append({
                "logits": logits,
                "features": fmaps,
            })

        return outputs