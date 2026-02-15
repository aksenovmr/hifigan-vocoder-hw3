import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


LRELU_SLOPE = 0.1


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2

def init_weights(m: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.normal_(m.weight, mean, std)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ResBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilations=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(
                channels, channels, kernel_size,
                stride=1, dilation=d,
                padding=get_padding(kernel_size, d)
            )) for d in dilations
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(
                channels, channels, kernel_size,
                stride=1, dilation=1,
                padding=get_padding(kernel_size, 1)
            )) for _ in dilations
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = x + xt
        return x


class MRF(nn.Module):
    def __init__(self, channels: int, kernel_sizes=(3, 7, 11), dilations=(1, 3, 5)):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResBlock1(channels, k, dilations=dilations) for k in kernel_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = 0.0
        for b in self.blocks:
            out = out + b(x)
        return out / len(self.blocks)


class HiFiGANGeneratorV2(nn.Module):
    def __init__(
        self,
        n_mels: int = 80,
        hu: int = 128,
        upsample_rates=(8, 8, 2, 2),
        upsample_kernel_sizes=(16, 16, 4, 4),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilations=(1, 3, 5),
    ):
        super().__init__()

        assert len(upsample_rates) == len(upsample_kernel_sizes)

        self.pre = weight_norm(nn.Conv1d(n_mels, hu, kernel_size=7, padding=3))

        self.ups = nn.ModuleList()
        self.mrfs = nn.ModuleList()

        in_ch = hu
        for u, k in zip(upsample_rates, upsample_kernel_sizes):
            out_ch = in_ch // 2
            self.ups.append(
                weight_norm(nn.ConvTranspose1d(
                    in_ch, out_ch,
                    kernel_size=k,
                    stride=u,
                    padding=(k - u) // 2
                ))
            )
            self.mrfs.append(MRF(out_ch, kernel_sizes=resblock_kernel_sizes, dilations=resblock_dilations))
            in_ch = out_ch

        self.post = weight_norm(nn.Conv1d(in_ch, 1, kernel_size=7, padding=3))
        self.apply(init_weights)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.pre(mel)
        for up, mrf in zip(self.ups, self.mrfs):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = up(x)
            x = mrf(x)

        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.post(x)
        x = torch.tanh(x)
        return x