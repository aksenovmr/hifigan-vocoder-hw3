import torch
import torch.nn as nn


class DummyMetric(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        return torch.tensor(0.0)