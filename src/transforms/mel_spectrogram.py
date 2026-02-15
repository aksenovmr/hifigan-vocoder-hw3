from dataclasses import dataclass
import torch
from torch import nn

import torchaudio
import librosa


@dataclass
class MelSpectrogramConfig:
    sample_rate: int = 22050
    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 256

    n_mels: int = 80
    f_min: int = 0
    f_max: int = 11025

    power: float = 2.0
    log_eps: float = 1e-5


class MelSpectrogramTransform(nn.Module):
    def __init__(self, config: MelSpectrogramConfig):
        super().__init__()
        self.config = config

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max,
            power=config.power,
            center=True,
            normalized=False,
        )

        mel_basis = librosa.filters.mel(
            sr=config.sample_rate,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=config.f_min,
            fmax=config.f_max,
            htk=False,
            norm="slaney",
        ).T

        self.mel.mel_scale.fb.copy_(torch.tensor(mel_basis, dtype=torch.float32))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        device = audio.device

        if self.mel.spectrogram.window.device != device:
            self.mel.spectrogram.window = self.mel.spectrogram.window.to(device)
    
        if self.mel.mel_scale.fb.device != device:
            self.mel.mel_scale.fb = self.mel.mel_scale.fb.to(device)
    
        mel = self.mel(audio)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel
