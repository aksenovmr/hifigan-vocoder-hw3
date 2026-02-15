from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F
import torch.nn as nn


@dataclass
class HiFiGANLossConfig:
    lambda_fm: float = 2.0
    lambda_mel: float = 45.0


def _ls_gan_discriminator_loss(d_out_real: List[Dict], d_out_fake: List[Dict]) -> torch.Tensor:

    loss = 0.0
    for real, fake in zip(d_out_real, d_out_fake):
        real_logits = real["logits"]
        fake_logits = fake["logits"]
        loss = loss + torch.mean((real_logits - 1.0) ** 2) + torch.mean((fake_logits) ** 2)
    return loss


def _ls_gan_generator_loss(d_out_fake: List[Dict]) -> torch.Tensor:

    loss = 0.0
    for fake in d_out_fake:
        fake_logits = fake["logits"]
        loss = loss + torch.mean((fake_logits - 1.0) ** 2)
    return loss


def feature_matching_loss(d_out_real: List[Dict], d_out_fake: List[Dict]) -> torch.Tensor:
    loss = 0.0
    for real, fake in zip(d_out_real, d_out_fake):
        real_fmaps = real["features"]
        fake_fmaps = fake["features"]

        real_fmaps = real_fmaps[:-1]
        fake_fmaps = fake_fmaps[:-1]

        for rf, ff in zip(real_fmaps, fake_fmaps):
            loss = loss + torch.mean(torch.abs(rf - ff))
    return loss


def mel_spectrogram_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    mel_transform,
) -> torch.Tensor:

    with torch.no_grad():
        mel_true = mel_transform(y_true)
    mel_pred = mel_transform(y_pred)
    return torch.mean(torch.abs(mel_true - mel_pred))


class HiFiGANLoss(nn.Module):
    def __init__(self, config: HiFiGANLossConfig):
        super().__init__()
        self.config = config

    def discriminator_loss(
        self,
        mpd_real: List[Dict],
        mpd_fake: List[Dict],
        msd_real: List[Dict],
        msd_fake: List[Dict],
    ) -> torch.Tensor:
        loss_mpd = _ls_gan_discriminator_loss(mpd_real, mpd_fake)
        loss_msd = _ls_gan_discriminator_loss(msd_real, msd_fake)
        return loss_mpd + loss_msd

    def generator_loss(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        mel_transform,
        mpd_real: List[Dict],
        mpd_fake: List[Dict],
        msd_real: List[Dict],
        msd_fake: List[Dict],
    ) -> Dict[str, torch.Tensor]:
        adv = _ls_gan_generator_loss(mpd_fake) + _ls_gan_generator_loss(msd_fake)
        fm = feature_matching_loss(mpd_real, mpd_fake) + feature_matching_loss(msd_real, msd_fake)
        mel = mel_spectrogram_loss(y_true, y_pred, mel_transform)

        total = adv + self.config.lambda_fm * fm + self.config.lambda_mel * mel

        return {
            "total": total,
            "adv": adv,
            "fm": fm,
            "mel": mel,
        }