import torch
import torchaudio
from pathlib import Path
from typing import Any, Dict, Optional, Union
from omegaconf import OmegaConf

from src.trainer.base_trainer import BaseTrainer


def _set_requires_grad(module: torch.nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def _detach_disc_outputs(d_out):
    if isinstance(d_out, list):
        return [_detach_disc_outputs(x) for x in d_out]

    if not isinstance(d_out, dict):
        return d_out.detach() if torch.is_tensor(d_out) else d_out

    out = {}
    for k, v in d_out.items():
        if torch.is_tensor(v):
            out[k] = v.detach()
        elif isinstance(v, list):
            out[k] = [
                t.detach() if torch.is_tensor(t) else t
                for t in v
            ]
        else:
            out[k] = v
    return out


class HiFiGANTrainer(BaseTrainer):

    def __init__(
        self,
        generator,
        mpd,
        msd,
        loss_obj,
        mel_transform,
        optimizer_g,
        optimizer_d,
        lr_scheduler_g,
        lr_scheduler_d,
        cfg,
        device,
        dataloaders,
        logger,
        writer,
        epoch_len=None,
        skip_oom=True,
        batch_transforms=None,
        **kwargs,
    ):

        self.generator = generator
        self.mpd = mpd
        self.msd = msd

        self.loss_obj = loss_obj
        self.mel_transform = mel_transform.to(device)

        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d

        self.lr_scheduler_g = lr_scheduler_g
        self.lr_scheduler_d = lr_scheduler_d

        self.sample_rate = 22050
        try:
            self.sample_rate = int(self.mel_transform.config.sample_rate)
        except Exception:
            pass

        self._audio_logged_train = False
        self._audio_logged_val = False

        self.mos_dir = Path("data/mos_wavs")
        self.mos_files = sorted(self.mos_dir.glob("*.wav")) if self.mos_dir.exists() else []

        super().__init__(
            model=self.generator,
            criterion=None,
            metrics={"train": [], "inference": []},
            optimizer=self.optimizer_g,
            lr_scheduler=None,
            config=cfg,
            device=device,
            dataloaders=dataloaders,
            logger=logger,
            writer=writer,
            epoch_len=epoch_len,
            skip_oom=skip_oom,
            batch_transforms=batch_transforms,
        )

    def _save_checkpoint(self, epoch: int, save_best: bool = False, only_best: bool = False):

        state: Dict[str, Any] = {
            "epoch": epoch,
            "global_step": getattr(self, "_global_step", None),
            "config": OmegaConf.to_container(self.config, resolve=True),
            "monitor_best": getattr(self, "monitor_best", None),
            "generator": self.generator.state_dict(),
            "mpd": self.mpd.state_dict(),
            "msd": self.msd.state_dict(),
            "optimizer_g": self.optimizer_g.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict(),
            "lr_scheduler_g": self.lr_scheduler_g.state_dict() if self.lr_scheduler_g else None,
            "lr_scheduler_d": self.lr_scheduler_d.state_dict() if self.lr_scheduler_d else None,
        }

        filename = self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth"
        self.logger.info(f"Saving checkpoint: {filename}")
        torch.save(state, filename)

        if save_best:
            best_path = self.checkpoint_dir / "model_best.pth"
            torch.save(state, best_path)

    def _resume_checkpoint(self, resume_path: Union[str, Path]):

        resume_path = Path(resume_path)
        self.logger.info(f"Loading checkpoint: {resume_path}")

        checkpoint = torch.load(resume_path, map_location=self.device)

        self.generator.load_state_dict(checkpoint["generator"])
        self.mpd.load_state_dict(checkpoint["mpd"])
        self.msd.load_state_dict(checkpoint["msd"])

        if checkpoint.get("optimizer_g"):
            self.optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        if checkpoint.get("optimizer_d"):
            self.optimizer_d.load_state_dict(checkpoint["optimizer_d"])

        if self.lr_scheduler_g and checkpoint.get("lr_scheduler_g"):
            self.lr_scheduler_g.load_state_dict(checkpoint["lr_scheduler_g"])
        if self.lr_scheduler_d and checkpoint.get("lr_scheduler_d"):
            self.lr_scheduler_d.load_state_dict(checkpoint["lr_scheduler_d"])

        if checkpoint.get("monitor_best") is not None:
            self.monitor_best = checkpoint["monitor_best"]

        self.start_epoch = checkpoint["epoch"] + 1

        if checkpoint.get("global_step") is not None:
            self._global_step = checkpoint["global_step"]

        self.logger.info(f"Resume from epoch {self.start_epoch}")

    def process_batch(self, batch, metrics=None):

        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        mel = batch["mel"]
        y_true = batch["waveform"]

        if not self.is_train:
            with torch.no_grad():
                y_pred = self.generator(mel)   

                mpd_real = self.mpd(y_true)
                msd_real = self.msd(y_true)
                mpd_fake = self.mpd(y_pred)
                msd_fake = self.msd(y_pred)

                d_loss = self.loss_obj.discriminator_loss(
                    mpd_real=mpd_real,
                    mpd_fake=mpd_fake,
                    msd_real=msd_real,
                    msd_fake=msd_fake,
                )

                g_losses = self.loss_obj.generator_loss(
                    y_true=y_true,
                    y_pred=y_pred,
                    mel_transform=self.mel_transform,
                    mpd_real=_detach_disc_outputs(mpd_real),
                    mpd_fake=mpd_fake,
                    msd_real=_detach_disc_outputs(msd_real),
                    msd_fake=msd_fake,
                )

                total = (d_loss + g_losses["total"]).detach()

            batch.update({"y_pred": y_pred.detach(), "loss": total})

            if metrics:
                metrics.update("loss", total.item())

            return batch

        y_pred = self.generator(mel)

        _set_requires_grad(self.mpd, True)
        _set_requires_grad(self.msd, True)

        self.optimizer_d.zero_grad(set_to_none=True)

        mpd_real = self.mpd(y_true)
        msd_real = self.msd(y_true)
        mpd_fake_d = self.mpd(y_pred.detach())
        msd_fake_d = self.msd(y_pred.detach())

        d_loss = self.loss_obj.discriminator_loss(
            mpd_real=mpd_real,
            mpd_fake=mpd_fake_d,
            msd_real=msd_real,
            msd_fake=msd_fake_d,
        )

        d_loss.backward()
        self.optimizer_d.step()

        _set_requires_grad(self.mpd, False)
        _set_requires_grad(self.msd, False)

        self.optimizer_g.zero_grad(set_to_none=True)

        mpd_fake_g = self.mpd(y_pred)
        msd_fake_g = self.msd(y_pred)

        with torch.no_grad():
            mpd_real_g = self.mpd(y_true)
            msd_real_g = self.msd(y_true)

        g_losses = self.loss_obj.generator_loss(
            y_true=y_true,
            y_pred=y_pred,
            mel_transform=self.mel_transform,
            mpd_real=_detach_disc_outputs(mpd_real_g),
            mpd_fake=mpd_fake_g,
            msd_real=_detach_disc_outputs(msd_real_g),
            msd_fake=msd_fake_g,
        )

        g_loss = g_losses["total"]
        g_loss.backward()
        self._clip_grad_norm()
        self.optimizer_g.step()

        total = (d_loss + g_loss).detach()

        batch.update({"y_pred": y_pred.detach(), "loss": total})

        if metrics:
            metrics.update("loss", total.item())

        return batch


    @torch.no_grad()
    def _log_mos_samples(self, epoch: int):

        if self.writer is None or not self.mos_files:
            return

        self.generator.eval()

        for wav_path in self.mos_files:

            waveform, _ = torchaudio.load(wav_path)
            waveform = waveform.to(self.device)

            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            mel = self.mel_transform(waveform)
            y_pred = self.generator(mel)

            y_pred = y_pred.squeeze(0).cpu().clamp(-1, 1)

            self.writer.add_audio(
                f"mos/epoch_{epoch}/{wav_path.stem}",
                y_pred,
                sample_rate=self.sample_rate,
            )

    def _train_epoch(self, epoch: int):

        result = super()._train_epoch(epoch)

        self._log_mos_samples(epoch)

        if self.lr_scheduler_g:
            self.lr_scheduler_g.step()
        if self.lr_scheduler_d:
            self.lr_scheduler_d.step()

        return result