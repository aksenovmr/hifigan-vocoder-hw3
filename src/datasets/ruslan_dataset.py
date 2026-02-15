from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset
import soundfile as sf

from src.transforms import MelSpectrogramTransform


class RUSLANDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        mel_transform: MelSpectrogramTransform,
        split: str = "train",
        split_ratio: float = 0.9,
        max_audio_length: int | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.mel_transform = mel_transform
        self.max_audio_length = max_audio_length

        self.wav_files: List[Path] = sorted(self.root_dir.rglob("*.wav"))
        if len(self.wav_files) == 0:
            raise RuntimeError(f"No wav files found in {self.root_dir}")

        n_total = len(self.wav_files)
        n_train = int(n_total * split_ratio)

        if split == "train":
            self.wav_files = self.wav_files[:n_train]
        elif split == "val":
            self.wav_files = self.wav_files[n_train:]
        else:
            raise ValueError("split must be 'train' or 'val'")

    def __len__(self) -> int:
        return len(self.wav_files)

    def _load_audio(self, path: Path) -> torch.Tensor:
        waveform, sr = sf.read(path)
        waveform = torch.from_numpy(waveform).float()

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.mean(dim=1, keepdim=True)

        if self.max_audio_length is not None:
            if waveform.shape[1] > self.max_audio_length:
                waveform = waveform[:, : self.max_audio_length]

        return waveform

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        wav_path = self.wav_files[index]

        waveform = self._load_audio(wav_path)
        mel = self.mel_transform(waveform)

        return {
            "waveform": waveform,
            "mel": mel,
            "wav_path": str(wav_path),
        }