from pathlib import Path
import torch
import torchaudio


class CustomDirDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        mel_transform,
        sample_rate: int = 22050,
    ):
        self.root_dir = Path(root_dir)
        self.wav_files = sorted(self.root_dir.glob("*.wav"))

        if len(self.wav_files) == 0:
            raise RuntimeError(f"No wav files found in {self.root_dir}")

        self.mel_transform = mel_transform
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        wav_path = self.wav_files[idx]

        waveform, sr = torchaudio.load(wav_path)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sr, self.sample_rate
            )

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        mel = self.mel_transform(waveform)

        return {
            "waveform": waveform,
            "mel": mel,
            "wav_path": wav_path.name,
        }