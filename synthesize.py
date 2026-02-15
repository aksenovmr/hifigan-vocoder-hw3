import argparse
from pathlib import Path

import torch
import torchaudio
from omegaconf import OmegaConf
from hydra.utils import instantiate

from src.datasets.custom_dir_dataset import CustomDirDataset
from src.transforms.mel_spectrogram import (
    MelSpectrogramTransform,
    MelSpectrogramConfig,
)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="HiFi-GAN vocoder synthesis")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config (hifigan.yaml)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint (.pth)")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory with audio/ and transcriptions/")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save synthesized wavs")
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda or cpu")

    args = parser.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )

    config = OmegaConf.load(args.config)

    mel_cfg_dict = OmegaConf.to_object(
        OmegaConf.load("src/configs/transforms/mel.yaml")
        .instance_transforms
        .config
    )
    mel_cfg_dict.pop("_target_", None)

    mel_cfg = MelSpectrogramConfig(**mel_cfg_dict)
    mel_transform = MelSpectrogramTransform(mel_cfg).to(device)

    model_cfg = OmegaConf.load("src/configs/model/hifigan.yaml")
    generator = instantiate(model_cfg)
    generator.to(device)
    generator.eval()

    checkpoint = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(checkpoint["generator"])

    dataset = CustomDirDataset(
        root_dir=args.input_dir,
        mel_transform=mel_transform,
        sample_rate=mel_cfg.sample_rate,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Synthesizing {len(dataset)} utterances...")

    for item in dataset:
        mel = item["mel"].to(device)
        utt_id = Path(item["wav_path"]).stem

        y = generator(mel)
        y = y.squeeze(0).cpu()

        out_path = output_dir / f"{utt_id}.wav"

        torchaudio.save(
            out_path,
            y,
            sample_rate=mel_cfg.sample_rate,
            encoding="PCM_S",
            bits_per_sample=16,
        )

        print(f"Saved: {out_path}")

    print("Synthesis completed.")


if __name__ == "__main__":
    main()