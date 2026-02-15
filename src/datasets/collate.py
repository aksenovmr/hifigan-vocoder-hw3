import torch
import random
import torch.nn.functional as F


# def collate_fn(dataset_items: list[dict]):
#     """
#     Collate and pad fields in the dataset items.
#     Converts individual items into a batch.

#     Args:
#         dataset_items (list[dict]): list of objects from
#             dataset.__getitem__.
#     Returns:
#         result_batch (dict[Tensor]): dict, containing batch-version
#             of the tensors.
#     """

#     result_batch = {}

#     # example of collate_fn
#     result_batch["data_object"] = torch.vstack(
#         [elem["data_object"] for elem in dataset_items]
#     )
#     result_batch["labels"] = torch.tensor([elem["labels"] for elem in dataset_items])

#     return result_batch


def collate_fn_vocoder(dataset_items: list[dict], segment_length: int = 8192, hop_length: int = 256):
    waveforms = []
    mels = []

    mel_frames = segment_length // hop_length

    for item in dataset_items:
        waveform = item["waveform"]
        mel = item["mel"]

        T = waveform.shape[1]
        Fm = mel.shape[2]

        if T >= segment_length:
            start = random.randint(0, T - segment_length)
            waveform = waveform[:, start:start + segment_length]

            mel_start = start // hop_length
            mel_start = min(mel_start, max(0, Fm - mel_frames))
            mel = mel[:, :, mel_start:mel_start + mel_frames]
        else:
            pad = segment_length - T
            waveform = F.pad(waveform, (0, pad))

            mel_pad = mel_frames - Fm
            if mel_pad > 0:
                mel = F.pad(mel, (0, mel_pad))

        waveforms.append(waveform)
        mels.append(mel.squeeze(0))

    return {
        "waveform": torch.stack(waveforms, dim=0),
        "mel": torch.stack(mels, dim=0),
    }

