import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.trainer.hifigan_trainer import HiFiGANTrainer
from src.datasets.data_utils import get_dataloaders
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    dataloaders, batch_transforms = get_dataloaders(config, device)

    generator = instantiate(config.model).to(device)
    logger.info(generator)

    mpd = instantiate(config.trainer.mpd).to(device)
    msd = instantiate(config.trainer.msd).to(device)

    loss_obj = instantiate(config.loss_function)

    mel_transform = instantiate(config.transforms.instance_transforms).to(device)

    optimizer_g = instantiate(
        config.optimizer,
        params=generator.parameters(),
    )

    optimizer_d = instantiate(
        config.optimizer,
        params=list(mpd.parameters()) + list(msd.parameters()),
    )

    lr_scheduler_g = None
    lr_scheduler_d = None

    epoch_len = config.trainer.get("epoch_len")

    trainer = HiFiGANTrainer(
        generator=generator,
        mpd=mpd,
        msd=msd,
        loss_obj=loss_obj,
        mel_transform=mel_transform,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        lr_scheduler_g=lr_scheduler_g,
        lr_scheduler_d=lr_scheduler_d,
        cfg=config,
        device=device,
        dataloaders=dataloaders,
        logger=logger,
        writer=writer,
        epoch_len=epoch_len,
        batch_transforms=batch_transforms,
    )

    trainer.train()


if __name__ == "__main__":
    main()