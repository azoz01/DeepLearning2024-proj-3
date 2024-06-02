from pathlib import Path

import torch
from diffusers import DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader

from engine.args import get_shell_args
from engine.data import DatasetFromDisk
from engine.factories import get_scheduler
from engine.model import get_unet2d_model
from engine.train import train_model
from engine.utils import TrainingConfig, device


def main():
    args = get_shell_args()
    print(args.__dict__)
    config = TrainingConfig(f"results/{args.name}", 10, train_batch_size=12)
    train_dataset = DatasetFromDisk(Path("data/splitted/train"), device=device)
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.train_batch_size
    )
    if Path(f"results/{args.name}").exists():
        model = DDPMPipeline.from_pretrained(
            f"results/{args.name}", use_safe_tensors=True
        ).config.unet
    else:
        model = get_unet2d_model(config)
    noise_scheduler = get_scheduler(args.noise_scheduler, args.denoising_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )
    train_model(
        config,
        model,
        noise_scheduler,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )


if __name__ == "__main__":
    main()
