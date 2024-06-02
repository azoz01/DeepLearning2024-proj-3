import os
from math import sqrt
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.pipelines.ddpm.pipeline_ddpm import DDPMPipeline
from diffusers.utils.pil_utils import make_image_grid
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .utils import TrainingConfig


def train_model(
    config: TrainingConfig,
    model: UNet2DModel,
    noise_scheduler: Any,
    optimizer: Optimizer,
    train_dataloader: DataLoader,
    lr_scheduler: LRScheduler,
) -> None:
    accelerator, model, optimizer, train_dataloader, lr_scheduler = (
        __setup_accelerator(
            config, model, optimizer, train_dataloader, lr_scheduler
        )
    )
    global_step = 0
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(
            train_dataloader,
            total=len(train_dataloader),
            disable=not accelerator.is_local_main_process,
        )
        progress_bar.set_description(f"Epoch {epoch}")

        for i, batch in enumerate(progress_bar):
            clean_images = batch
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            batch_size = clean_images.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,  # type: ignore
                (batch_size,),
                device=clean_images.device,
                dtype=torch.int64,
            )

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)  # type: ignore # noqa: E501

            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps, return_dict=False)[
                    0
                ]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        pipeline = DDPMPipeline(
            unet=accelerator.unwrap_model(model), scheduler=noise_scheduler
        )
        evaluate(
            config, epoch, pipeline, noise_scheduler.config.num_train_timesteps
        )
        pipeline.save_pretrained(config.output_dir)


def __setup_accelerator(
    config: TrainingConfig,
    model: UNet2DModel,
    optimizer: Optimizer,
    train_dataloader: DataLoader,
    lr_scheduler: LRScheduler,
) -> tuple:
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    os.makedirs(config.output_dir, exist_ok=True)
    if accelerator.is_main_process:
        accelerator.init_trackers("train")

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    return accelerator, model, optimizer, train_dataloader, lr_scheduler


@torch.no_grad()
def evaluate(
    config: TrainingConfig,
    epoch: int,
    pipeline: Any,
    num_inference_steps: int,
) -> None:
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
        num_inference_steps=num_inference_steps,
    ).images  # type: ignore
    rows = int(sqrt(config.eval_batch_size))
    cols = config.eval_batch_size // rows
    if rows * cols < config.eval_batch_size:
        cols += 1
    image_grid = make_image_grid(images, rows=rows, cols=cols)  # type: ignore

    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")
