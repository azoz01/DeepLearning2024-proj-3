from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_euler_discrete import (
    EulerDiscreteScheduler,
)


def get_scheduler(scheduler_name: str, denoising_steps: int):
    match scheduler_name:
        case "DDPMScheduler":
            return DDPMScheduler(num_train_timesteps=denoising_steps)
        case "DDIMScheduler":
            return DDIMScheduler(num_train_timesteps=denoising_steps)
        case "EulerDiscreteScheduler":
            return EulerDiscreteScheduler(num_train_timesteps=denoising_steps)
