from dataclasses import dataclass

import matplotlib.pyplot as plt
from torch import cuda
from torch.utils.data import Dataset

device = "cuda" if cuda.is_available else "cpu"


@dataclass
class TrainingConfig:
    output_dir: str
    num_epochs: int
    train_batch_size: int = 32
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    image_size: int = 64
    mixed_precision: str = "fp16"
    seed: int = 0


def show_exemplar_images(dataset: Dataset) -> None:
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    for i, image in enumerate(dataset[:4]):
        axs[i].imshow(image.cpu().permute(1, 2, 0).numpy())
        axs[i].set_axis_off()
    fig.show()
