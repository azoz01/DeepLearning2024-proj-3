import shutil
from pathlib import Path

import pytorch_lightning as pl
import torch
from tqdm import tqdm

OUTPUT_PATH = Path("data/splitted")
RAW_DATA_PATH = Path("data/raw/sample")


def main():
    pl.seed_everything(1)

    OUTPUT_PATH.mkdir(exist_ok=True, parents=True)
    for sample_path in ["train", "val", "test"]:
        (OUTPUT_PATH / sample_path).mkdir(exist_ok=True)

    imgs_paths = list(sorted(RAW_DATA_PATH.rglob("*.jpg")))
    splits = torch.rand(size=[len(imgs_paths)])

    for split, img_path in tqdm(
        zip(splits, imgs_paths), total=len(imgs_paths)
    ):
        if split <= 0.7:
            split = "train"
        else:
            split = "test"
        shutil.copy(img_path, OUTPUT_PATH / split / img_path.name)


if __name__ == "__main__":
    main()
