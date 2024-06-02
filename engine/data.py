from pathlib import Path

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class DatasetFromDisk(Dataset):
    def __init__(
        self,
        data_path: Path,
        file_mask: str = "*.jpg",
        output_img_size: int = 64,
        device: str = "cpu",
    ):
        self.data_path = data_path
        self.img_paths = list(data_path.rglob(file_mask))
        self.output_img_size = output_img_size
        self.transforms = T.Compose(
            [
                T.CenterCrop((256, 256)),
                T.Resize((output_img_size, output_img_size)),
                T.ToTensor(),
            ]
        )
        self.device = device

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int | slice):
        if isinstance(idx, slice):
            return [self.__get_item(i) for i in list(range(idx.stop))[idx]]

        else:
            return self.__get_item(idx)

    def __get_item(self, idx: int):
        path = self.img_paths[idx]
        img = Image.open(path)
        img = self.transforms(img).to(self.device)
        return img
