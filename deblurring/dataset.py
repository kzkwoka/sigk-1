import os
from typing import List, Tuple

import cv2
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


class DeblurringDataset(Dataset):
    def __init__(self, image_dir: str, img_size: int = 256, kernel_size: int = 5):
        self.image_paths: List[str] = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png"))]
        self.image_dir: str = image_dir
        self.img_size: int = img_size
        self.kernel_size: int = kernel_size
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, i: int) -> Tuple[Tensor, Tensor]:
        img_path = os.path.join(self.image_dir, self.image_paths[i])
        sharp_image = cv2.imread(img_path)
        sharp_image = cv2.cvtColor(sharp_image, cv2.COLOR_BGR2RGB)
        blur_image = cv2.GaussianBlur(sharp_image, (self.kernel_size, self.kernel_size), sigmaX=0)

        sharp_image_tensor: Tensor = self.transform(sharp_image)
        blur_image_tensor: Tensor = self.transform(blur_image)

        return blur_image_tensor, sharp_image_tensor
