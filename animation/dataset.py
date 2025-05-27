import os
import re

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset


class AnimationDataset(Dataset):
    def __init__(self, data_path="./Depth/201403121135", transform=None):
        self.data_path = data_path
        self.transform = torchvision.transforms.ToTensor() if transform is None else transform
        pattern = re.compile(r"image_(\d+)\.png")

        self.indices = sorted([
            int(match.group(1))
            for fname in os.listdir(data_path)
            if (match := pattern.match(fname))
        ])
        index_set = set(self.indices)

        self.triplets = []
        self.sequence_len = 5
        self.mid_offset = self.sequence_len // 2
        for i in self.indices:
            if all((i + offset) in index_set for offset in range(self.sequence_len)):
                self.triplets.append((
                    i,
                    i + self.mid_offset,
                    i + self.sequence_len - 1
                ))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        i0, i1, i2 = self.triplets[idx]

        def load_image(index):
            path = os.path.join(self.data_path, f"image_{index:04d}.png")
            depth = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            # image = Image.open(path) # Convert to grayscale
            # return self.transform(image) if self.transform else image

            min_depth = 160.0
            max_depth = 32000.0
            depth = np.clip(depth, min_depth, max_depth)
            depth = (depth - min_depth) / (max_depth - min_depth)
            return torch.from_numpy(depth).unsqueeze(0)

        return load_image(i0), load_image(i1), load_image(i2)


if __name__ == "__main__":
    d = AnimationDataset()
    print(f"Dataset contains {len(d)} files.")
    # print("First file:", d[0])
    print("", d[0][0].shape) #[1, 240, 320]
    print(d[0][0].min().item(), d[0][0].max().item())

    # ult_low, ult_high = np.inf, -np.inf
    # for img, _, _ in d:
    #     if img.min() < ult_low:
    #         ult_low = img.min().item()
    #     if img.max() > ult_high:
    #         ult_high = img.max().item()
    # print(f"Min: {ult_low:.4f}, Max: {ult_high:.4f}")



