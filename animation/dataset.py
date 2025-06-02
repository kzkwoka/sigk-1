import os
import re
from glob import glob

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset


class AnimationDataset(Dataset):
    def __init__(self, data_path="./Depth", transform=None):
        self.data_path = data_path
        self.transform = torchvision.transforms.ToTensor() if transform is None else transform
        pattern = re.compile(r"image_(\d+)\.png")
        self.triplets = []
        self.sequence_len = 10
        self.mid_offset = self.sequence_len // 2

        candidate_folders = [
            os.path.dirname(p)
            for p in glob(os.path.join(self.data_path, "**", "image_*.png"), recursive=True)
        ]
        unique_folders = sorted(set(candidate_folders))

        for folder in unique_folders:
            files = os.listdir(folder)
            indices = sorted([
                int(match.group(1))
                for fname in files
                if (match := pattern.match(fname))
            ])
            index_set = set(indices)

            for i in indices:
                mid = i + self.mid_offset
                end = i + self.sequence_len - 1
                if mid in index_set and end in index_set:
                    self.triplets.append((
                        os.path.join(folder, f"image_{i:04d}.png"),
                        os.path.join(folder, f"image_{mid:04d}.png"),
                        os.path.join(folder, f"image_{end:04d}.png")
                    ))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        i0, i1, i2 = self.triplets[idx]

        def load_image(path):
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



