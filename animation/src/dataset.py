from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class AnimationTripletDataset(Dataset):
    def __init__(self, root_dir, optical_flow=False, ext="jpg"):
        self.root_dir = Path(root_dir)
        self.flow_dir = None
        self.optical_flow = optical_flow
        self.ext = ext
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),  # scales to [0, 1]
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # to [-1, 1]
            ]
        )
        self.triplets = []

        for subdir in sorted(self.root_dir.iterdir()):
            if not subdir.is_dir():
                continue
            frames = sorted(subdir.glob(f"*.{self.ext}"))
            for i in range(1, len(frames) - 1):
                self.triplets.append((frames[i - 1], frames[i], frames[i + 1]))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        p0, p1, p2 = self.triplets[idx]
        i0 = Image.open(p0).convert("RGB")
        i1 = Image.open(p1).convert("RGB")
        i2 = Image.open(p2).convert("RGB")
        img0 = self.transform(i0)
        img1 = self.transform(i1)
        img2 = self.transform(i2)
        if self.optical_flow:
            flow = self._get_flow(i0, i2)
            return img0, img1, img2, flow
        return img0, img1, img2

    def _get_flow(self, prev, next):
        if self.flow_dir is None:
            prev = cv2.cvtColor(np.array(prev), cv2.COLOR_RGB2GRAY)
            next = cv2.cvtColor(np.array(next), cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev, next, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )

            # Convert flow to tensor (H, W, 2) -> (2, H, W)
            flow = flow.astype(np.float32) / 255.0  # Scale to reduce memory
            flow = torch.from_numpy(flow).permute(2, 0, 1)
        else:
            flow = torch.load(self.flow_dir / f"flow_{prev.stem}.pt")
        return flow


def visualize_flow(flow):
    hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[0].numpy(), flow[1].numpy())
    hsv[..., 0] = ang * 180 / np.pi / 2  # Hue
    hsv[..., 1] = 255  # Saturation
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


if __name__ == "__main__":
    dataset = AnimationTripletDataset("frames_merged/train")
    for i in range(5):
        img0, img1, img2, flow = dataset[i]
        print(f"Sample {i}: {img0.shape}, {img1.shape}, {img2.shape}, {flow.shape}")
        # vis = visualize_flow(flow)
        # plt.imshow(vis)
        # plt.axis("off")
        # plt.show()
