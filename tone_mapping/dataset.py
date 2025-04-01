import os

import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from utils import norm_mean, mu_law, multi_exposure


class TMDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.images = sorted(os.listdir(image_dir))

        self.tensor = ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        i_src = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        i_src = np.maximum(cv2.cvtColor(i_src, cv2.COLOR_BGR2RGB), 0.0)
        i_hdr = norm_mean(i_src)
        i_mu = self.tensor(mu_law(i_hdr))
        i_low, i_mid, i_high = multi_exposure(self.tensor(i_hdr))
        return i_low, i_mid, i_high, i_mu, i_src


if __name__ == '__main__':
    dataset = TMDataset('tone_mapping/sihdr/resized')
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
    print(dataset[0][2].shape)
    print(dataset[0][3].shape)