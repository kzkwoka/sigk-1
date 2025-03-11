import os
from PIL import Image
from torch.utils.data import Dataset


class ImagePairDataset(Dataset):
    def __init__(self, image_dir, transform=None, original_transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.original_transform = original_transform
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(image_path)
        image = self.original_transform(image) if self.original_transform else image

        original = image.copy()
        transformed = self.transform(image) if self.transform else image
        return original, transformed
