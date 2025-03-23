import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class ImagePairDataset(Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        self.image_dir = image_dir
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(image_path)
        target = image.copy()
        if self.input_transform:
            image = self.input_transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target

