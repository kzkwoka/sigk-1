from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class AnimationTripletDataset(Dataset):
    def __init__(self, root_dir, ext="jpg"):
        self.root_dir = Path(root_dir)
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
        img0 = self.transform(Image.open(p0).convert("RGB"))
        img1 = self.transform(Image.open(p1).convert("RGB"))
        img2 = self.transform(Image.open(p2).convert("RGB"))
        return img0, img1, img2
