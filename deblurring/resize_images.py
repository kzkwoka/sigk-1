import argparse
import os
from typing import Tuple

from PIL import Image


def resize_images(source_dir: str, target_dir: str, resize_shape: Tuple[int, int]) -> None:
    os.makedirs(target_dir, exist_ok=True)

    for filename in os.listdir(source_dir):
        if filename.lower().endswith(".png"):
            path = os.path.join(source_dir, filename)
            image = Image.open(path)
            image_resized = image.resize(resize_shape, Image.LANCZOS)
            image_resized.save(os.path.join(target_dir, filename))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resize all PNG images in a directory.")
    parser.add_argument("--source_dir", type=str, required=True, help="Path to the source directory with images")
    parser.add_argument("--target_dir", type=str, required=True, help="Path to save resized images")
    parser.add_argument("--width", type=int, default=256, help="Target width")
    parser.add_argument("--height", type=int, default=256, help="Target height")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resize_shape = (args.width, args.height)
    resize_images(args.source_dir, args.target_dir, resize_shape)


if __name__ == "__main__":
    main()
