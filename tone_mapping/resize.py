import os
import cv2
import argparse
import random
from utils import read_exr
from tqdm import tqdm


def resize_image(image_path: str, output_path: str, size: int) -> None:
    image = read_exr(im_path=image_path)
    resized_image = cv2.resize(image,
                               dsize=None,
                               fx=size / max(image.shape[:2]),
                               fy=size / max(image.shape[:2]),
                               interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(output_path, resized_image)


def resize_images(input_folder: str, output_folder: str, size: int) -> list:
    os.makedirs(output_folder, exist_ok=True)
    image_paths = []
    for filename in tqdm(os.listdir(input_folder), desc="Resizing"):
        if filename.endswith('.exr'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            resize_image(input_path, output_path, size)
            image_paths.append(output_path)
    return image_paths


def split_dataset(image_paths: list, split_root: str, seed: int = 42):
    random.seed(seed)
    random.shuffle(image_paths)
    n = len(image_paths)
    n_tv = int(n * 0.75)
    n_train = int(n_tv * 0.8)
    n_valid = n_tv - n_train

    splits = {
        'train': image_paths[:n_train],
        'valid': image_paths[n_train:n_train + n_valid],
        'test': image_paths[n_tv:]
    }

    for split, paths in splits.items():
        split_dir = os.path.join(split_root, split)
        os.makedirs(split_dir, exist_ok=True)
        for path in tqdm(paths, desc=f"Moving to {split}"):
            filename = os.path.basename(path)
            os.rename(path, os.path.join(split_dir, filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default='tone_mapping/sihdr/reference/')
    parser.add_argument('--resized_folder', type=str, default='tone_mapping/sihdr/resized/')
    parser.add_argument('--split_folder', type=str, default='tone_mapping/sihdr/split/')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split_only', action='store_true')
    args = parser.parse_args()

    if args.split_only:
        image_paths = [os.path.join(args.resized_folder, f)
                       for f in os.listdir(args.resized_folder) if f.endswith('.exr')]
        split_dataset(image_paths, args.split_folder, args.seed)
    else:
        image_paths = resize_images(args.input_folder, args.resized_folder, args.size)
        split_dataset(image_paths, args.split_folder, args.seed)
