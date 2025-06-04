import shutil
import random
from pathlib import Path

def make_dirs(*dirs):
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def move_folders(folders, target_dir):
    for folder in folders:
        shutil.move(str(folder), target_dir / folder.name)

def split_dataset(src_dir, dest_root, train_ratio=0.8, valid_ratio=0.1, seed=42):
    train_dir = dest_root / "train"
    valid_dir = dest_root / "valid"
    test_dir = dest_root / "test"
    make_dirs(train_dir, valid_dir, test_dir)

    all_folders = sorted([p for p in src_dir.iterdir() if p.is_dir()])
    random.seed(seed)
    random.shuffle(all_folders)

    total = len(all_folders)
    n_train = int(train_ratio * total)
    n_valid = int(valid_ratio * total)
    n_test = total - n_train - n_valid

    move_folders(all_folders[:n_train], train_dir)
    move_folders(all_folders[n_train:n_train + n_valid], valid_dir)
    move_folders(all_folders[n_train + n_valid:], test_dir)

    print(f"✅ Moved {n_train} folders to train/, {n_valid} to valid/, {n_test} to test/")

if __name__ == "__main__":
    src = Path("frames_merged/frames")
    dest = Path("frames_merged")
    split_dataset(src, dest, train_ratio=0.8, valid_ratio=0.1, seed=42)
