import os
import cv2
from utils import read_exr


def resize_image(image_path: str, output_folder: str, size: int) -> None:
    image = read_exr(im_path=image_path)
    resized_image = cv2.resize(image,
                               dsize=None,
                               fx=size / max(image.shape[:2]),
                               fy=size / max(image.shape[:2]),
                               interpolation=cv2.INTER_CUBIC)
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, resized_image)


if __name__ == '__main__':
    input_folder = 'tone_mapping/sihdr/reference/'
    output_folder = 'tone_mapping/sihdr/resized/'
    target_size = 256  # for longer side

    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith('.exr'):
            file_path = os.path.join(input_folder, filename)
            resize_image(file_path, output_folder, target_size)
