import os

import cv2
import numpy as np
import torch
from numpy import ndarray
from brisque import BRISQUE

# enable using OpenEXR with OpenCV
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"


def read_exr(im_path: str) -> ndarray:
    return cv2.imread(
        filename=im_path,
        flags=cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
    )


def tone_map_reinhard(image: ndarray) -> ndarray:
    tonemap_operator = cv2.createTonemapReinhard(
        gamma=2.2,
        intensity=0.0,
        light_adapt=0.0,
        color_adapt=0.0
    )
    result = tonemap_operator.process(src=image)
    return result


def tone_map_mantiuk(image: ndarray) -> ndarray:
    tonemap_operator = cv2.createTonemapMantiuk(
        gamma=2.2,
        scale=0.85,
        saturation=1.2
    )
    result = tonemap_operator.process(src=image)
    return result


def evaluate_batch(images: torch.Tensor) -> float:
    b = 0
    for i in images:
        b += evaluate_image(i)
    b /= len(images)
    return b


def evaluate_image(image) -> float:
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
    metric = BRISQUE(url=False)
    return metric.score(img=image)


# image transforms
def norm_mean(img):
    img = 0.5 * img / img.mean()
    return img


def mu_law(img):
    median_value = np.median(img)
    scale = 8.759 * np.power(median_value, 2.148) + 0.1494 * np.power(median_value, -2.067)
    out = np.log(1 + scale * img) / np.log(1 + scale)
    return out


def multi_exposure(img):
    x_p = 1.21497
    log2 = torch.log(torch.tensor(2.0))

    c_start = torch.log(x_p / img.max()) / log2
    c_end = torch.log(x_p / torch.quantile(img, 0.5)) / log2

    exp_values = torch.tensor([c_start, (c_start + c_end) / 2, c_end])

    sc_factors = torch.pow(torch.sqrt(torch.tensor(2.0)), exp_values).view(-1, 1, 1, 1)
    img_scaled = img * sc_factors

    return torch.minimum(img_scaled, torch.tensor(1.0))
