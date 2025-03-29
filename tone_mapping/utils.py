import os

import cv2
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


def evaluate_image(image: ndarray) -> float:
    metric = BRISQUE(url=False)
    return metric.score(img=image)
