import os

import cv2
from numpy import ndarray
from brisque import BRISQUE

from utils import read_exr, tone_map_reinhard, tone_map_mantiuk, evaluate_image

# enable using OpenEXR with OpenCV
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"
FILE_PATH = os.path.normpath("tone_mapping/sihdr/reference/001.exr")

if __name__ == '__main__':
    image = read_exr(im_path=FILE_PATH)
    tone_mapped_reinhard = tone_map_reinhard(image)
    tone_mapped_mantiuk = tone_map_mantiuk(image)
    cv2.imshow('original', image)
    cv2.imshow('tone_mapped_reinhard', tone_mapped_reinhard)
    cv2.imshow('tone_mapped_mantiuk', tone_mapped_mantiuk)
    print('tone_mapped_reinhard', evaluate_image(image=tone_mapped_reinhard))
    print('tone_mapped_mantiuk', evaluate_image(image=tone_mapped_mantiuk))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
