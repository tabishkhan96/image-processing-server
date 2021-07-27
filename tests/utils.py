import math
import os
from io import BytesIO
from typing import Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageOps
from PIL.Image import Image as ImageObj


def get_image_size(image: Union[str, bytes]) -> Tuple[int, int]:
    im_obj = get_pillow_image_obj(image)
    size = im_obj.size
    return size


def get_pillow_image_obj(image):
    if isinstance(image, str):
        img_obj = Image.open(image)
    elif isinstance(image, bytes):
        img_obj = Image.open(BytesIO(image))
    return img_obj


def calc_size_after_crop(w: int, h: int, x1: int, y1: int, x2: int, y2: int):
    width = w - (x1 + x2)
    height = h - (y1 + y2)
    return width, height


def calc_thumb_size(w1: int, h1: int, w2: int, h2: int) -> Tuple[int, int]:
    def round_aspect(number, key):
        return max(min(math.floor(number), math.ceil(number), key=key), 1)

    x, y = math.floor(w2), math.floor(h2)
    if x >= w1 and y >= h1:
        return

    aspect = w1 / h1
    if x / y >= aspect:
        x = round_aspect(y * aspect, key=lambda n: abs(aspect - n / y))
    else:
        y = round_aspect(x / aspect, key=lambda n: 0 if n == 0 else abs(aspect - x / n))
    size = (x, y)

    return size


def is_images_similar(im_obj_1: ImageObj, im_obj_2: Image) -> bool:
    diff = ImageChops.difference(im_obj_1, im_obj_2)
    if diff.getbbox():
        return False
    return True


def is_blurred_image(im_obj: ImageObj, threshold: float = 50.0) -> bool:

    img_byte_arr = BytesIO()
    im_obj.save(img_byte_arr, format=im_obj.format)

    image = np.asarray(bytearray(img_byte_arr.getvalue()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return var < threshold


def get_file_size(file_path: str) -> int:
    size_in_bytes = os.path.getsize(file_path)
    return size_in_bytes


def is_grayscaled(image):
    im_obj = get_pillow_image_obj(image)

    if im_obj.mode not in ("L", "RGB"):
        raise ValueError("Unsuported image mode")

    if im_obj.mode == "RGB":
        rgb = im_obj.split()
        if ImageChops.difference(rgb[0], rgb[1]).getextrema()[1] != 0:
            return False
        if ImageChops.difference(rgb[0], rgb[2]).getextrema()[1] != 0:
            return False
    return True


def remove_border(im_obj: ImageObj, size: int) -> ImageObj:
    im_obj = ImageOps.crop(im_obj, border=size)
    return im_obj
