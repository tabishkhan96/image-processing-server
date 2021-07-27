import colorsys
import os
from io import BytesIO
from typing import IO, Callable, Optional, Tuple, Union

import colorific
import numpy as np
from api.schemas import CompressMethod
from api.schemas.schemas import PlacePosition, Position
from api.utils.convert import sizeof_fmt
from colorific import norm_color, rgb_to_hex
from fastapi.datastructures import UploadFile
from PIL import (Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont,
                 ImageOps)
from PIL.ExifTags import TAGS
from PIL.Image import Image as ImageObj
from skimage.util import random_noise

FILTERS = {
    "contour": ImageFilter.CONTOUR,
    "edges": ImageFilter.EDGE_ENHANCE,
    "sharpen": ImageFilter.SHARPEN,
    "detail": ImageFilter.DETAIL,
    "smooth": ImageFilter.SMOOTH_MORE,
    "emboss": ImageFilter.EMBOSS,
    "lighten": ImageFilter.MaxFilter,
    "darken": ImageFilter.MinFilter,
}

BLUR = {
    "box": ImageFilter.BoxBlur,
    "gaussian": ImageFilter.GaussianBlur,
}


def get_pillow_im_obj(image: IO) -> ImageObj:
    image_obj = Image.open(image)
    return image_obj


def thumbnail_generator(
    im_obj: ImageObj,
    width: int,
    height: int,
    round: Optional[bool] = False,
    r_radius: Optional[int] = 15,
) -> ImageObj:
    if round:
        im_obj.thumbnail((width * 2, height * 2))
        im_obj = round_corners(im_obj, radius=r_radius)
    else:
        im_obj.thumbnail((width, height))

    return im_obj


def round_corners(im_obj: ImageObj, radius: int) -> ImageObj:
    circle = Image.new("L", (radius * 2, radius * 2), 0)
    draw = ImageDraw.Draw(circle)
    draw.ellipse((0, 0, radius * 2, radius * 2), fill=255)
    alpha = Image.new("L", im_obj.size, 255)
    w, h = im_obj.size
    alpha.paste(circle.crop((0, 0, radius, radius)), (0, 0))
    alpha.paste(circle.crop((0, radius, radius, radius * 2)), (0, h - radius))
    alpha.paste(circle.crop((radius, 0, radius * 2, radius)), (w - radius, 0))
    alpha.paste(
        circle.crop((radius, radius, radius * 2, radius * 2)),
        (w - radius, h - radius),
    )
    im_obj.putalpha(alpha)

    # TODO: ANTIALIAS corners
    im_obj = im_obj.resize((w // 2, h // 2), resample=Image.ANTIALIAS)
    return im_obj


def crop_image(im_obj: ImageObj, x1: int, y1: int, x2: int, y2: int) -> ImageObj:
    w, h = im_obj.size
    if is_crop_possible(w, h, x1, x2, y1, y2):
        im_cropped = im_obj.crop((x1, y1, x2, y2))
        return im_cropped
    else:
        return False


def scale_image(
    im_obj: ImageObj,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> ImageObj:
    w, h = im_obj.size
    if width and height:
        max_size = (width, height)
    elif width:
        max_size = (width, h)
    elif height:
        max_size = (w, height)

    im_obj.thumbnail(max_size, Image.ANTIALIAS)

    return im_obj


def rotate_image(im_obj: ImageObj, angle: float) -> ImageObj:
    im_obj = im_obj.rotate(angle, expand=True, fillcolor=(255, 255, 255, 0))
    return im_obj


def resize_image(im_obj: ImageObj, width: int, height: int) -> ImageObj:
    return im_obj.resize((width, height))


def save_to_file_like(image: ImageObj, format: str, **params) -> bytes:
    content = BytesIO()
    image.save(content, format=format, **params)
    # content.seek(0)
    return content


def file_format(filename: str) -> str:
    _, ext = os.path.splitext(filename)
    if ext in (".jpg", ".jpeg"):
        return "JPEG"
    else:
        return "PNG"


def process_image_pillow(
    sr_image: UploadFile,
    proc_func: Callable,
    params: Optional[dict[str, Union[str, int]]] = dict(),
) -> Union[bytes, bool]:
    im_obj: Image = get_pillow_im_obj(sr_image.file)
    im_func_result = proc_func(im_obj, **params)
    if isinstance(im_func_result, dict):
        return im_func_result
    if isinstance(im_func_result, bool):
        return im_func_result
    im_format = file_format(sr_image.filename)
    im_content_file = save_to_file_like(im_func_result, format=im_format, **params)
    return im_content_file.getvalue()


def process_images(
    sr_image_1: UploadFile,
    sr_image_2: UploadFile,
    proc_func: Callable,
    params: Optional[dict[str, Union[str, int]]] = dict(),
) -> Union[bytes, bool]:
    im_obj_1: Image = get_pillow_im_obj(sr_image_1.file)
    im_obj_2: Image = get_pillow_im_obj(sr_image_2.file)
    im_func_result = proc_func(im_obj_1, im_obj_2, **params)
    im_format = file_format(sr_image_1.filename)
    im_content_file = save_to_file_like(im_func_result, format=im_format, **params)
    return im_content_file.getvalue()


def process_image_cv(
    sr_image: UploadFile,
    proc_func: Callable,
    params: Optional[dict[str, Union[str, int]]] = dict(),
) -> bytes:
    im_arr = np.asarray(get_pillow_im_obj(sr_image.file))
    im_func_result = proc_func(im_arr, **params)
    if isinstance(im_func_result, np.ndarray):
        im = Image.fromarray(im_func_result)
        im_format = file_format(sr_image.filename)
        content = save_to_file_like(im, format=im_format, **params)
        return content.getvalue()


def concat_images(
    im_obj_1: ImageObj,
    im_obj_2: ImageObj,
    position: str,
) -> ImageObj:
    if position == Position.horizontal:
        width = im_obj_1.width + im_obj_2.width
        height = min(im_obj_1.height, im_obj_2.height)
        im_obj_1 = scale_image(im_obj_1, width, height)
        im_obj_2 = scale_image(im_obj_2, width, height)
        new_image = Image.new("RGB", (im_obj_1.width + im_obj_2.width, height))
        new_image.paste(im_obj_1, (0, 0))
        new_image.paste(im_obj_2, (im_obj_1.width, 0))
    else:
        height = im_obj_1.height + im_obj_2.height
        width = min(im_obj_1.width, im_obj_2.width)
        im_obj_1 = scale_image(im_obj_1, width, height)
        im_obj_2 = scale_image(im_obj_2, width, height)
        new_image = Image.new("RGB", (width, im_obj_1.height + im_obj_2.height))
        new_image.paste(im_obj_1, (0, 0))
        new_image.paste(im_obj_2, (0, im_obj_1.height))

    return new_image


def calc_position(w1: int, h1: int, w2: int, h2: int, position: str) -> Tuple[int, int]:
    if position == PlacePosition.center:
        return (int(w1 / 2) - int(w2 / 2), int(h1 / 2) - int(h2 / 2))
    elif position == PlacePosition.top:
        return (int(w1 / 2) - int(w2 / 2), 0)
    elif position == PlacePosition.bottom:
        return (0, h1 - h2)


def place_watermark(
    im_obj_1: ImageObj,
    watermark: ImageObj,
    position: str,
    repeat: bool,
):
    width, height = im_obj_1.size
    new_image = Image.new("RGB", (width, height), (0, 0, 0, 0))
    new_image.paste(im_obj_1, (0, 0))
    if repeat:
        w_width, w_height = watermark.size
        for left in range(0, width, w_width):
            for top in range(0, height, w_height):
                new_image.paste(watermark, (left, top), mask=watermark)

    else:
        watermark = scale_image(watermark, width)
        position = calc_position(
            width, height, watermark.width, watermark.height, position
        )
        new_image.paste(watermark, position, mask=watermark)
        return new_image

    return new_image


def draw_text(
    im_obj: ImageObj,
    text: str,
    size: int,
    color: str,
    position: str,
) -> ImageObj:
    width, height = im_obj.size
    draw = ImageDraw.Draw(im_obj)
    font = ImageFont.truetype("files/mononoki.ttf", size=size, encoding="utf-8")
    # font = ImageFont.load_default()
    font_width, font_height = font.getsize(text)
    position = calc_position(width, height, font_width, font_height, position=position)
    # new_width = (width - font_width) / 2
    # new_height = (height - font_height) / 2
    draw.text(position, text, font=font, fill=color)
    return im_obj


def add_noise(im_arr, variance: float):
    noise_img = random_noise(im_arr, mode="gaussian", var=variance)
    noise_img = np.array(255 * noise_img, dtype="uint8")
    return noise_img


def apply_filter(im_obj: ImageObj, filter_name: str) -> ImageObj:
    filter_func = FILTERS[filter_name]
    filtered_image = im_obj.filter(filter_func)
    return filtered_image


def apply_blur(im_obj: ImageObj, blur_type: str, radius: int) -> ImageObj:
    blur_func = BLUR[blur_type]
    blurred_image = im_obj.filter(blur_func(radius=radius))
    return blurred_image


def compress_image(
    im_obj: ImageObj,
    quality: int,
    compress_method: str,
    compress_level: int,
) -> ImageObj:
    # TODO: Try to improve logic and add new methods.

    if im_obj.format == "JPEG":
        return im_obj
    if im_obj.format == "PNG":
        if compress_method == CompressMethod.quantize:
            im_obj = im_obj.quantize(colors=256)
        if compress_method == CompressMethod.rgb:
            im_obj = im_obj.convert("RGB")
    return im_obj


def color_palette_generator(
    im_obj: ImageObj,
    max_colors: int = 5,
    display_hex_values: bool = True,
) -> ImageObj:
    palette = colorific.extract_colors(im_obj, max_colors=max_colors)
    im_obj = save_palette_as_image(palette, display_hex_values=display_hex_values)
    return im_obj


def save_palette_as_image(palette, display_hex_values=False) -> ImageObj:
    size = (128 * len(palette.colors), 128)
    im = Image.new("RGB", size)
    draw = ImageDraw.Draw(im)
    for i, c in enumerate(palette.colors):
        v = colorsys.rgb_to_hsv(*norm_color(c.value))[2]
        (x1, y1) = (i * 128, 0)
        (x2, y2) = ((i + 1) * 128 - 1, 127)
        draw.rectangle([(x1, y1), (x2, y2)], fill=c.value)
        if display_hex_values:
            if v < 0.6:
                draw.text((x1 + 4, y1 + 4), rgb_to_hex(c.value), (90, 90, 90))
                draw.text((x1 + 3, y1 + 3), rgb_to_hex(c.value))
            else:
                draw.text((x1 + 4, y1 + 4), rgb_to_hex(c.value), (230, 230, 230))
                draw.text((x1 + 3, y1 + 3), rgb_to_hex(c.value), (0, 0, 0))

    return im


def extract_image_info(im_obj: ImageObj, extract_exif=True) -> dict:
    info = {}
    sizeb = len(im_obj.tobytes())
    info["basic"] = {
        "width": im_obj.width,
        "height": im_obj.height,
        "image_file_size": sizeof_fmt(sizeb),
        "format": im_obj.format,
        "mode": im_obj.mode,
    }
    if extract_exif:
        info["exif_data"] = {}
        exif_data = im_obj.getexif()
        for tag, value in exif_data.items():
            decoded = TAGS.get(tag, tag)
            if isinstance(value, (str, int)):
                if isinstance(value, str):
                    value = value.strip()

                if value:
                    info["exif_data"][decoded] = value

    return info


def is_image_content(image_bytes: bytes) -> bool:
    png_file_signature = "89504e470d0a1a0a"
    jpg_jpeg = ("ffd8ffe0", "ffd8ffe3", "ffd8ffe2", "ffd8ffe1")
    if image_bytes.hex() == png_file_signature:
        return True
    elif image_bytes.hex()[:8] in jpg_jpeg:
        return True

    return False


async def check_for_image(image: UploadFile) -> bool:
    first_bytes = await image.read(size=8)
    png_file_signature = "89504e470d0a1a0a"
    jpg_jpeg = ("ffd8ffe0", "ffd8ffe3", "ffd8ffe2", "ffd8ffe1")
    if first_bytes.hex() == png_file_signature:
        return True
    elif first_bytes.hex()[:8] in jpg_jpeg:
        return True

    return False


def set_grayscale(im_obj: ImageObj, mode: str) -> ImageObj:
    gray_scaled = im_obj.convert(mode=mode)
    return gray_scaled


def mirror_apply(im_obj: ImageObj) -> ImageObj:
    mirrow_image = ImageOps.mirror(im_obj)
    return mirrow_image


def image_fit(im_obj: ImageObj, width: int, height: int) -> ImageObj:
    im_obj_fitted = ImageOps.fit(im_obj, (width, height))
    return im_obj_fitted


def set_brightness(im_obj: ImageObj, factor: int) -> ImageObj:
    enhancer = ImageEnhance.Brightness(im_obj)
    im_obj = enhancer.enhance(factor=factor)
    return im_obj


def invert_apply(im_obj: ImageObj) -> ImageObj:
    inverted = ImageOps.invert(im_obj)
    return inverted


def add_border(im_obj: ImageObj, size: int, color: str) -> ImageObj:
    im_obj = ImageOps.expand(im_obj, border=size, fill=color)
    return im_obj


def is_crop_possible(w: int, h: int, x1: int, x2: int, y1: int, y2: int) -> bool:
    crop_possible = True
    if not 0 <= x1 < w:
        crop_possible = False
    if not 0 < x2 <= w:
        crop_possible = False
    if not 0 <= y1 < h:
        crop_possible = False
    if not 0 < y2 <= h:
        crop_possible = False
    if not x1 < x2:
        crop_possible = False
    if not y1 < y2:
        crop_possible = False

    return crop_possible
