import os
from random import randint

import pytest
from api.main import create_app
from api.utils.image import FILTERS, is_crop_possible, mirror_apply
from fastapi.testclient import TestClient

from tests.utils import *

app = create_app()
client = TestClient(app)

images = [
    (os.path.join("tests/images", path))
    for path in os.listdir("tests/images")
    if path.endswith(("jpeg", "jpg", "png"))
]
palettes = [("tests/images/4.jpg", "tests/images/4-palette-with-hex.jpg")]
jpeg_images = [i for i in images if i.endswith((".jpeg", "jpg"))]
png_images = [i for i in images if i.endswith((".png"))]


@pytest.mark.parametrize("image_path", images)
def test_crop(image_path):
    ow, oh = get_image_size(image_path)
    x1, y1, x2, y2 = [randint(1, 200) for _ in range(4)]
    if is_crop_possible(ow, oh, x1, x2, y1, y2):
        files = {"image": open(image_path, "rb")}
        params = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        response = client.post("/crop", files=files, data=params)
        w, h = get_image_size(response.content)
        assert (w, h) == (x2 - x1, y2 - y1)


@pytest.mark.parametrize("image_path", images)
def test_thumbnail(image_path):
    ow, oh = get_image_size(image_path)
    nwp = randint(1, 200)
    nhp = randint(1, 300)
    ew, eh = calc_thumb_size(ow, oh, nwp, nhp)
    files = {"image": open(image_path, "rb")}
    params = {"width": nwp, "height": nhp}
    response = client.post("/thumbnail", files=files, data=params)
    nw, nh = get_image_size(response.content)
    assert nw == ew and nh == eh


@pytest.mark.parametrize("image_path", images)
def test_filter(image_path):
    im_obj = get_pillow_image_obj(image_path)
    params = {"filter_name": "emboss"}
    files = {"image": open(image_path, "rb")}
    response = client.post("/filter", files=files, data=params)
    im_obj_res = get_pillow_image_obj(response.content)
    assert not is_images_similar(im_obj, im_obj_res)


@pytest.mark.parametrize("image_path", images)
def test_blur_applied(image_path):
    params = {"blur_type": "gaussian", "radius": randint(5, 15)}
    files = {"image": open(image_path, "rb")}
    response = client.post("/blur", files=files, data=params)
    im_obj = get_pillow_image_obj(response.content)
    assert is_blurred_image(im_obj) == True


@pytest.mark.parametrize("image_path", images)
def test_blur_not_applied(image_path):
    params = {"blur_type": "gaussian", "radius": 0}
    files = {"image": open(image_path, "rb")}
    response = client.post("/blur", files=files, data=params)
    im_obj = get_pillow_image_obj(response.content)
    assert is_blurred_image(im_obj) == False


@pytest.mark.parametrize("image_path, image_palette", palettes)
def test_color_palette(image_path, image_palette):
    params = {"max_colors": 5, "display_hex_values": True}
    files = {"image": open(image_path, "rb")}
    response = client.post("/getColorPalette", files=files, params=params)
    im_obj_1 = get_pillow_image_obj(response.content)
    im_obj_2 = get_pillow_image_obj(image_palette)
    assert is_images_similar(im_obj_1, im_obj_2)
    assert im_obj_1.size == im_obj_2.size


@pytest.mark.parametrize("image_path", jpeg_images)
def test_compress_jpeg(image_path):
    size_before = get_file_size(image_path)
    files = {"image": open(image_path, "rb")}
    params = {
        "compress_method": "jpegoptim",
        "quality": randint(30, 60),
        "compress_level": 8,
    }
    response = client.post("/compress", files=files, params=params)
    size_after = len(response.content)
    percent = round(100 - (100 / size_before) * size_after)
    assert percent >= 0


@pytest.mark.parametrize("image_path", png_images)
def test_compress_png(image_path):
    size_before = get_file_size(image_path)
    files = {"image": open(image_path, "rb")}
    params = {"compress_method": "rgb", "quality": randint(40, 90), "compress_level": 8}
    response = client.post("/compress", files=files, params=params)
    size_after = len(response.content)
    percent = round(100 - (100 / size_before) * size_after)
    assert percent >= 0


@pytest.mark.parametrize("image_path", images)
def test_get_image_info(image_path):
    params = {"extract_exif": False}
    files = {"image": open(image_path, "rb")}
    response = client.post("/getImageInfo", files=files, data=params)
    assert response.status_code == 200
    json_data = response.json()
    assert json_data.get("ImageWidth")
    assert json_data.get("ImageHeight")
    assert json_data.get("ImageFileSize")


@pytest.mark.parametrize(
    "image_path", ["tests/images/file1.txt", "tests/images/file2.exe"]
)
def test_get_image_info_fail_file_format(image_path):
    params = {"extract_exif": False}
    files = {"image": open(image_path, "rb")}
    response = client.post("/getImageInfo", files=files, data=params)
    assert response.status_code == 422


@pytest.mark.parametrize("image_path", images)
def test_grayscale(image_path):
    files = {"image": open(image_path, "rb")}
    response = client.post("/grayscale", files=files)
    assert is_grayscaled(response.content)


@pytest.mark.parametrize("image_path", images)
def test_mirror(image_path):
    files = {"image": open(image_path, "rb")}
    response = client.post("/mirror", files=files)
    im_obj_1 = get_pillow_image_obj(image_path)
    im_obj_2 = get_pillow_image_obj(response.content)
    assert not is_images_similar(im_obj_1, im_obj_2)


@pytest.mark.parametrize("image_path", images)
def test_fit(image_path):
    files = {"image": open(image_path, "rb")}
    params = {"width": randint(200, 400), "height": randint(200, 400)}
    response = client.post("/fit", files=files, data=params)
    ow, oh = get_image_size(image_path)
    assert (ow, oh) != get_image_size(response.content)


@pytest.mark.parametrize("image_path", images)
def test_border(image_path):
    ow, oh = get_image_size(image_path)
    files = {"image": open(image_path, "rb")}
    size = randint(10, 50)
    params = {"size": size, "color": "black"}
    response = client.post("/addBorder", files=files, data=params)
    im_obj = get_pillow_image_obj(response.content)
    im_obj_no_border = remove_border(im_obj, size=size)
    assert (ow, oh) == im_obj_no_border.size


@pytest.mark.parametrize("image_path", images)
def test_resize(image_path):
    size = (randint(100, 300), randint(100, 300))
    files = {"image": open(image_path, "rb")}
    params = {"width": size[0], "height": size[1]}
    response = client.post("/resize", files=files, data=params)
    im_obj = get_pillow_image_obj(response.content)
    assert size == im_obj.size
