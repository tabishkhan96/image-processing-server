from api.schemas import AllowedBlurType, AllowedFilter, CompressMethod
from api.schemas.schemas import PlacePosition, Position
from api.utils.convert import sizeof_fmt, to_dict
from api.utils.executor import executor
from api.utils.image import *
from fastapi import APIRouter, File, Form, UploadFile, params
from fastapi.exceptions import HTTPException
from fastapi.responses import Response
from pydantic.color import Color
from pydantic.types import NonNegativeInt, PositiveFloat, PositiveInt

router = APIRouter(tags=["tools"])


@router.post("/crop")
async def crop(
    image: UploadFile = File(...),
    x1: PositiveInt = Form(...),
    y1: PositiveInt = Form(...),
    x2: PositiveInt = Form(...),
    y2: PositiveInt = Form(...),
):
    content_type = image.content_type
    params = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    content = await executor.run_task(
        func=process_image_pillow,
        proc_func=crop_image,
        sr_image=image,
        params=params,
    )
    if not content:
        raise HTTPException(
            status_code=422,
            detail=f"Crop is not possible with your coordinates.",
        )
    else:
        return Response(content, media_type=content_type)


@router.post("/scale")
async def scale(
    image: UploadFile = File(...),
    width: Optional[PositiveInt] = Form(default=None),
    height: Optional[PositiveInt] = Form(default=None),
):
    if not (width or height):
        raise HTTPException(status_code=422, detail=f"Width or height is required!")
    else:
        content_type = image.content_type
        params = {"width": width, "height": height}
        content = await executor.run_task(
            func=process_image_pillow,
            proc_func=scale_image,
            sr_image=image,
            params=params,
        )
        return Response(content, media_type=content_type)


@router.post("/rotate")
async def rotate(
    image: UploadFile = File(...),
    angle: float = Form(...),
):

    content_type = image.content_type
    params = {"angle": angle}
    content = await executor.run_task(
        func=process_image_pillow,
        proc_func=rotate_image,
        sr_image=image,
        params=params,
    )
    return Response(content, media_type=content_type)


@router.post("/resize")
async def resize(
    image: UploadFile = File(...),
    width: PositiveInt = Form(...),
    height: PositiveInt = Form(...),
):

    content_type = image.content_type
    params = {"width": width, "height": height}
    content = await executor.run_task(
        func=process_image_pillow,
        proc_func=resize_image,
        sr_image=image,
        params=params,
    )
    return Response(content, media_type=content_type)


@router.post("/thumbnail")
async def thumbnail(
    image: UploadFile = File(...),
    width: PositiveInt = Form(...),
    height: PositiveInt = Form(...),
    round: bool = Form(default=False),
    r_radius: PositiveInt = Form(default=5),
):
    content_type = image.content_type
    params = {
        "width": width,
        "height": height,
        "round": round,
        "r_radius": r_radius,
    }
    content = await executor.run_task(
        func=process_image_pillow,
        proc_func=thumbnail_generator,
        sr_image=image,
        params=params,
    )
    return Response(content, media_type=content_type)


@router.post("/filter")
async def filter(
    image: UploadFile = File(...),
    filter_name: AllowedFilter = Form(...),
):
    content_type = image.content_type
    params = {"filter_name": filter_name}
    content = await executor.run_task(
        func=process_image_pillow,
        proc_func=apply_filter,
        sr_image=image,
        params=params,
    )
    return Response(content, media_type=content_type)


@router.post("/compress")
async def compress(
    image: UploadFile = File(...),
    quality: PositiveInt = Form(default=70),
    compress_method: CompressMethod = Form(default=CompressMethod.default),
    compress_level: PositiveInt = Form(default=9),
):
    content_type = image.content_type
    params = {
        "quality": quality,
        "compress_method": compress_method,
        "compress_level": compress_level,
    }

    if compress_method == CompressMethod.jpegoptim:
        image_bytes = await image.read()
        content = await executor.run_in_shell(
            command=f"jpegoptim --strip-all --max {params['quality']} -",
            input_file=image_bytes,
        )
    else:
        content = await executor.run_task(
            func=process_image_pillow,
            proc_func=compress_image,
            sr_image=image,
            params=params,
        )

    return Response(content, media_type=content_type)


@router.post("/blur")
async def blur(
    image: UploadFile = File(...),
    blur_type: AllowedBlurType = Form(default=AllowedBlurType.gaussian),
    radius: NonNegativeInt = Form(default=5),
):
    content_type = image.content_type
    params = {"blur_type": blur_type, "radius": radius}
    content = await executor.run_task(
        func=process_image_pillow,
        proc_func=apply_blur,
        sr_image=image,
        params=params,
    )
    return Response(content, media_type=content_type)


@router.post("/getColorPalette")
async def color_palette(
    image: UploadFile = File(...),
    max_colors: PositiveInt = Form(default=5),
    display_hex_values: bool = Form(default=True),
):
    params = {"max_colors": max_colors, "display_hex_values": display_hex_values}
    content = await executor.run_task(
        func=process_image_pillow,
        proc_func=color_palette_generator,
        sr_image=image,
        params=params,
    )
    return Response(content, media_type="image/png")


@router.post("/getImageInfo")
async def get_image_info(
    image: UploadFile = File(...),
    extract_exif: bool = Form(default=True),
):
    image_first_bytes = await image.read(size=8)
    if is_image_content(image_first_bytes):
        await image.seek(0)
        image_bytes = await image.read()
        content = await executor.run_in_shell(
            command="exiftool -c '%.6f' -S -",
            input_file=image_bytes,
            convert_output_func=to_dict,
        )
        content["ImageFileSize"] = sizeof_fmt(len(image_bytes))
    else:
        raise HTTPException(
            status_code=422,
            detail="Unable to determine file type. Supported formats: jpg, jpeg, png",
        )

    return content


@router.post("/concat")
async def concat(
    image_1: UploadFile = File(...),
    image_2: UploadFile = File(...),
    position: Position = Form(default=Position.vertical),
):
    content_type = image_1.content_type
    params = {"position": position}
    content = await executor.run_task(
        func=process_images,
        proc_func=concat_images,
        sr_image_1=image_1,
        sr_image_2=image_2,
        params=params,
    )

    return Response(content, media_type=content_type)


@router.post("/watermark")
async def watermark(
    image_1: UploadFile = File(...),
    watermark: UploadFile = File(...),
    position: PlacePosition = Form(default=PlacePosition.center),
    repeat: Optional[bool] = Form(default=False),
):
    content_type = image_1.content_type
    params = {"position": position, "repeat": repeat}
    content = await executor.run_task(
        func=process_images,
        proc_func=place_watermark,
        sr_image_1=image_1,
        sr_image_2=watermark,
        params=params,
    )
    return Response(content, media_type=content_type)


@router.post("/addText")
async def add_text(
    image: UploadFile = File(...),
    text: str = Form(...),
    size: PositiveInt = Form(...),
    color: Color = Form(default="pink"),
    position: PlacePosition = Form(default=PlacePosition.center),
):
    content_type = image.content_type
    params = {"text": text, "size": size, "color": color.as_hex(), "position": position}
    content = await executor.run_task(
        func=process_image_pillow,
        proc_func=draw_text,
        sr_image=image,
        params=params,
    )
    return Response(content, media_type=content_type)


@router.post("/grayscale")
async def get_image_info(
    image: UploadFile = File(...),
):
    content_type = image.content_type
    params = {"mode": "L"}
    content = await executor.run_task(
        func=process_image_pillow,
        proc_func=set_grayscale,
        sr_image=image,
        params=params,
    )
    return Response(content, media_type=content_type)


@router.post("/mirror")
async def mirror(
    image: UploadFile = File(...),
):
    content_type = image.content_type
    content = await executor.run_task(
        func=process_image_pillow,
        proc_func=mirror_apply,
        sr_image=image,
    )
    return Response(content, media_type=content_type)


@router.post("/fit")
async def fit(
    image: UploadFile = File(...),
    width: PositiveInt = Form(...),
    height: PositiveInt = Form(...),
):
    content_type = image.content_type
    params = {"width": width, "height": height}
    content = await executor.run_task(
        func=process_image_pillow,
        proc_func=image_fit,
        sr_image=image,
        params=params,
    )
    return Response(content, media_type=content_type)


@router.post("/brightness")
async def brightness(
    image: UploadFile = File(...),
    factor: PositiveFloat = Form(...),
):
    content_type = image.content_type
    params = {"factor": factor}
    content = await executor.run_task(
        func=process_image_pillow,
        proc_func=set_brightness,
        sr_image=image,
        params=params,
    )
    return Response(content, media_type=content_type)


@router.post("/invert")
async def invert(
    image: UploadFile = File(...),
):
    content_type = image.content_type
    content = await executor.run_task(
        func=process_image_pillow,
        proc_func=invert_apply,
        sr_image=image,
    )
    return Response(content, media_type=content_type)


@router.post("/addBorder")
async def border(
    image: UploadFile = File(...),
    size: int = Form(...),
    color: Color = Form(...),
):
    content_type = image.content_type
    params = {"size": size, "color": color.as_hex()}
    content = await executor.run_task(
        func=process_image_pillow,
        proc_func=add_border,
        sr_image=image,
        params=params,
    )
    return Response(content, media_type=content_type)


@router.post("/addNoise")
async def noise(
    image: UploadFile = File(...),
    variance: PositiveFloat = Form(...),
):
    content_type = image.content_type
    params = {"variance": variance}
    content = await executor.run_task(
        func=process_image_cv,
        proc_func=add_noise,
        sr_image=image,
        params=params,
    )
    return Response(content, media_type=content_type)
