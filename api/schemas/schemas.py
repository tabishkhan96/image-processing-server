from enum import Enum


class AllowedFilter(str, Enum):
    contour = "contour"
    edges = "edges"
    sharpen = "sharpen"
    detail = "detail"
    smooth = "smooth"
    emboss = "emboss"
    lighten = "lighten"
    darken = "darken"


class AllowedBlurType(str, Enum):
    box = "box"
    gaussian = "gaussian"


class CompressMethod(str, Enum):
    quantize = "quantize"
    rgb = "rgb"
    default = "default"
    jpegoptim = "jpegoptim"


class Position(str, Enum):
    vertical = "vertical"
    horizontal = "horizontal"


class PlacePosition(str, Enum):
    center = "center"
    top = "top"
    bottom = "bottom"
