from typing import Optional, Union


def sizeof_fmt(num: int, suffix: Optional[str] = "B") -> str:
    for unit in ["", "K", "M", "G"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Yi", suffix)


def to_dict(content: Union[str, bytes]) -> dict:
    out_content = {}
    if isinstance(content, bytes):
        decoded = content.decode()
        lines = decoded.splitlines()
        for line in lines:
            try:
                key, value = line.split(":")
                out_content[key] = value.strip()
            except ValueError:
                continue
    elif isinstance(content, str):
        pass

    return out_content
