import struct


def _read_png_dimensions(content: bytes) -> tuple[int, int] | None:
    if len(content) < 24:
        return None
    if content[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width, height = struct.unpack(">II", content[16:24])
    if width <= 0 or height <= 0:
        return None
    return width, height


def _read_gif_dimensions(content: bytes) -> tuple[int, int] | None:
    if len(content) < 10:
        return None
    if not (content.startswith(b"GIF87a") or content.startswith(b"GIF89a")):
        return None
    width, height = struct.unpack("<HH", content[6:10])
    if width <= 0 or height <= 0:
        return None
    return width, height


def _read_jpeg_dimensions(content: bytes) -> tuple[int, int] | None:
    if len(content) < 4 or content[0:2] != b"\xff\xd8":
        return None

    index = 2
    sof_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }

    while index + 1 < len(content):
        if content[index] != 0xFF:
            index += 1
            continue
        while index < len(content) and content[index] == 0xFF:
            index += 1
        if index >= len(content):
            break

        marker = content[index]
        index += 1

        if marker in {0xD8, 0xD9}:
            continue
        if marker == 0xDA:
            break
        if index + 1 >= len(content):
            break

        segment_length = struct.unpack(">H", content[index : index + 2])[0]
        if segment_length < 2 or index + segment_length > len(content):
            break

        if marker in sof_markers and segment_length >= 7:
            frame = content[index + 2 : index + 2 + 5]
            if len(frame) < 5:
                break
            height, width = struct.unpack(">HH", frame[1:5])
            if width > 0 and height > 0:
                return width, height

        index += segment_length

    return None


def _read_webp_dimensions(content: bytes) -> tuple[int, int] | None:
    if len(content) < 30:
        return None
    if content[0:4] != b"RIFF" or content[8:12] != b"WEBP":
        return None

    chunk_type = content[12:16]

    if chunk_type == b"VP8X" and len(content) >= 30:
        width_minus_one = content[24] | (content[25] << 8) | (content[26] << 16)
        height_minus_one = content[27] | (content[28] << 8) | (content[29] << 16)
        width = width_minus_one + 1
        height = height_minus_one + 1
        if width > 0 and height > 0:
            return width, height

    if chunk_type == b"VP8 " and len(content) >= 30:
        width, height = struct.unpack("<HH", content[26:30])
        width &= 0x3FFF
        height &= 0x3FFF
        if width > 0 and height > 0:
            return width, height

    if chunk_type == b"VP8L" and len(content) >= 25:
        bits = int.from_bytes(content[21:25], "little")
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
        if width > 0 and height > 0:
            return width, height

    return None


def extract_image_dimensions(content: bytes) -> tuple[int | None, int | None]:
    for parser in (_read_png_dimensions, _read_gif_dimensions, _read_jpeg_dimensions, _read_webp_dimensions):
        parsed = parser(content)
        if parsed is not None:
            return parsed
    return None, None
