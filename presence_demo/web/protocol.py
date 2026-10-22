"""In-memory video framing. 0x01/02/03 remain reserved for audio paths."""

from __future__ import annotations

import base64
import binascii
import math
import struct

VIDEO_TAG = 4
VIDEO_HEADER = struct.Struct("<BdB")  # tag, client performance.now() milliseconds, speech


def _jpeg(data: bytes) -> bytes:
    if len(data) < 4 or not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
        raise ValueError("invalid JPEG envelope")
    return data


def _timestamp(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("invalid video timestamp")
    return value


def binary_frame(packet: bytes) -> dict:
    """Validate before admitting video into the latest-frame slot."""
    if len(packet) < VIDEO_HEADER.size + 4:
        raise ValueError("short video packet")
    tag, timestamp, speech = VIDEO_HEADER.unpack_from(packet)
    if tag != VIDEO_TAG or speech not in (0, 1):
        raise ValueError("invalid video header")
    return {"data": _jpeg(packet[VIDEO_HEADER.size :]), "t": _timestamp(timestamp), "speech": bool(speech)}


def legacy_frame(message: dict) -> dict:
    """Keep legacy JSON clients compatible, with the same worker representation."""
    data = message.get("data")
    if not isinstance(data, str):
        raise ValueError("missing JPEG")
    try:
        jpeg = base64.b64decode(data.split(",")[-1], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("invalid base64 JPEG") from exc
    timestamp = message.get("t")
    if timestamp is not None:
        timestamp = _timestamp(timestamp)
    return {"data": _jpeg(jpeg), "t": timestamp, "speech": bool(message.get("speech"))}
