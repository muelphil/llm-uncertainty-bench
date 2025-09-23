from typing import Any
import zlib


class ZLibCompressionSerializer:

    def __init__(self, level=-1):
        self.level = level

    def serialize(self, item: Any) -> bytes:
        return zlib.compress(item, level=self.level)

    def deserialize(self, data: bytes) -> Any:
        return zlib.decompress(data)
