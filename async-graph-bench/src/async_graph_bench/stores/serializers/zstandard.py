try:
    import zstd
except ImportError as e:
    raise ImportError(
        "To use this functionality, you need to install the 'zstd' via 'pip install zstd' "
    ) from e

# zstandard base implementation (c) https://github.com/facebook/zstd
# language ports https://facebook.github.io/zstd/#other-languages
# chosen: https://github.com/sergey-dryabzhinsky/python-zstd

from typing import Any


class ZstdCompressionSerializer:

    def __init__(self, level=3):
        self.level = level

    def serialize(self, item: Any) -> bytes:
        return zstd.compress(item, self.level)

    def deserialize(self, data: bytes) -> Any:
        return zstd.decompress(data)
