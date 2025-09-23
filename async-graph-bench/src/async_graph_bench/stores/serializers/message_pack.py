try:
    import msgpack
except ImportError as e:
    raise ImportError(
        "To use this functionality, you need to install the 'msgpack'. "
    ) from e
from typing import Any


class MessagePackSerializer:
    def serialize(self, item: Any) -> bytes:
        return msgpack.packb(item)

    def deserialize(self, data: bytes) -> Any:
        return msgpack.unpackb(data, raw=False)
