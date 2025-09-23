from typing import Any, Protocol


class Serializer(Protocol):
    def serialize(self, item: Any) -> bytes:
        pass

    def deserialize(self, data: bytes) -> Any:
        pass
