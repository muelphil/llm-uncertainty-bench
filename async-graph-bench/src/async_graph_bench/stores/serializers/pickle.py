import pickle
from typing import Any


class PickleSerializer:
    def serialize(self, item: Any) -> bytes:
        return pickle.dumps(item)

    def deserialize(self, data: bytes) -> Any:
        return pickle.loads(data)
