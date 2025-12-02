import json
from typing import Optional

from .in_memory_store import InMemoryStore

def encode_tuples(obj):
    if isinstance(obj, tuple):
        return {"__tuple__": True, "items": [encode_tuples(x) for x in obj]}
    elif isinstance(obj, list):
        return [encode_tuples(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: encode_tuples(v) for k, v in obj.items()}
    else:
        return obj

def decode_tuples(obj):
    if isinstance(obj, dict) and "__tuple__" in obj:
        return tuple(decode_tuples(x) for x in obj["items"])
    elif isinstance(obj, list):
        return [decode_tuples(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: decode_tuples(v) for k, v in obj.items()}
    else:
        return obj



class JSONDataStore(InMemoryStore):
    def __init__(self, directory: str, filename: str, flush_every: Optional[int] = None, create_okay: bool = False):
        super().__init__(directory, filename, flush_every, create_okay)
        self.filepath = self.directory / f"{filename}.json"
        self.directory.mkdir(parents=True, exist_ok=True)

        if not self.create_okay and not self.filepath.exists():
            raise FileNotFoundError(f"Cache file {self.filepath} does not exist and create_okay is set to False.")

        if self.filepath.exists():
            self._load_from_file()

    def _load_from_file(self):
        with self.filepath.open(mode="r", encoding="utf-8") as f:
            self.data =decode_tuples(json.load(f))

    def _write_to_file(self):
        with self.filepath.open(mode="w", encoding="utf-8") as f:
            json.dump(encode_tuples(self.data), f, indent=4,)

    def _remove_file(self):
        if self.filepath.exists():
            self.filepath.unlink()

    def __str__(self):
        return f"JSONDataStore(filepath={self.filepath})"
