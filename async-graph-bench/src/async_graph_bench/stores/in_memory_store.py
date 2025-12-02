from abc import abstractmethod
from pathlib import Path
from typing import Optional, Any, Iterator, Tuple

import pandas as pd

from .store import DataStore


class InMemoryStore(DataStore):
    def __init__(self, directory: str, filename: str, flush_every: Optional[int] = None, create_okay: bool = False):
        self.directory = Path(directory)
        self.filename = filename
        self.flush_every = flush_every
        self.filepath = None  # Set by subclass
        self.data = []
        self.modified_count = 0
        self.create_okay = create_okay

    def save(self, item: dict):
        for i, saved_item in enumerate(self.data):
            if saved_item["id"] == item["id"] and saved_item.get("iter", 0) == item.get("iter", 0):
                self.data[i] = item
                return
        self.data.append(item)
        self.modified_count += 1
        if self.flush_every and self.modified_count >= self.flush_every:
            self.flush()

    def delete(self, id: Any, iteration=0) -> bool:
        for i, saved_item in enumerate(self.data):
            if saved_item["id"] == id and saved_item.get("iter", 0) == iteration:
                del self.data[i]
                self.modified_count += 1
                if self.flush_every and self.modified_count >= self.flush_every:
                    self.flush()
                return True
        return False

    def contains_key(self, id: Any, iteration=0) -> bool:
        return any(entry["id"] == id and entry.get("iter", 0) == iteration for entry in self.data)

    def load(self, id: Any, iteration=0) -> Optional[dict]:
        for entry in self.data:
            if entry["id"] == id and entry.get("iter", 0) == iteration:
                return entry
        return None

    def to_dataframe(self, properties=None):
        df = pd.DataFrame(self.data)
        return df[properties] if properties else df

    def iter_keys(self) -> Iterator[Tuple[int, int]]:
        for entry in self.data[:]:
            yield (entry["id"], entry.get("iter", 0))

    def iter_items(self) -> Iterator[Tuple[int, int]]:
        for entry in self.data[:]:
            yield entry

    def __len__(self):
        return len(self.data)

    # -------------------- Abstract I/O methods ---------------------

    @abstractmethod
    def _load_from_file(self):
        pass

    @abstractmethod
    def _write_to_file(self):
        pass

    @abstractmethod
    def _remove_file(self):
        pass

    # --------------------- Public File Methods ---------------------

    def flush(self):
        if self.data:
            self._write_to_file()
            self.modified_count = 0

    def clear(self):
        self._remove_file()
        self.data = []
        self.modified_count = 0
