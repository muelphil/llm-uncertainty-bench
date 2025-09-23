import json
import os
from collections import defaultdict
from typing import Any, Optional, Iterator, Tuple

import pandas as pd

from .store import DataStore
from .combined_id import get_combined_id


class JSONDataStore(DataStore):
    def __init__(self, directory: str, filename: str, flush_every: Optional[int] = None):
        self.directory = directory
        self.filepath = os.path.join(directory, f"{filename}.json")
        self.flush_every = flush_every
        self.data = []
        self.modified_count = 0
        self._initialize()

    def _initialize(self):
        os.makedirs(self.directory, exist_ok=True)
        if os.path.exists(self.filepath):
            with open(self.filepath, mode='r', encoding='utf-8') as f:
                self.data = json.load(f)

    def save(self, item: dict):
        for i, saved_item in enumerate(self.data):
            if saved_item["id"] == item["id"] and saved_item.get("iter", 0) == item.get("iter", 0):
                self.data[i] = item  # Replace existing item
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

    def contains_id(self, id: Any, iteration=0) -> bool:
        return any(entry["id"] == id and entry.get("iter", 0) == iteration for entry in self.data)

    def load(self, id: Any, iteration=0) -> dict:
        for entry in self.data:
            if entry["id"] == id and entry.get("iter", 0) == iteration:
                return entry
        return None

    def to_dataframe(self, properties=None):
        df = pd.DataFrame(self.data)
        return df[properties] if properties else df

    def flush(self):
        with open(self.filepath, mode='w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=4)
        self.modified_count = 0

    def get_length_per_iteration(self):
        lengths_per_iteration = defaultdict(int)
        for item in self.data:
            lengths_per_iteration[item.get("iter", 0)] += 1
        return lengths_per_iteration

    def iter_indices(self) -> Iterator[Tuple[int, int]]:
        for entry in self.data:
            yield get_combined_id(entry)

    def clear(self):
        if os.path.exists(self.filepath):
            os.remove(self.filepath)
        self.data = []
        self.modified_count = 0

    def __len__(self):
        return len(self.data)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.flush()

    def __str__(self):
        return f"JSONDataStore(filepath={self.filepath})"
