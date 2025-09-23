import ast
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Iterator, Tuple

import pandas as pd

from .combined_id import get_combined_id
from .store import DataStore


class CSVDataStore(DataStore):
    def __init__(self, directory: str, filename: str, flush_every: Optional[int] = 1000, create_okay: bool = False):
        self.directory = Path(directory)
        self.filepath = self.directory / f"{filename}.csv"
        self.flush_every = flush_every
        self.separator = ";"
        self.data = []
        self.modified_count = 0

        if not create_okay and not self.filepath.exists():
            raise FileNotFoundError(f"Cache file {self.filepath} does not exist and create_okay is set to False.")

        self.directory.mkdir(parents=True, exist_ok=True)
        if self.filepath.exists():
            with self.filepath.open(mode="r", newline="") as f:
                reader = csv.DictReader(f, delimiter=self.separator)
                self.data = [
                    {key: None if value == "" else ast.literal_eval(value) for key, value in row.items()}
                    for row in reader
                ]

    def save(self, item: dict):
        # if not all(key in item for key in self.properties):
        #     raise ValueError(f"Missing properties in item: {set(self.properties) - set(item.keys())}")
        # Find index of existing item with the same id and iteration
        for i, saved_item in enumerate(self.data):
            if saved_item["id"] == item["id"] and saved_item.get("iter", 0) == item.get("iter", 0):
                self.data[i] = item  # Replace existing item
                return

        # If no match was found, append the new item
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
        # raise KeyError(f"Item with ID {id} in iteration {iteration} not found in the store.")

    def to_dataframe(self, properties=None):
        df = pd.DataFrame(self.data)
        if properties:
            return df[properties]
        else:
            return df


    def flush(self):
        if self.data:
            with self.filepath.open(mode='w', newline='') as f:
                columns = set().union(*self.data) if self.data else set()
                writer = csv.DictWriter(f, columns, delimiter=self.separator)
                writer.writeheader()
                writer.writerows(self.data)
            self.modified_count = 0

    def get_length_per_iteration(self):
        lengths_per_iteration = defaultdict(int)
        for item in self.data:
            lengths_per_iteration[item.get("iter", 0)] += 1
        return lengths_per_iteration

    def iter_indices(self) -> Iterator[Tuple[int, int]]:
        for entry in self.data[:]:  # shallow copy, necessary to use with delete
            yield get_combined_id(entry)

    def clear(self):
        if self.filepath.exists:
            self.filepath.unlink()  # remove
        self.data = []
        self.modified_count = 0

    def __len__(self):
        return len(self.data)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.flush()

    def __str__(self):
        return f"CSVDataStore(filepath={self.filepath})"
