import ast
import csv
from typing import Optional

from .in_memory_store import InMemoryStore


def safe_literal_eval(value: str):
    try:
        # Try to parse as a Python literal (int, float, list, dict, etc.)
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # Fallback: return the original string
        return value


class CSVDataStore(InMemoryStore):
    def __init__(self, directory: str, filename: str, flush_every: Optional[int] = 1000, create_okay: bool = False):
        super().__init__(directory, filename, flush_every, create_okay)
        self.filepath = self.directory / f"{filename}.csv"
        self.separator = ";"

        if not self.create_okay and not self.filepath.exists():
            raise FileNotFoundError(f"Cache file {self.filepath} does not exist and create_okay is set to False.")

        self.directory.mkdir(parents=True, exist_ok=True)
        if self.filepath.exists():
            self._load_from_file()

    def _load_from_file(self):
        with self.filepath.open(mode="r", newline="") as f:
            reader = csv.DictReader(f, delimiter=self.separator)
            self.data = [
                {key: None if value == "" else safe_literal_eval(value) for key, value in row.items()}
                for row in reader
            ]

    def _write_to_file(self):
        with self.filepath.open(mode="w", newline="") as f:
            if not self.data:
                f.write("")  # Nothing to write
                return
            all_columns = {key for item in self.data for key in item}
            preferred = ["id", "iter"]
            # Keep preferred ones first (if they exist), then the rest sorted alphabetically
            columns = [col for col in preferred if col in all_columns] + sorted(all_columns - set(preferred))

            writer = csv.DictWriter(f, columns, delimiter=self.separator)
            writer.writeheader()
            writer.writerows(self.data)

    def _remove_file(self):
        if self.filepath.exists():
            self.filepath.unlink()

    def __str__(self):
        return f"CSVDataStore(filepath={self.filepath})"
