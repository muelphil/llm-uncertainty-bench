import itertools
import json
from typing import Iterable, Generator, Any, Tuple

import numpy as np
from bitarray import bitarray
from os.path import join, exists
from os import makedirs

from os import makedirs, remove


def resolved_ids_to_bitarray(id_to_idx_map: dict, resolved_ids: Iterable) -> bitarray:
    result = bitarray(len(id_to_idx_map))
    result.setall(False)

    for rid in resolved_ids:
        idx = id_to_idx_map.get(rid)
        if idx is not None:
            result[idx] = True

    return result


def build_combined_ids(
        ids: Iterable,
        iterations: int,
        iterations_first: bool
) -> Generator[Tuple[Any, ...], None, None]:
    if iterations_first:
        for id in ids:
            for iter in range(iterations):
                yield (iter, *id) if isinstance(id, tuple) else (iter, id)
    else:
        ids_iterators = itertools.tee(ids, iterations)
        for i, it in enumerate(ids_iterators):
            for id in it:
                yield (i, *id) if isinstance(id, tuple) else (i, id)


def is_fully_resolved(id_to_idx: dict, resolved_ids: Iterable) -> bool:
    count = 0
    for rid in resolved_ids:
        if rid in id_to_idx:
            count += 1
    return count == len(id_to_idx)


def expand_resolved_ids(
        resolved_ids: Iterable[Tuple[int, ...]],
        iterations: int,
        sample_size: int
) -> set:
    resolved_set = set(resolved_ids)
    expanded_resolved_ids = set(resolved_set)

    for resolved_id in resolved_ids:
        resolved_iteration = resolved_id[0]
        resolved_item_id = resolved_id[1:]
        for i in range(1, sample_size):
            expanded_resolved_ids.add((resolved_iteration + i, *resolved_item_id))

    return expanded_resolved_ids


def adjust_string_length(s: str, length: int) -> str:
    return s[:length].ljust(length)


def find_largest_consecutive(set_of_integers: set[int]) -> int:
    i = 0
    while i in set_of_integers:
        i += 1
    return i - 1


def flatten_recursive(iterable: Iterable) -> Generator[Any, None, None]:
    for item in iterable:
        if isinstance(item, (list, tuple)):
            yield from flatten_recursive(item)
        else:
            yield item


def check_unique_strings(xs: Iterable[str]) -> None:
    seen = set()
    for s in xs:
        if s in seen:
            raise Exception(f"Duplicate string found: {s!r}")
        seen.add(s)


def _delete_nans(ue: np.ndarray, metric: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    new_ue, new_metric = [], []
    for i in range(len(metric)):
        if not np.isnan(metric[i]) and not np.isnan(ue[i]):
            new_ue.append(ue[i].real if isinstance(ue[i], complex) else ue[i])
            new_metric.append(metric[i])

    return np.array(new_ue), np.array(new_metric)


def get_metadata(path, filename):
    meta_file = join(path, f"{filename}.metadata.json")
    if not exists(meta_file):
        return {}
    with open(meta_file, "r", encoding="utf-8") as f:
        return json.load(f)


def update_metadata(path, filename, metadata):
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dict")
    prev_metadata = get_metadata(path, filename)
    metadata = prev_metadata | metadata
    meta_file = join(path, f"{filename}.metadata.json")
    makedirs(path, exist_ok=True)  # ensure directory exists
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def clear_metadata(path, filename):
    """Delete the metadata file if it exists."""
    meta_file = join(path, f"{filename}.metadata.json")
    if exists(meta_file):
        remove(meta_file)
