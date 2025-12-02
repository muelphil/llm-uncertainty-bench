import itertools
import json
from os import makedirs, remove
from os.path import join, exists
from typing import Iterable, Generator, Any, Tuple, List

from bitarray import bitarray

from ..stores.combined_id import get_combined_id_from_parts
from ..stores.store import DataStore


def get_resolved_keys(store: DataStore, node_config: "NodeConfig", iteration_count: int = 1):
    """
    Return the set of combined keys that are resolved in a given store.

    If sampling is enabled and the node uses "first" mode, the resolved keys
    are expanded to include all sampled iterations.

    Args:
        store: Data store providing (id, iteration) keys.
        node_config: Node configuration defining sampling behavior.
        iteration_count: Total number of iterations per item.

    Returns:
        A set of combined IDs (tuples) representing resolved items.
    """
    resolved_ids = set(get_combined_id_from_parts(id, iter) for id, iter in store.iter_keys())
    if node_config.is_sampling() and node_config.sampling_mode == "first":
        resolved_ids = expand_resolved_ids(resolved_ids, iteration_count, node_config.sampling_config.sampling_size)
    return resolved_ids


def resolved_ids_to_bitarray(id_to_idx_map: dict, resolved_ids: Iterable) -> bitarray:
    """
    Convert a set of resolved IDs into a bitarray mask.

    Args:
        id_to_idx_map: Mapping of combined IDs to index positions.
        resolved_ids: Iterable of resolved combined IDs.

    Returns:
        A bitarray where True indicates a resolved ID.
    """
    result = bitarray(len(id_to_idx_map))
    result.setall(False)

    for rid in resolved_ids:
        idx = id_to_idx_map.get(rid)
        if idx is not None:
            result[idx] = True

    return result


def get_duplicates(keys):
    """
    Identify duplicate elements in an iterable.

    Args:
        keys: Iterable of hashable items.

    Returns:
        A set containing all duplicate elements.
    """
    seen = set()
    duplicates = set()
    for key in keys:
        if key in seen:
            duplicates.add(key)
        else:
            seen.add(key)
    return duplicates


def build_combined_keys(
        ids: Iterable,
        iterations: int,
        iterations_first: bool
) -> List[Tuple[Any, ...]]:
    """
    Build combined keys by pairing item IDs with iteration indices.

    Args:
        ids: Iterable of item IDs (tuples or scalars).
        iterations: Number of iterations to generate per ID.
        iterations_first: If True, group all iterations per ID consecutively;
            otherwise, interleave iterations across all IDs.

    Returns:
        A list of combined (iteration, id...) tuples.
    """
    result = []
    if iterations_first:
        for id in ids:
            for iter in range(iterations):
                result.append((iter, *id) if isinstance(id, tuple) else (iter, id))
    else:
        ids_iterators = itertools.tee(ids, iterations)
        for i, it in enumerate(ids_iterators):
            for id in it:
                result.append((i, *id) if isinstance(id, tuple) else (i, id))

    return result


def is_fully_resolved(id_to_idx: dict, resolved_ids: Iterable) -> bool:
    """
    Check whether all expected IDs are present in the resolved set.

    Args:
        id_to_idx: Mapping of expected combined IDs to indices.
        resolved_ids: Iterable of resolved combined IDs.

    Returns:
        True if all IDs are resolved, False otherwise.
    """
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
    """
    Expand resolved IDs to include all iterations within a sampling batch.

    Used in "first" sampling mode to mark additional sampled iterations
    as resolved.

    Args:
        resolved_ids: Iterable of resolved combined IDs.
        iterations: Total number of iterations per item.
        sample_size: Number of iterations per sample batch.

    Returns:
        A set of expanded combined IDs.
    """
    resolved_set = set(resolved_ids)
    expanded_resolved_ids = set(resolved_set)

    for resolved_id in resolved_ids:
        resolved_iteration = resolved_id[0]
        resolved_item_id = resolved_id[1:]
        for i in range(1, sample_size):
            expanded_resolved_ids.add((resolved_iteration + i, *resolved_item_id))

    return expanded_resolved_ids


def adjust_string_length(s: str, length: int) -> str:
    """
    Truncate or pad a string to the specified length.

    Args:
        s: Input string.
        length: Desired string length.

    Returns:
        The adjusted string, padded with spaces if necessary.
    """
    return s[:length].ljust(length)


def flatten_recursive(iterable: Iterable) -> Generator[Any, None, None]:
    """
    Recursively flatten a nested iterable of lists or tuples.

    Args:
        iterable: Possibly nested iterable.

    Yields:
        Flattened elements from the nested structure.
    """
    for item in iterable:
        if isinstance(item, (list, tuple)):
            yield from flatten_recursive(item)
        else:
            yield item


def check_unique_strings(xs: Iterable[str]) -> None:
    """
    Ensure that all strings in an iterable are unique.

    Args:
        xs: Iterable of strings.

    Raises:
        Exception: If a duplicate string is found.
    """
    seen = set()
    for s in xs:
        if s in seen:
            raise Exception(f"Duplicate string found: {s!r}")
        seen.add(s)


def get_metadata(path, filename):
    """
    Load JSON metadata for a file if available.

    Args:
        path: Directory path containing the metadata file.
        filename: Base filename (without extension).

    Returns:
        Metadata dictionary, or an empty dict if no metadata file exists.
    """
    meta_file = join(path, f"{filename}.metadata.json")
    if not exists(meta_file):
        return {}
    with open(meta_file, "r", encoding="utf-8") as f:
        return json.load(f)


def update_metadata(path, filename, metadata):
    """
    Update or create a JSON metadata file for a given file.

    Args:
        path: Directory path where metadata is stored.
        filename: Base filename (without extension).
        metadata: Dictionary of metadata values to add or update.

    Raises:
        TypeError: If metadata is not a dictionary.
    """
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dict")
    prev_metadata = get_metadata(path, filename)
    metadata = prev_metadata | metadata
    meta_file = join(path, f"{filename}.metadata.json")
    makedirs(path, exist_ok=True)  # ensure directory exists
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def clear_metadata(path, filename):
    """
    Delete the metadata file associated with a given file, if it exists.

    Args:
        path: Directory path containing the metadata file.
        filename: Base filename (without extension).
    """
    meta_file = join(path, f"{filename}.metadata.json")
    if exists(meta_file):
        remove(meta_file)
