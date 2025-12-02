import logging
import os
from typing import List, Any

import numpy as np
import pandas as pd
from diskcache import Cache
from tqdm import tqdm

from .combined_id import get_combined_id_from_parts
from .serializers import ZLibCompressionSerializer, PickleSerializer, Serializer
from .store import DataStore

log = logging.getLogger(__name__)


# temporary fix for length mismatch in alternatives
def truncate_innermost_arrays(arr: np.ndarray) -> np.ndarray:
    """
    Ensures that all innermost arrays in a nested np.ndarray have the same length
    by truncating them to the minimum length found.
    """
    # Flatten to find all innermost arrays
    innermost_arrays = [inner for outer in arr for inner in outer]

    # Find the minimum length among innermost arrays
    min_length = min(len(inner) for inner in innermost_arrays)

    # Truncate all innermost arrays to this minimum length
    truncated_arr = np.array([[inner[:min_length] for inner in outer] for outer in arr], dtype=object)

    return truncated_arr


_sentinel = object()


class DiskCacheStore(DataStore):

    def __init__(self, directory, filename, serializers: List[Serializer] = None, create_okay=False, *args, **kwargs):
        """
        Initialize the store.

        :param directory: Path to the directory where the cache will be stored.
        :param filename: Name of the cache file.
        :param properties: (Optional) List of property names to be stored in the cache, required if items are to be stored.
        """
        self.cache_path = os.path.join(directory, filename)
        if not os.path.exists(self.cache_path):
            if not create_okay:
                raise FileNotFoundError(f"Cache file {self.cache_path} does not exist and create_okay is set to False.")
            log.info(f"Diskcache {self.cache_path} not found, creating it.")
        self.cache = Cache(
            self.cache_path,
            # Note: The actual size on disk may exceed this size_limit due to 'cull_limit' set to 0 -- 20 GB is only the max file size that can be written
            # This does not limit the cache size, only the size of individual files written to disk.
            size_limit=20 * 1024 ** 3,  # 20GB
            disk_min_file_size=2 ** 18,  # 256 kb
            eviction_policy='none'  # should not remove items
        )
        self.cache.reset('cull_limit', 0)  # Disable automatic evictions.
        self.serializers = serializers or [PickleSerializer(), ZLibCompressionSerializer()]
        # self.properties = properties

    def save(self, item):
        """
        Save an item to the store.

        :param item_index: The unique identifier for the item.
        :param item: A dictionary with properties matching the initialized properties.
        """
        # if cache.volume() > self.max_size:
        #     raise RuntimeError(f"Cache volume {cache.volume():,} exceeds limit {self.max_size:,}")

        serialized = item
        for serializer in self.serializers:
            serialized = serializer.serialize(serialized)

        id = item["id"]
        iter = item.get("iter", 0)
        item_index = (iter, *id) if isinstance(id, tuple) else (iter, id)

        # Store the serialized item in the cache
        self.cache[item_index] = serialized

    def delete(self, id: Any, iter=0, default=_sentinel):
        combined_id = get_combined_id_from_parts(id, iter)
        return self.cache.delete(combined_id)

    def iter_keys(self):
        """
        Lazily retrieve all item IDs currently in the store.

        :yield: Each key (ID) one by one.
        """
        for combined_key in self.cache.iterkeys():
            yield (combined_key[1:], combined_key[0]) if len(combined_key) > 2 else (combined_key[1], combined_key[0])
        # yield from self.cache.iterkeys()

    def iter_items(self):
        for key in self.cache.iterkeys():
            serialized = self.cache[key]
            for serializer in reversed(self.serializers):
                serialized = serializer.deserialize(serialized)
            yield serialized

    def contains_key(self, id, iter=0):
        """
        Check if the cache contains an item with the given ID.

        :param item_index: The unique identifier for the item.
        :return: True if the item exists, False otherwise.
        """
        combined_id = get_combined_id_from_parts(id, iter)
        return combined_id in self.cache

    def load(self, id, iter=0):
        """
        Load the data for the specified ID.

        :param item_index: The unique identifier for the item.
        :return: The decompressed item as a dictionary.
        :raises KeyError: If the item does not exist in the cache.
        """
        combined_id = get_combined_id_from_parts(id, iter)
        if combined_id not in self.cache:
            return None

        serialized = self.cache[combined_id]
        # Deserialize the item in reverse order
        for serializer in reversed(self.serializers):
            serialized = serializer.deserialize(serialized)

        return serialized

    def to_dataframe(self, properties=None, show_progress=False):
        """
        Load all cached data into a pandas DataFrame.

        :return: A pandas DataFrame containing all cached items, with IDs and properties as columns.
        """
        item_data = []

        keys = self.cache.iterkeys()
        if show_progress:
            keys = tqdm(keys, total=len(self.cache), desc="Loading cache")

        for item_index in keys:
            serialized = self.cache[item_index]
            # Deserialize the item in reverse order
            for serializer in reversed(self.serializers):
                serialized = serializer.deserialize(serialized)
            if not "iter" in serialized:
                serialized["iter"] = 0

            if properties is not None:
                serialized = {key: serialized[key] for key in properties}

            # Append the ID and corresponding properties to the data
            # item_indices.append(item_index)
            item_data.append(serialized)

        # Create a DataFrame
        df = pd.DataFrame(item_data)

        if "iter" in df.columns:
            df["iter"] = df["iter"].fillna(0)
        # if "index" not in df.columns:
        #     df.insert(0, "index", item_indices)  # Add the IDs as the first column
        return df

    def flush(self):
        """
        Close the cache explicitly.
        """
        pass
        # Closed Cache objects will automatically re-open when accessed. But opening Cache objects is relatively slow,
        # and since all operations are atomic, may be safely left open.
        # Source: https://grantjenks.com/docs/diskcache/tutorial.html
        # self.cache.close()

    def __str__(self):
        return f"DiskCacheStoreStatCalculator(cache={self.cache_path})"

    def clear(self):
        self.cache.clear()

    def __len__(self):
        return len(self.cache)
