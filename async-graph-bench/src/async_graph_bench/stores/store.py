from abc import ABC, abstractmethod
from typing import Iterator, Tuple, Any, Dict

from ..data_source import Id


class DataStore(ABC):
    @abstractmethod
    def __init__(self, directory: str, filename: str, create_okay: bool = False):
        pass

    @abstractmethod
    def save(self, item):
        """
        Save an item to the store.

        :param item: A dictionary with properties to be stored. The dictionary must have the key id, which has to be hashable.
        """
        pass

    @abstractmethod
    def delete(self, id: Any, iteration=0) -> bool:
        """
        Save an item to the store.

        :param item: A dictionary with properties to be stored. The dictionary must have the key id, which has to be hashable.
        """
        pass

    @abstractmethod
    def iter_keys(self) -> Iterator[Tuple[Id, int]]:
        """
        Iterate over all item IDs & Iterations currently in the store.
        """
        pass

    @abstractmethod
    def iter_items(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate over all items currently in the store.
        """
        pass

    @abstractmethod
    def contains_key(self, id: Any, iteration=0):
        """
        Check if the store contains an item with the given ID.

        :param item_id: The unique identifier for the item.
        :return: True if the item exists, False otherwise.
        """
        pass

    @abstractmethod
    def load(self, id: Any, iteration=0) -> dict:
        """
        Load the data for the specified ID.

        :param item_id: The unique identifier for the item.
        :return: The item as a dictionary.
        """
        pass

    @abstractmethod
    def to_dataframe(self, properties=None):
        """
        Load all data in the store into a pandas DataFrame.

        :return: A pandas DataFrame containing all items.
        """
        pass

    @abstractmethod
    def flush(self):
        """
        Flush the store data to file.
        """
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    def __enter__(self):
        """
        Enter the runtime context related to this object.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the runtime context and ensure resources are cleaned up.
        """
        self.flush()
