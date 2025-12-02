import inspect
from abc import ABC, abstractmethod

from typing import List, Dict, Union, Iterator, AsyncIterator, Tuple, Any

Id = Union[int, str, Tuple[Union[int, str], ...]]


class DataSource(ABC):
    """Abstract base class for benchmark input data sources.

    A `DataSource` provides streaming access to the input dataset used in a benchmark.
    Each benchmark must have exactly one `DataSource`, which defines the items
    that will flow through the computation graph.

    Unlike regular nodes, a `DataSource` does not consume dependencies from other nodes.
    Instead, it produces initial values that other nodes can depend on. Each item
    produced must include a unique `"id"` and values for all keys declared in the
    `provides` property.
    """

    @property
    def id(self) -> str:
        """Return a unique identifier for the DataSource.

        By default, this is the class name. Used to identify the
        DataSource as a node in the benchmark graph.

        Returns:
            str: Unique DataSource identifier.
        """
        return self.__class__.__name__

    @property
    @abstractmethod
    def provides(self) -> List[str]:
        """List of features or statistics produced by this DataSource.

        Every item yielded by `iter_items` must include values for
        all keys in this list alongside a unique `"id"`.

        Returns:
            List[str]: Names of statistics or features provided.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of items in the dataset.

        This allows the benchmark framework to query dataset size
        without iterating through all items.

        Returns:
            int: Number of items in the dataset.
        """
        pass

    @abstractmethod
    def iter_items(self) -> Union[Iterator[Dict[str, Any]], AsyncIterator[Dict[str, Any]]]:
        """Iterate over items in the dataset, yielding one at a time.

        Each yielded item is a dictionary containing:
          * `"id"`: Unique identifier for the item (int, str, or tuple).
          * One value per key listed in `provides`.

        Returns:
            Iterator[Dict[str, Any]] or AsyncIterator[Dict[str, Any]]:
                A synchronous or asynchronous iterator over dataset items.
        """
        pass

    @abstractmethod
    def iter_ids(self) -> Iterator[Any]:
        """Iterate over all unique item IDs without loading full items.

        Useful for building indices and caches without consuming memory
        by iterating the entire dataset.

        Returns:
            Iterator[Any]: Iterator over item identifiers.
        """
        pass


# TODO implement to_dataframe on DataSource

class DataSourcePartitioner(DataSource):
    """
    A wrapper around a DataSource that partitions it into equal-sized splits.
    This instance represents one specific split.
    """

    def __init__(self, source: DataSource, num_splits: int, split_index: int):
        assert num_splits > 0, "num_splits must be positive"
        assert 0 <= split_index < num_splits, "split_index out of range"

        self._source = source
        self._num_splits = num_splits
        self._split_index = split_index

        self._length = len(source)
        # Compute partition boundaries
        base_size = self._length // num_splits
        remainder = self._length % num_splits

        # Distribute the remainder across the first `remainder` splits
        start = split_index * base_size + min(split_index, remainder)
        end = start + base_size + (1 if split_index < remainder else 0)

        self._start = start
        self._end = end

    @property
    def provides(self) -> List[str]:
        return self._source.provides

    def __len__(self) -> int:
        return self._end - self._start

    def iter_ids(self) -> Iterator[Id]:  # TODO rename iter_ids
        for i, key in enumerate(self._source.iter_ids()):
            if self._start <= i < self._end:
                yield key

    def iter_items(
            self,
    ) -> Union[Iterator[Dict[str, List[str]]], AsyncIterator[Dict[str, List[str]]]]:
        src_iter = self._source.iter_items()

        if inspect.isasyncgen(src_iter):

            async def async_gen():
                for i, item in enumerate(src_iter):
                    if self._start <= i < self._end:
                        yield item

            return async_gen()
        else:

            def sync_gen():
                for i, item in enumerate(src_iter):
                    if self._start <= i < self._end:
                        yield item

            return sync_gen()
