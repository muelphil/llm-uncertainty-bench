import inspect
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Iterator, AsyncIterator, Tuple


class DataSource(ABC):
    """
    Abstract base class for data sources providing streaming access to datasets.
    """

    @property
    @abstractmethod
    def stats(self) -> List[str]:
        """
        List of statistics or features provided by iter_items alongside an id.

        Returns:
            List of stat names.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the length (number of items) of the dataset.

        Returns:
            Integer length of dataset.
        """
        pass

    @abstractmethod
    def iter_items(self) -> Union[Iterator[Dict[str, List[str]]], AsyncIterator[Dict[str, List[str]]]]:
        """
        Asynchronously iterate over items in the dataset.

        Each yielded item is a dictionary mapping feature/statistic names
        to lists of string values (e.g., input_texts, target_texts).

        Returns:
            Sync or async iterator yielding dictionaries representing the data that is to be processed.
        """
        pass

    @abstractmethod
    def iter_keys(self) -> Iterator[Union[int, str, Tuple[Union[str, int], ...]]]:
        pass


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
    def stats(self) -> List[str]:
        return self._source.stats

    def __len__(self) -> int:
        return self._end - self._start

    def iter_keys(self) -> Iterator[Union[int, str, Tuple[Union[str, int], ...]]]:
        for i, key in enumerate(self._source.iter_keys()):
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
