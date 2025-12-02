import inspect
import logging
from typing import AsyncIterator, Callable, Dict, Any, Iterable
from collections.abc import AsyncIterable
from ..utils.end_of_data import EndOfData  # Adjust import if needed

log = logging.getLogger(__name__)


async def _ensure_async(source):
    if isinstance(source, AsyncIterable):
        return source
    elif isinstance(source, Iterable):
        async def async_wrap():
            for item in source:
                yield item

        return async_wrap()
    else:
        raise TypeError("data_source must return an async iterator or an iterable")


class DataSourceExecutionWrapper:
    """
    Wraps a data source and emits its items with optional iteration logic.

    Adds a unique '_idx' field to each item, and optionally an 'iter' field
    to distinguish repeated iterations.

    Args:
        data_source: A callable returning an async iterable of data items.
        iterations: Number of times to repeat each data item.
        iterations_first: If True, iteration varies faster than items.
    """

    def __init__(
            self,
            data_source: Callable[[], AsyncIterator[Dict[str, Any]]],
            iterations: int = 1,
            iterations_first: bool = True
    ):
        self.data_source_gen = data_source
        self.iterations = iterations
        self.iterations_first = iterations_first

    async def execute(self, *args, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """
        Asynchronously yields items from the data source, attaching '_idx'
        and optional 'iter' fields, and ends with an EndOfData marker.

        Yields:
            Augmented data items with tracking metadata.
        """
        counter = 0  # Provides unique integer ID for each emitted item

        if self.iterations == 1:
            source_iter = await _ensure_async(self.data_source_gen())
            async for item in source_iter:
                item['_idx'] = counter
                counter += 1
                yield item

        elif self.iterations_first:
            source_iter = await _ensure_async(self.data_source_gen())
            async for item in source_iter:
                for i in range(self.iterations):
                    item_i = item.copy()
                    item_i['_idx'] = counter
                    item_i['iter'] = i
                    counter += 1
                    yield item_i

        else:
            for i in range(self.iterations):
                # Note: THIS MAY NOT BE MOVED OUTSIDE THE LOOP !!!!
                # iterators may only be used once -> for additional runs, the iterator needs to be created again!
                source_iter = await _ensure_async(self.data_source_gen())
                async for item in source_iter:
                    item_i = item.copy()
                    item_i['_idx'] = counter
                    item_i['iter'] = i
                    counter += 1
                    yield item_i

        yield EndOfData()
