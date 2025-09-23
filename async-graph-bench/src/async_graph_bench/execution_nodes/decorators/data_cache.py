from asyncio import Lock
from typing import Union, List

from ...utils.end_of_data import EndOfData
from ...stores import DataStore


def data_cache(generator, store: DataStore, properties: Union[List[str], "all"]):
    """
    Adds caching functionality to avoid recomputation for previously processed items.

    Args:
        generator: The async function to wrap.
        store: An object implementing a data store interface (`has_id`, `load_data_for_id`, `save`, and `close` methods).

    Returns:
        A wrapped function.
    """
    if properties != "all":
        properties = ["id", "iter"] + list((set(properties) - {"id", "iter"}))

    async def wrapped(item):
        if not isinstance(item, EndOfData):
            stored_data = store.load(item["id"], item.get("iter", 0))
            if stored_data is not None:
                item.update(stored_data)
                yield item
                return

        async for result in generator(item):
            if not isinstance(result, EndOfData):
                keys = result.keys() if properties == "all" else properties
                serialized = {key: value for key, value in result.items() if key in keys}
                store.save(serialized)
            else:
                store.flush()
            yield result

    return wrapped
