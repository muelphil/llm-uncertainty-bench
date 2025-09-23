from collections import defaultdict
from asyncio import Lock
from ...utils.end_of_data import EndOfData
from ...stores import get_combined_id


def multi_incoming_node(generator, incoming_nodes_count):
    """
    Waits for a specific number of items with the same `index` from multiple nodes before calling the generator.

    Args:
        generator: The async function to wrap.
        incoming_nodes_count: The number of items required for each `index`.

    Returns:
        A wrapped function.
    """
    count_per_index = defaultdict(int)
    lock = Lock()  # Ensures safe access to the buffer across multiple tasks

    async def wrapped(item):
        combined_id = get_combined_id(item) if not isinstance(item, EndOfData) else EndOfData

        if (count_per_index[combined_id] + 1) == incoming_nodes_count:
            del count_per_index[combined_id]
            async for result in generator(item):
                yield result
        else:
            async with lock:
                count_per_index[combined_id] += 1

    return wrapped
