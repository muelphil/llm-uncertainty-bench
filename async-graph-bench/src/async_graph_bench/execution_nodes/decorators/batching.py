from asyncio import Lock
from collections import deque

from ...utils.end_of_data import EndOfData


def batching(generator, batch_size):
    """
    Wraps an asynchronous generator to process items in batches.

    Args:
        generator: The asynchronous generator function to wrap.
        batch_size: The size of each batch to process.

    Returns:
        A wrapped asynchronous generator function.
    """
    buffer = deque()
    lock = Lock()  # Ensures safe access to the buffer across multiple tasks
    end_of_data_seen = False

    async def wrapped(item):
        nonlocal buffer, end_of_data_seen

        # Process incoming item
        if not isinstance(item, EndOfData):
            assert end_of_data_seen is False, "Items were pushed to the batching layer after EndOfData signal was already seen"
            async with lock:
                buffer.append(item)

        current_batch = None
        async with (lock):
            # Process a full batch if buffer size meets or exceeds batch_size
            if len(buffer) >= batch_size:
                current_batch = [buffer.popleft() for _ in range(batch_size)]

            # If EndOfData is received, process remaining items in the buffer
            elif isinstance(item, EndOfData):
                end_of_data_seen = True
                if buffer:
                    current_batch = [buffer.popleft() for _ in range(len(buffer))]

        # Process the batch (outside the lock)
        if current_batch:
            async for batch_item in generator(current_batch):
                yield batch_item

        # Handle EndOfData
        if isinstance(item, EndOfData):
            async for eod in generator(item):
                yield eod

    return wrapped
