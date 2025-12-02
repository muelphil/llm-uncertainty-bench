from bitarray import bitarray

from ...utils.end_of_data import EndOfData


def skip_indices(generator, indices_to_skip: bitarray):
    """
    Skips processing items with indices in the provided `indices_to_skip` set.

    Args:
        indices_to_skip: A set of indices that should be skipped.

    Returns:
        A wrapped function that checks if the item should be skipped.
    """

    async def wrapped(item):
        if not isinstance(item, EndOfData) and indices_to_skip[item.get('_idx')]:
            return

        # If the item is not in the skip set, continue processing
        async for result in generator(item):
            yield result

    return wrapped


def skip_indices_data_source(item_source, indices_to_skip: bitarray):
    """
    Skips processing items with indices in the provided `indices_to_skip` set.

    Args:
        indices_to_skip: A set of indices that should be skipped.

    Returns:
        A wrapped function that checks if the item should be skipped.
    """

    async def wrapped():

        # If the item is not in the skip set, continue processing
        async for item in item_source():
            if not isinstance(item, EndOfData) and item.get('_idx') < len(indices_to_skip) and indices_to_skip[
                item.get('_idx')]:  # _idx is individual for every single item TODO review this logic
                continue
            yield item

    return wrapped
