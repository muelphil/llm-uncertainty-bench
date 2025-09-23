import sys

from ...utils.end_of_data import EndOfData


def progress_wrapper(generator, progress_bar):
    """
    A wrapper that updates a tqdm progress bar every time an item is yielded.

    Args:
        generator: The async generator to wrap.
        progress_bar: An instance of `tqdm` progress bar.

    Returns:
        A wrapped generator function.
    """

    async def wrapped(item):

        async for result in generator(item):
            if isinstance(result, EndOfData):
                progress_bar.refresh()
                sys.stdout.flush()
                sys.stderr.flush()
                progress_bar.disable = True
            else:
                progress_bar.update(1)  # Increment the progress bar
            yield result  # Yield the item as usual

    return wrapped
