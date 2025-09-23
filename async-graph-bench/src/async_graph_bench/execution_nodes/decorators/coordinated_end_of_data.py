import asyncio
from functools import wraps

from ...utils.end_of_data import EndOfData


def coordinated_end_of_data(generator, name):
    """
    Ensures that EndOfData is only passed downstream once all parallel instances
    of the wrapped async generator are complete.
    """
    active_instances = 0
    end_of_data_seen = False
    end_of_data = None
    lock = asyncio.Lock()

    async def wrapped(item):
        nonlocal active_instances, end_of_data_seen, end_of_data

        async with lock:
            active_instances += 1

        try:
            if isinstance(item, EndOfData):
                async with lock:
                    end_of_data_seen = True
                    end_of_data = item
                # Don't forward EndOfData yet — wait until all finish
                return

            async for result in generator(item):
                yield result

        finally:
            send_end = False
            async with lock:
                active_instances -= 1
                if end_of_data_seen and active_instances:
                    send_end = True
            if send_end:
                # All done — now release EndOfData downstream
                # print(f"[coordinated_end_of_data] EndOfData seen in {name}, all instances of this node finished, pushing EndOfData") #TODO
                async for result in generator(end_of_data):
                    yield result
                # print(f"[coordinated_end_of_data] EndOfData seen from {name} emitted") #TODO
            # elif end_of_data_seen:
            #     print(f"[coordinated_end_of_data] EndOfData seen in {name}, but there are still {active_instances} instances of this node running, waiting for them to finish...") #TODO

    return wrapped
