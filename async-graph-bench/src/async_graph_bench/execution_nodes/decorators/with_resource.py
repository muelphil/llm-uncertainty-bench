from typing import List

from ...utils import acquire_from_many
from ...utils.end_of_data import EndOfData


def with_resources(generator, resource_pools: List["ResourcePool"]):
    """
    Decorator that acquires resources from one or multiple ResourcePools,
    passes them into the generator, and ensures resources are released
    after use.

    Args:
        generator: The async generator function for the node.
                   Must accept (item, *resources).
        resource_pools: A list of ResourcePool instances.

    Returns:
        An async wrapper that manages resource acquisition/release.
    """

    if len(resource_pools) == 1:
        pool = resource_pools[0]

        async def single_resource_wrapper(item):
            if isinstance(item, EndOfData):
                async for output in generator(item):
                    yield output
            else:
                async with await pool.acquire() as resource:
                    async for output in generator(item, resource):
                        yield output

        return single_resource_wrapper

    # Case 2: multiple ResourcePools
    else:
        async def multi_resource_wrapper(item):
            if isinstance(item, EndOfData):
                async for output in generator(item):
                    yield output
            else:
                async with await acquire_from_many(resource_pools) as resources:
                    # ensure tuple, so we can unpack consistently
                    if not isinstance(resources, tuple):
                        resources = (resources,)
                    async for output in generator(item, *resources):
                        yield output

        return multi_resource_wrapper