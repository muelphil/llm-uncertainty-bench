import asyncio
from typing import Any, Iterable, List, Optional, Tuple, Union


class ResourceHandle:
    """
    Async context manager that holds one or multiple resources and
    returns them on __aenter__. Call .release() to return them early.
    """

    def __init__(self, pool: "ResourcePool", resources: List[Any]):
        self._pool = pool
        self._resources = list(resources)
        self._released = False

    async def __aenter__(self):
        # Return single item for convenience, or tuple/list for multiple.
        return self._resources[0] if len(self._resources) == 1 else tuple(self._resources)

    async def __aexit__(self, exc_type, exc, tb):
        await self.release()

    async def release(self):
        """Return resources back to the pool. Safe to call multiple times."""
        if self._released:
            return
        self._released = True
        # Put back each resource. We use put_nowait as a fast path;
        # if that fails (rare), we await put().
        for r in self._resources:
            try:
                self._pool._queue.put_nowait(r)
            except asyncio.QueueFull:
                await self._pool._queue.put(r)


class ResourcePool:
    """
    Manage a set of resources (any Python objects with async methods).
    Use await pool.acquire() or `async with pool.acquire(): ...`
    To take multiple resources from the same pool, use acquire(n=2).
    """

    def __init__(self, resources: Iterable[Any]):
        resources = list(resources)
        self._total = len(resources)
        # The asyncio.Queue stores *available* resources
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=self._total)
        for r in resources:
            self._queue.put_nowait(r)

    async def acquire(self, n: int = 1, timeout: Optional[float] = None) -> ResourceHandle:
        """
        Acquire `n` resources from the pool and return a ResourceHandle.
        If timeout is set, raises asyncio.TimeoutError if resources cannot be acquired in time.
        """
        if n <= 0:
            raise ValueError("n must be >= 1")
        if n == 1:
            coro = self._queue.get()
            if timeout is None:
                res = await coro
            else:
                res = await asyncio.wait_for(coro, timeout=timeout)
            return ResourceHandle(self, [res])

        # Acquire n resources sequentially (order doesn't matter inside pool)
        async def _get_n():
            got = []
            for _ in range(n):
                got.append(await self._queue.get())
            return got

        if timeout is None:
            resources = await _get_n()
        else:
            resources = await asyncio.wait_for(_get_n(), timeout=timeout)
        return ResourceHandle(self, resources)

    def available(self) -> int:
        """Number of resources currently available."""
        return self._queue.qsize()

    def total(self) -> int:
        """Total number of resources originally in pool."""
        return self._total

    # Convenience synchronous peek (non-blocking)
    def __repr__(self):
        return f"<ResourcePool total={self._total} available={self.available()}>"


# ---- Helper: acquire resources from multiple pools in canonical order ----
class MultiResourceHandle:
    """Context manager that manages several ResourceHandles and exposes resources."""

    def __init__(self, ordered_handles: List[Tuple[int, ResourceHandle]]):
        # ordered_handles: list of (index_in_request, ResourceHandle)
        # sort to original requested order
        self._ordered_handles = [h for idx, h in sorted(ordered_handles, key=lambda t: t[0])]

    async def __aenter__(self):
        results = []
        for h in self._ordered_handles:
            obj = await h.__aenter__()  # returns resource or tuple
            results.append(obj)
        # Flatten: if user requested single resource per pool, return tuple of singletons
        if len(results) == 1:
            return results[0]
        return tuple(results)

    async def __aexit__(self, exc_type, exc, tb):
        # release in reverse order to be polite (not required)
        for h in reversed(self._ordered_handles):
            await h.__aexit__(exc_type, exc, tb)


async def acquire_from_many(
        pools: List[ResourcePool],
        counts: Optional[List[int]] = None,
        timeout: Optional[float] = None
) -> MultiResourceHandle:
    """
    Acquire resources from multiple pools. To avoid deadlocks, this helper
    acquires them in a stable canonical order (sort by id()).
    - pools: list of ResourcePool objects
    - counts: list of ints (how many from each pool) or None (1 each)
    - returns a MultiResourceHandle usable with `async with`.
    Usage:
        async with await acquire_from_many([p1,p2], counts=[1,2]) as (r1, (g1,g2)):
            ...
    """
    if counts is None:
        counts = [1] * len(pools)
    if len(counts) != len(pools):
        raise ValueError("counts must have same length as pools")

    # Canonical ordering prevents deadlocks *if every coroutine uses the same rule*.
    # Order by id(p) — stable and fast.
    order = sorted(range(len(pools)), key=lambda i: id(pools[i]))
    acquired: List[Tuple[int, ResourceHandle]] = []
    try:
        for i in order:
            pool = pools[i]
            cnt = counts[i]
            handle = await pool.acquire(n=cnt, timeout=timeout)
            acquired.append((i, handle))
    except Exception:
        # On failure, release all resources we already acquired
        for _, h in acquired:
            # release using handle.release() to avoid double __aexit__ behavior
            await h.release()
        raise

    # Return a MultiResourceHandle to be used with `async with`
    return MultiResourceHandle(acquired)
