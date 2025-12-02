import asyncio
import inspect
from typing import Any, Iterable, List, Optional, Tuple, Union, Callable, Awaitable

ResourceCloseFunc = Union[Callable[[], None], Callable[[], Awaitable[None]]]


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
            await self._pool.release_resource(r)


class ResourcePool:
    """
    Resource pool that automatically rebinds its internal queue
    whenever used from a new event loop.
    """

    def __init__(self, resources: Iterable[Any], stack_mode: bool = False):
        resources = list(resources)
        self._resources = resources                # NEW: raw storage
        self._stack_mode = stack_mode
        self._total = len(resources)

        # Stable IDs
        self._resource_ids = {id(r): i for i, r in enumerate(resources)}
        self._use_counts = {i: 0 for i in range(self._total)}

        self._queue = None                         # NEW: lazy initialization
        self._loop = None                          # NEW: event loop tracking

        self._on_close: List[ResourceCloseFunc] = []

    # ----------------------------
    # Queue (re)initialization
    # ----------------------------
    def _ensure_queue(self):
        """
        Ensure the internal queue exists and is bound to the *current* event loop.
        If called from a different loop, reinitialize queue.
        """
        loop = asyncio.get_running_loop()

        # Need to initialize or reinitialize?
        if self._queue is None or self._loop is not loop:
            # Choose correct queue type
            if self._stack_mode:
                q = asyncio.LifoQueue(maxsize=self._total)
            else:
                q = asyncio.Queue(maxsize=self._total)

            # refill queue
            for r in self._resources:
                q.put_nowait(r)

            self._queue = q
            self._loop = loop

    # ----------------------------
    # Acquire
    # ----------------------------
    async def acquire(self, n: int = 1) -> ResourceHandle:
        if n <= 0:
            raise ValueError("n must be >= 1")

        self._ensure_queue()       # <---- IMPORTANT

        async def _get_one():
            res = await self._queue.get()
            self._use_counts[self._resource_ids[id(res)]] += 1
            return res

        if n == 1:
            return ResourceHandle(self, [await _get_one()])

        resources = [await _get_one() for _ in range(n)]
        return ResourceHandle(self, resources)

    # ----------------------------
    # Release
    # ----------------------------
    async def release_resource(self, r: Any):
        self._ensure_queue()       # safe even if already correct

        try:
            self._queue.put_nowait(r)
        except asyncio.QueueFull:
            await self._queue.put(r)

    # ----------------------------
    # Misc
    # ----------------------------
    def available(self) -> int:
        self._ensure_queue()
        return self._queue.qsize()

    def total(self) -> int:
        return self._total

    def __repr__(self):
        return f"<ResourcePool total={self._total} available={self.available()}>"

    def on_close(self, func: ResourceCloseFunc):
        self._on_close.append(func)

    def close(self):
        # DO NOT call asyncio.run() here — it breaks loops
        for func in self._on_close:
            if inspect.iscoroutinefunction(func):
                # schedule into current loop instead
                loop = asyncio.get_event_loop()
                loop.create_task(func())
            else:
                func()

    def get_usage_distribution(self) -> str:
        """Return a summary of how often each resource was used."""
        total_uses = sum(self._use_counts.values()) or 1
        parts = []
        for rid, count in sorted(self._use_counts.items()):
            pct = (count / total_uses) * 100
            parts.append(f"Resource {rid}: {count} ({pct:.1f}%)")
        return "[Resource usage distribution] " + ", ".join(parts)


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
