import logging
import os
import signal
import time
from typing import Tuple

log = logging.getLogger(__name__)

# vllm_pool.py (or your module that contains start_workers)
import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable
import aioprocessing

from .vllm_worker import _worker_main
from ..worker_client import WorkerClient  # updated below

log = logging.getLogger(__name__)


def chunk_gpus(gpus: List[int], gpus_per_worker: int) -> List[List[int]]:
    chunks = []
    for i in range(0, len(gpus), gpus_per_worker):
        if i + gpus_per_worker <= len(gpus):
            chunks.append(gpus[i:i + gpus_per_worker])
    return chunks


async def start_workers(model_name: str, gpus_per_worker: int, llm_kwargs: Optional[Dict[str, Any]] = None,
                        gpus: Optional[List[Optional[int]]] = None, log_level=logging.INFO) -> Tuple[
    List[WorkerClient], Callable]:
    """
    Start several subprocesses running vLLM instances.

    Each worker gets a dedicated `init_q` and MUST put a single init message into it.
    start_workers waits for all init messages and fails fast on any init error.
    A shared asyncio.Event `shutdown_event` is used to interrupt pending client waits if any worker fails at runtime.
    """
    start_time = time.time()
    assert len(gpus) >= gpus_per_worker, "Not enough GPUs to start even one worker."
    gpu_chunks = chunk_gpus(gpus or [], gpus_per_worker=gpus_per_worker)
    num_workers = len(gpu_chunks)

    workers: List[WorkerClient] = []
    processes: List[aioprocessing.AioProcess] = []
    response_queues: List[aioprocessing.AioQueue] = []
    init_queues: List[aioprocessing.AioQueue] = []

    # Shared asyncio event to notify worker clients of global shutdown
    shutdown_in_progress = False

    def shutdown():
        nonlocal shutdown_in_progress
        """Synchronous-ish shutdown/helper for immediate termination. It's safe to call multiple times."""
        if shutdown_in_progress:
            return
        log.info("Global shutdown requested: terminating vLLM subprocesses...")
        shutdown_in_progress = True
        for proc in processes:
            if proc.is_alive():
                os.kill(proc.pid, signal.SIGTERM)
        for proc in processes:
            proc.join(timeout=7.0)
            if proc.is_alive():
                os.kill(proc.pid, signal.SIGINT)
                proc.join(timeout=3.0)
                if proc.is_alive():
                    os.kill(proc.pid, signal.SIGKILL)
                    proc.join(timeout=3.0)
                log.warning(f"Process {proc.pid} killed")
            else:
                log.warning(f"Process {proc.pid} terminated")

        for rq in response_queues:
            try:
                rq.put_nowait(None)
            except Exception:
                pass
        log.info("Global shutdown complete.")

    # create transports and start processes
    for i in range(num_workers):
        request_q = aioprocessing.AioQueue()
        response_q = aioprocessing.AioQueue()
        init_q = aioprocessing.AioQueue()
        response_queues.append(response_q)
        init_queues.append(init_q)

        gpu_ids = gpu_chunks[i]
        proc = aioprocessing.AioProcess(
            target=_worker_main,
            args=(init_q, request_q, response_q, model_name, llm_kwargs or {}, gpu_ids,
                  log_level if i == 0 else logging.CRITICAL)
        )
        processes.append(proc)
        proc.start()

        wc = WorkerClient(request_q, response_q, proc, shutdown, _id=i)
        workers.append(wc)

    # Wait concurrently for init messages from each worker.
    # NOTE: AioQueue.coro_get() returns an awaitable Future (not a coroutine), so use ensure_future.
    init_tasks = [asyncio.ensure_future(q.coro_get()) for q in init_queues]

    try:
        init_results = await asyncio.gather(*init_tasks)
    except Exception as e:
        # ensure we cancel any pending init tasks and shutdown
        for t in init_tasks:
            if not t.done():
                t.cancel()
        shutdown()
        await asyncio.gather(*(w.stop() for w in workers), return_exceptions=True)
        raise RuntimeError("Failed to receive initialization messages from workers") from e

    # inspect init results
    init_errors = []
    for idx, msg in enumerate(init_results):
        if not isinstance(msg, dict) or msg.get("status") != "ok":
            init_errors.append((idx, msg))

    if init_errors:
        # one or more workers failed to initialize: kill everything and raise
        log.info("Shutting down vLLM workers due to error on initialization...")
        try:
            shutdown()
            # await asyncio.gather(*(w.stop() for w in workers), return_exceptions=True)
        except Exception:  # stopping the workers will again create an exception, ignore this one
            pass

        traceback = "\n\n".join(
            [f"Error {idx} in Process {err['pid']} - Traceback: {err['traceback']}" for idx, err in init_errors])
        raise RuntimeError(f"One or more workers failed to initialize:\n{traceback}")
    end_time = time.time()
    elapsed = time.strftime("%Hh %Mm %Ss", time.gmtime(end_time - start_time))

    # all workers initialized successfully
    log.info(f"All {len(workers)} vLLM workers initialized successfully in {elapsed}.")
    return workers, shutdown
