import logging
import os
import signal

log = logging.getLogger(__name__)

# vllm_pool.py (or your module that contains start_workers)
import asyncio
import logging
from typing import Any, Dict, List, Optional
import aioprocessing
import torch
from .encoder_worker import _encoder_worker_main
from ..multi_vllm_instances.worker_client import WorkerClient

log = logging.getLogger(__name__)


async def wait_with_timeout(init_queues, timeout: float):
    # Create tasks
    tasks = [asyncio.ensure_future(q.coro_get()) for q in init_queues]

    done, pending = await asyncio.wait(tasks, timeout=timeout)

    results = []
    for t in done:
        try:
            results.append(t.result())
        except Exception as e:
            logging.error(f"Task failed: {e}")

    if pending:
        logging.warning(
            f"Only {len(done)} / {len(tasks)} init tasks finished within {timeout}s"
        )
        # If you want, cancel them
        for p in pending:
            p.cancel()

    return results

# TODO can I dcombine this with start_workers to avoid duplicate code?
async def start_encoder_workers(model_name: str, cache_path, model_kwargs: Optional[Dict[str, Any]] = None,
                                gpus: Optional[List[int]] = None, models_per_gpu: int = 1):
    """
    Start subprocesses running encoder models.

    Each GPU can host multiple encoder models (lighter than LLMs).
    """
    gpus = gpus or list(range(torch.cuda.device_count()))
    processes = []
    workers = []
    response_queues = []
    init_queues = []

    shutdown_in_progress = False

    def shutdown():
        nonlocal shutdown_in_progress
        if shutdown_in_progress:
            return
        shutdown_in_progress = True
        for proc in processes:
            if proc.is_alive():
                os.kill(proc.pid, signal.SIGTERM)
        for proc in processes:
            proc.join(timeout=5.0)
            if proc.is_alive():
                os.kill(proc.pid, signal.SIGKILL)
                proc.join(timeout=2.0)
        for rq in response_queues:
            try:
                rq.put_nowait(None)
            except Exception:
                pass

    # Start workers
    for gpu_id in gpus:
        for j in range(models_per_gpu):
            request_q = aioprocessing.AioQueue()
            response_q = aioprocessing.AioQueue()
            init_q = aioprocessing.AioQueue()
            response_queues.append(response_q)
            init_queues.append(init_q)

            proc = aioprocessing.AioProcess(
                target=_encoder_worker_main,
                args=(init_q, request_q, response_q, model_name, model_kwargs or {}, [gpu_id],
                      True, cache_path
                      # (gpu_id == gpus[0] and j == 0)
                )
            )
            processes.append(proc)
            proc.start()

            wc = WorkerClient(request_q, response_q, proc, shutdown, _id=f"{gpu_id}-{j}")
            workers.append(wc)

    # Wait for init results
    init_results = await wait_with_timeout(init_queues, timeout=90.0)

    # init_tasks = [asyncio.ensure_future(q.coro_get()) for q in init_queues]
    # init_results = await asyncio.gather(*init_tasks)

    init_errors = [(idx, msg) for idx, msg in enumerate(init_results) if
                   not (isinstance(msg, dict) and msg.get("status") == "ok")]
    if init_errors:
        print(f"error during initialization, shutting down...")
        shutdown()
        raise RuntimeError(f"Encoder worker init failed: {init_errors}")
    else:
        print(f"successfully initialized {len(workers)} models!")

    return workers, shutdown
