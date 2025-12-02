import logging
import os
import signal

log = logging.getLogger(__name__)

import asyncio
import logging
from typing import Any, Dict, List, Optional
import aioprocessing
import torch
from .encoder_worker import _encoder_worker_main
from ..worker_client import WorkerClient


async def wait_with_timeout(init_queues, timeout: float):
    tasks = [asyncio.ensure_future(q.coro_get()) for q in init_queues]

    done, pending = await asyncio.wait(tasks, timeout=timeout)

    results = []
    done_indices = set()

    for idx, t in enumerate(tasks):
        if t in done:
            done_indices.add(idx)
            try:
                results.append(t.result())
            except Exception as e:
                logging.error(f"Task failed: {e}")

    if pending:
        logging.warning(
            f"Only {len(done)} / {len(tasks)} init tasks finished within {timeout}s"
        )
        for p in pending:
            p.cancel()

    return results, done_indices


async def start_encoder_workers(
    model_name: str,
    cache_path,
    model_kwargs: Optional[Dict[str, Any]] = None,
    gpus: Optional[List[int]] = None,
    models_per_gpu: int = 1,
):
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
                args=(
                    init_q,
                    request_q,
                    response_q,
                    model_name,
                    model_kwargs or {},
                    [gpu_id],
                    logging.INFO,
                    cache_path
                ),
            )
            processes.append(proc)
            proc.start()

            wc = WorkerClient(request_q, response_q, proc, shutdown, _id=f"{gpu_id}-{j}")
            workers.append(wc)

    # Wait for init results
    init_results, done_indices = await wait_with_timeout(init_queues, timeout=90.0)

    # Filter: keep only workers whose init queue responded
    if len(done_indices) == 0:
        log.error("No encoder workers initialized successfully.")
        shutdown()
        raise RuntimeError("No encoder workers initialized.")

    # Drop uninitialized workers
    filtered_workers = []
    filtered_processes = []
    filtered_response_queues = []

    for idx, (w, p, rq) in enumerate(zip(workers, processes, response_queues)):
        if idx in done_indices:
            filtered_workers.append(w)
            filtered_processes.append(p)
            filtered_response_queues.append(rq)
        else:
            # kill dead/uninitialized worker
            if p.is_alive():
                os.kill(p.pid, signal.SIGTERM)
                p.join(timeout=3.0)
                if p.is_alive():
                    os.kill(p.pid, signal.SIGKILL)
                    p.join(timeout=1.0)

    workers = filtered_workers
    processes = filtered_processes
    response_queues = filtered_response_queues

    # Validate init messages
    init_errors = [
        (idx, msg)
        for idx, msg in enumerate(init_results)
        if not (isinstance(msg, dict) and msg.get("status") == "ok")
    ]

    if init_errors:
        print("Error during initialization, shutting down...")
        shutdown()
        raise RuntimeError(f"Encoder worker init failed: {init_errors}")

    print(f"Successfully initialized {len(workers)} models!")
    return workers, shutdown