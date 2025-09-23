import asyncio
import os
import socket
import time
import requests
from typing import List, Dict, Any, Tuple


def _get_free_ports(n: int) -> List[int]:
    """Reserve n free ports to avoid collisions."""
    sockets = []
    ports = []
    try:
        for _ in range(n):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("127.0.0.1", 0))
            ports.append(s.getsockname()[1])
            sockets.append(s)
    finally:
        for s in sockets:
            s.close()
    return ports


def _model_params_to_cli_args(model_params: Dict[str, Any]) -> List[str]:
    args = []
    for k, v in model_params.items():
        key = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:
                args.append(key)
        elif v is not None:
            args.extend([key, str(v)])
    return args


async def poll_url_until_available(url: str, interval: float = 0.5, timeout: float = 60.0) -> None:
    """Poll a URL until it returns 200 OK or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        await asyncio.sleep(interval)
    raise TimeoutError(f"Timeout waiting for {url} after {timeout}s")


# ============ NEW helper for logging first process ============
async def _pipe_to_console(stream: asyncio.StreamReader, prefix: str):
    while True:
        line = await stream.readline()
        if not line:
            break
        print(f"[{prefix}] {line.decode(errors='replace').rstrip()}")


# ===============================================================

# def _raise_on_exception(task: asyncio.Task):
#     try:
#         task.result()
#     except Exception as e:
#         # make it crash loud & early
#         asyncio.get_event_loop().call_exception_handler({
#             "message": "Unhandled exception in vLLM watcher",
#             "exception": e,
#             "task": task,
#         })
#         raise   # re-raise into the main loop if desired


# ============ NEW helper for logging first process ============
async def _pipe_to_console(stream: asyncio.StreamReader, prefix: str):
    while True:
        line = await stream.readline()
        if not line:
            break
        print(f"[{prefix}] {line.decode(errors='replace').rstrip()}")
# ===============================================================


async def start_vllm_instances(
        available_gpus: List[int],
        gpus_per_model: int,
        model: str,
        model_params: Dict[str, Any],
        vllm_executable: str = "vllm",
        poll_interval: float = 2,
        poll_timeout: float = 300,
        host: str = "127.0.0.1",
) -> List[int]:
    """
    Start multiple vllm serve instances, each bound to a chunk of GPUs,
    and wait until /v1/models is ready for all of them.
    """
    if gpus_per_model <= 0:
        raise ValueError("gpus_per_model must be >= 1")
    if len(available_gpus) < gpus_per_model:
        raise ValueError("Not enough GPUs for one instance.")

    num_instances = len(available_gpus) // gpus_per_model
    gpu_chunks = [
        available_gpus[i * gpus_per_model: (i + 1) * gpus_per_model]
        for i in range(num_instances)
    ]

    ports = _get_free_ports(num_instances)
    cli_args = _model_params_to_cli_args(model_params)

    processes: List[Tuple[int, asyncio.subprocess.Process, float]] = []

    for idx, (port, gpus) in enumerate(zip(ports, gpu_chunks)):
        cmd = [vllm_executable, "serve", model, "--port", str(port)] + cli_args
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
        env["VLLM_LOGGING_LEVEL"] = "ERROR"

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        processes.append((port, proc, time.monotonic()))

        # ============ start piping logs for the first process ============
        asyncio.create_task(_pipe_to_console(proc.stdout, f"PORT {port} STDOUT"))
        asyncio.create_task(_pipe_to_console(proc.stderr, f"PORT {port} STDERR"))
        # =================================================================

    watcher_tasks: List[asyncio.Task] = []

    # Launch watchers in background
    for port, proc, _ in processes:
        async def watch_process_exit(port=port, proc=proc):
            returncode = await proc.wait()
            print(f"Process {port} exited with returncode {returncode}")
            if returncode != 0:
                stdout = await proc.stdout.read()
                stderr = await proc.stderr.read()
                raise RuntimeError(
                    f"vllm on port {port} exited early with code {returncode}\n"
                    f"STDOUT:\n{stdout.decode(errors='replace')}\n"
                    f"STDERR:\n{stderr.decode(errors='replace')}"
                )
        t = asyncio.create_task(watch_process_exit(port, proc))
        # t.add_done_callback(_raise_on_exception)

    # Only wait for readiness
    async with asyncio.TaskGroup() as tg:
        for port, proc, start_time in processes:
            async def wait_for_ready(port=port, start_time=start_time):
                url = f"http://{host}:{port}/v1/models"
                await poll_url_until_available(url, poll_interval, poll_timeout)
                elapsed = time.monotonic() - start_time
                print(f"[INFO] vllm serve process on port {port} ready in {elapsed:.2f}s")

            tg.create_task(wait_for_ready())

    def close():
        for port, proc, _ in processes:
            if proc.returncode is None:  # still running
                proc.kill()
            else:
                print(f"[INFO] vllm serve process on port {port} was already closed before termination")


        # for _, proc, _ in processes:
        #     if proc.returncode is None:  # still running
        #         proc.terminate()
        #         try:
        #             await asyncio.wait_for(proc.wait(), 5)
        #         except asyncio.TimeoutError:
        #             proc.kill()

    return ports, close
    # TODO return a function to close the resources