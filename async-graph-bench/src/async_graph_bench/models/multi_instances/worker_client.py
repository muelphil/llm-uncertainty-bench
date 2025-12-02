import asyncio
import logging
import os
import signal
import uuid
from typing import Any

import aioprocessing

log = logging.getLogger(__name__)


class WorkerClient:
    def __init__(
            self,
            request_q: aioprocessing.AioQueue,
            result_q: aioprocessing.AioQueue,
            proc: aioprocessing.AioProcess,
            shutdown,
            _id=None
    ):
        """
        - request_q/result_q: AioQueues for messages to/from worker
        - proc: worker process wrapper (AioProcess)
        - shutdown_callback: callable to perform global shutdown (should be safe to call multiple times)
        - shutdown_event: asyncio.Event that will be set when the system is being shutdown (other clients watch it)
        """
        self.request_q = request_q
        self.result_q = result_q
        self.proc = proc
        self._closed = False
        self.shutdown = shutdown
        self._id = _id

    async def call(self, method: str, *args, **kwargs) -> Any:
        """Send a request to the worker and await the response.
        This method will be interrupted if the shared shutdown_event is set.
        """
        if self._closed:
            raise RuntimeError(f"Worker Client {self._id} closed")

        req_id = uuid.uuid4().hex
        # log.info("WorkerClient %s sending request %s", self._id, req_id)
        await self.request_q.coro_put({
            "id": req_id,
            "method": method,
            "args": args,
            "kwargs": kwargs
        })

        try:
            res = await self.result_q.coro_get()
        except asyncio.CancelledError:
            raise RuntimeError(f"Worker Client {self._id} read cancelled")
        except Exception as e:
            # unexpected error reading queue -> treat as fatal
            try:
                self.shutdown()
            except Exception:
                pass
            raise RuntimeError(f"Worker Client {self._id} failed to read from result queue: {e}") from e

        if res is None:
            # worker signalled termination
            self._closed = True
            log.error("Worker Client %s got sentinel None from worker (worker shutting down)", self._id)
            try:
                self.shutdown()
            except Exception:
                pass
            raise RuntimeError(f"Worker Client {self._id} got interrupted (worker shutting down)")

        # log.debug("Worker Client %s got response=%s|%s", self._id, str(res.get("id")), str(res.get("status")))

        if res.get("status") == "ok":
            if res.get("id") == req_id:
                return res.get("result")
            else:
                # mismatch: indicates protocol corruption or concurrent requests on same worker
                self._closed = True
                try:
                    self.shutdown()
                except Exception:
                    pass
                raise RuntimeError(
                    "Worker process did not receive responses in correct order - only one instance "
                    "may be queried at any given time!"
                )
        else:
            # error from worker
            error_message = res.get("error") or "worker_error"
            log.error("Worker Client %s received an error: %s", self._id, error_message)
            print("Traceback=", res.get("traceback"))
            exc = RuntimeError(error_message)
            exc.traceback = res.get("traceback")
            # instruct global shutdown (kill all processes) - further WorkerClients will see shutdown_event
            try:
                self.shutdown()
            except Exception:
                pass
            raise exc

    async def stop(self):
        """Shut down the worker cleanly."""
        log.info("Worker Client %s stopping...", self._id)
        if not self._closed:
            try:
                if self.proc.is_alive():
                    os.kill(self.proc.pid, signal.SIGTERM)
                    self.proc.join(timeout=1.0)
                    if self.proc.is_alive():
                        os.kill(self.proc.pid, signal.SIGKILL)
                        self.result_q.put_nowait(None)
            except Exception:
                pass
            self._closed = True
