import logging
import signal
from typing import Any, Dict, List
import contextlib
import gc, torch
log = logging.getLogger(__name__)


# re-use the normalization helper from your server

def normalize_chat_input(user_input: Any) -> List[List[Dict[str, Any]]]:
    if isinstance(user_input, str):
        return [[{"role": "user", "content": user_input}]]
    if isinstance(user_input, list) and all(isinstance(x, str) for x in user_input):
        return [[{"role": "user", "content": s}] for s in user_input]
    if isinstance(user_input, list) and all(isinstance(x, dict) for x in user_input):
        return [user_input]
    if isinstance(user_input, list) and all(isinstance(x, list) for x in user_input):
        return user_input
    if isinstance(user_input, dict):
        return [[user_input]]
    raise ValueError(f"Unsupported chat input type: {type(user_input)}")


# ---------------- worker function that runs inside subprocess ----------------

import os
import sys
import traceback
import logging


def _worker_main(init_q, request_q, result_q, model_name, llm_kwargs, gpu_ids, debug):
    """
    Runs in subprocess. Communicates:
      - initialization status via `init_q` (single message: {"status":"ok"} or {"status":"error", ...})
      - runtime results/errors via `result_q` (as before)

    NOTE: we intentionally DO NOT use the runtime `result_q` to indicate initialization.
    """
    llm = None

    def shutdown(*args, **kwargs):
        # print("vllm worker attempting graceful shutdown with args=%s kwargs=%s", args, kwargs)
        try:
            result_q.put(None)
        except Exception:
            pass
        try:
            if llm and hasattr(llm, "close"):
                llm.close()
        except Exception:
            pass
        try:
            with contextlib.suppress(AssertionError):
                torch.distributed.destroy_process_group()
            gc.collect()
            torch.cuda.empty_cache()
            del llm.llm_engine.model_executor.driver_worker
            del llm.llm_engine.model_executor
            del llm
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)

    # silence child process logs for non-debug
    if not debug:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
        logging.disable(logging.CRITICAL)

    if gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    try:
        # local import to avoid vllm import in parent process
        from vllm import LLM, SamplingParams  # noqa: F401
        # Initialize LLM
        llm = LLM(model=model_name, **(llm_kwargs or {}))
        # report success of initialization (only to init_q)
        try:
            init_q.put({"status": "ok"})
        except Exception:
            # if we cannot report init, still continue � parent will eventually time out or handle.
            pass
    except Exception:
        tb = traceback.format_exc()
        log.error("vllm worker init failed: %s", tb)
        try:
            pid = os.getpid()
            init_q.put(
                {"status": "error", "error": "llm_init_failed", "pid": pid, "traceback": tb.replace('\\n', '\n')})
        except Exception:
            pass
        # don't continue main loop: parent should handle this init error
        return

    # main loop: serve requests over request_q and put responses into result_q
    try:
        while True:
            req = request_q.get()
            if req is None:
                # parent asked to exit
                break
            try:
                req_id = req.get("id")
                method = req.get("method")
                args = req.get("args", []) or []
                kwargs = req.get("kwargs", {}) or {}

                if method == "chat":
                    messages_raw = args[0] if args else kwargs.get("messages")
                    sampling_params = kwargs.get("sampling_params")
                    prompts = normalize_chat_input(messages_raw)
                    outputs = llm.chat(prompts, sampling_params, use_tqdm=False)
                    result_q.put({"id": req_id, "status": "ok", "result": outputs})

                elif method == "generate":
                    prompts = args[0] if args else kwargs.get("prompts")
                    sampling_params = kwargs.get("sampling_params")
                    if isinstance(prompts, str):
                        prompts = [prompts]
                    if not (isinstance(prompts, list) and all(isinstance(p, str) for p in prompts)):
                        raise ValueError("prompts must be str or list[str]")
                    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
                    result_q.put({"id": req_id, "status": "ok", "result": outputs})

                else:
                    result_q.put({"id": req_id, "status": "error", "error": f"unknown_method:{method}"})

            except Exception as e:
                tb = traceback.format_exc()
                log.error("Exception inside vllm worker runtime loop: %s", tb)
                result_q.put({"id": req.get("id"), "status": "error", "error": str(e), "traceback": tb})
    finally:
        shutdown()
