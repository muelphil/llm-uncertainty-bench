# server.py
import argparse
import pickle
import traceback
import gc
import sys
from typing import Any, Dict, List
from fastapi import FastAPI, Request, Response, HTTPException

DEBUG = True  # always True
MODEL_NAME: str = ""
LLM_KWARGS: Dict[str, Any] = {}
ALLOW_REMOTE = False
PORT = 8000
llm = None
is_initialized = False

app = FastAPI()

def unknown_args_to_dict(unknown_args):
    """
    Transforms a list of unknown command-line arguments into a dictionary.
    Values are parsed as bool, int, or str as appropriate.

    Args:
        unknown_args: List of strings, e.g. ['--foo', '42', '--bar', 'hello']

    Returns:
        Dictionary with keys as argument names (without '--') and values parsed as bool, int, or str.
    """
    result = {}
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if arg.startswith('--'):
            key = arg[2:].replace('-', '_')
            if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith('--'):
                # Try to parse as int
                try:
                    value = int(unknown_args[i + 1])
                except ValueError:
                    # Try to parse as bool (if 'true' or 'false', case-insensitive)
                    val_lower = unknown_args[i + 1].lower()
                    if val_lower in ('true', 'false'):
                        value = val_lower == 'true'
                    else:
                        value = unknown_args[i + 1]
                result[key] = value
                i += 2
            else:
                # Flag with no value: treat as True
                result[key] = True
                i += 1
        else:
            i += 1
    return result



def normalize_chat_input(user_input: Any) -> List[List[Dict[str, Any]]]:
    """Forgiving normalization to list-of-conversations format."""
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


def client_is_local(request: Request) -> bool:
    if ALLOW_REMOTE:
        return True
    host = request.client.host if request.client else ""
    return host in ("127.0.0.1", "::1", "localhost")


@app.on_event("startup")
async def startup_event():
    global llm, is_initialized
    from vllm import LLM
    if DEBUG:
        print(f"[DEBUG] Initializing LLM: {MODEL_NAME}, kwargs={LLM_KWARGS}")
    llm = LLM(model=MODEL_NAME, **LLM_KWARGS)
    is_initialized = True


@app.on_event("shutdown")
def shutdown_event():
    global llm
    if not llm:
        return
    try:
        if hasattr(llm, "close"):
            llm.close()
    except Exception:
        pass
    try:
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass
    if DEBUG:
        print("[DEBUG] LLM shutdown complete")


@app.post("/chat")
async def chat_endpoint(request: Request):
    if not client_is_local(request):
        raise HTTPException(status_code=403, detail="Local requests only")
    body = await request.body()
    try:
        messages_raw, sampling_kwargs = pickle.loads(body)
        prompts = normalize_chat_input(messages_raw)
        from vllm import SamplingParams
        sp = SamplingParams(**sampling_kwargs) if sampling_kwargs else None
        outputs = llm.chat(prompts, sp, use_tqdm=False)
        return Response(content=pickle.dumps(outputs),
                        media_type="application/octet-stream")
    except Exception as e:
        if DEBUG:
            tb = traceback.format_exc()
            return Response(content=pickle.dumps({"status": "error", "error": str(e), "traceback": tb}),
                            media_type="application/octet-stream", status_code=500)
        raise HTTPException(status_code=500, detail="Chat failed")


@app.post("/generate")
async def generate_endpoint(request: Request):
    if not client_is_local(request):
        raise HTTPException(status_code=403, detail="Local requests only")
    body = await request.body()
    try:
        prompts, sampling_kwargs = pickle.loads(body)
        if isinstance(prompts, str):
            prompts = [prompts]
        if not (isinstance(prompts, list) and all(isinstance(p, str) for p in prompts)):
            raise ValueError("prompts must be str or list[str]")
        from vllm import SamplingParams
        sp = SamplingParams(**sampling_kwargs) if sampling_kwargs else None
        outputs = llm.generate(prompts, sp, use_tqdm=False)
        return Response(content=pickle.dumps({"status": "ok", "outputs": outputs}),
                        media_type="application/octet-stream")
    except Exception as e:
        if DEBUG:
            tb = traceback.format_exc()
            return Response(content=pickle.dumps({"status": "error", "error": str(e), "traceback": tb}),
                            media_type="application/octet-stream", status_code=500)
        raise HTTPException(status_code=500, detail="Generate failed")

@app.get("/health")
async def health_endpoint():
    if is_initialized == False:
        raise HTTPException(status_code=503, detail="LLM not initialized")
    return {"status": "ok", "model": MODEL_NAME}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--allow-remote", action="store_true", default=False)

    args, unknown = parser.parse_known_args()
    unknown_dict = unknown_args_to_dict(unknown)

    import json
    MODEL_NAME = args.model
    PORT = args.port
    ALLOW_REMOTE = args.allow_remote
    LLM_KWARGS = unknown_dict

    import uvicorn
    host = "0.0.0.0" if ALLOW_REMOTE else "127.0.0.1"
    uvicorn.run(app, host=host, port=PORT)
