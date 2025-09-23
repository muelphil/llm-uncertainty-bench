import gc
import logging
import os
import signal
import string
import sys
import traceback
from typing import List, Union

import diskcache as dc
import torch
import torch.nn as nn
from cachetools import LRUCache, cached
from transformers import DebertaForSequenceClassification, DebertaTokenizer

log = logging.getLogger(__name__)


def _strip(w: str):
    return w.strip(string.punctuation + " \n")


worker_cache: Union[dc.Cache, None] = None
lru_cache = LRUCache(maxsize=100000)


@cached(cache=lru_cache)
def fast_get(k):
    return worker_cache.get(k)


def _prepare_nli_lists(greedy_alternatives):
    """
    Build nli_list (with None placeholders) and a queue of missing pairs.
    """

    nli_queue = []
    greedy_alternatives_nli = []

    for g_idx, sample_alternatives in enumerate(greedy_alternatives):
        nli_list = [[None] * (len(alternatives) - 1) for alternatives in sample_alternatives]
        for w_idx, word_alternatives in enumerate(sample_alternatives):
            word = _strip(word_alternatives[0][0])
            for alt_idx, alt in enumerate(word_alternatives[1:]):
                token = _strip(alt[0])
                token_tuple = (word, token) if word < token else (token, word)
                cached = fast_get(token_tuple)
                if cached is not None:
                    nli_list[w_idx][alt_idx] = cached
                else:
                    nli_queue.append((token_tuple, g_idx, w_idx, alt_idx))
        greedy_alternatives_nli.append(nli_list)

    return greedy_alternatives_nli, nli_queue


def _process_nli_queue(nli_queue, nli_list, model, tokenizer, device, batch_size,
                       ent_id, contra_id, neut_id):
    """
    Run batches, fill nli_list, update cache.
    """
    global worker_cache
    from collections import defaultdict
    occ = defaultdict(list)
    for token_tuple, g_idx, w_idx, alt_idx in nli_queue:
        occ[token_tuple].append((g_idx, w_idx, alt_idx))
    unique_token_tuples = list(occ.keys())
    if not unique_token_tuples:
        return

    # Build forward/backward pairs
    pairs = []
    for (a, b) in unique_token_tuples:
        pairs.append((a, b))
        pairs.append((b, a))

    all_preds: List[int] = []
    with torch.inference_mode():
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            firsts = [p[0] for p in batch_pairs]
            seconds = [p[1] for p in batch_pairs]

            encoded = tokenizer(
                firsts, seconds,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            encoded = {k: v.to(device, non_blocking=True) for k, v in encoded.items()}

            logits = model(**encoded).logits
            pred_ids = logits.argmax(dim=1).cpu().tolist()

            for pid in pred_ids:
                if pid == ent_id:
                    all_preds.append(1)
                elif pid == contra_id:
                    all_preds.append(-1)
                elif pid == neut_id:
                    all_preds.append(0)
                else:
                    raise ValueError(f"Unexpected class id {pid}")

    # Combine forward/backward
    for u_idx, token_tuple in enumerate(unique_token_tuples):
        forward = all_preds[2 * u_idx]
        backward = all_preds[2 * u_idx + 1]
        combined = _combine_nli(forward, backward)
        worker_cache[token_tuple] = combined
        lru_cache.pop(token_tuple, None)
        for (g_idx, w_idx, alt_idx) in occ[token_tuple]:
            nli_list[g_idx][w_idx][alt_idx] = combined


def _combine_nli(forward: int, backward: int) -> int:
    """
    Combine two integer NLI predictions NLI(x,y) and NLI(y,x) into a single integer:
      -1 = contradiction
       0 = neutral
       1 = entailment

    Rules (equivalent to your previous string logic):
    - If both agree -> that value
    - If one says entail (1) and the other says contra (-1) -> neutral (0)
    - If one says entail or contra and the other is neutral -> return the entail/contra
    - Otherwise -> neutral
    """
    if forward == backward:
        return forward
    # if one says entail (1) and the other says contra (-1) => neutral
    if {forward, backward} == {1, -1}:
        return 0
    # if either side is entail or contra, prefer it
    if forward in (1, -1):
        return forward
    if backward in (1, -1):
        return backward
    return 0


def _encoder_worker_main(init_q, request_q, result_q, model_name, model_kwargs, gpu_ids, debug, cache_path):
    """
    Runs in subprocess. Handles encoder model requests (token_tuples -> combined NLI results).
    """
    global worker_cache

    model = None
    tokenizer = None
    device = "cpu"

    def shutdown(*args, **kwargs):
        try:
            result_q.put(None)
        except Exception:
            pass
        try:
            if model is not None:
                del model
        except Exception:
            pass
        try:
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)

    if not debug:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
        logging.disable(logging.CRITICAL)

    if gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        device = "cuda" if len(gpu_ids) > 0 else "cpu"

    try:

        # Create cache in worker
        worker_cache = dc.Cache(
            cache_path,
            size_limit=20 * 1024 ** 3,
            disk_min_file_size=2 ** 18,
            eviction_policy='none'
        )
        tokenizer = DebertaTokenizer.from_pretrained(model_name)
        model = DebertaForSequenceClassification.from_pretrained(
            model_name,
            problem_type="multi_label_classification",
            **(model_kwargs or {})
        )
        model = torch.compile(model)  # optional but often a big win
        model.to(device)
        model.eval()
        torch.backends.cudnn.benchmark = True

        # Label IDs from config
        ent_id = model.config.label2id["ENTAILMENT"]
        contra_id = model.config.label2id["CONTRADICTION"]
        neut_id = model.config.label2id["NEUTRAL"]

        softmax = nn.Softmax(dim=1)
        init_q.put({"status": "ok"})
    except Exception:
        tb = traceback.format_exc()
        try:
            pid = os.getpid()
            init_q.put({
                "status": "error",
                "error": "encoder_init_failed",
                "pid": pid,
                "traceback": tb.replace('\\n', '\n')
            })
        except Exception:
            pass
        return

    # Main loop
    try:
        print("starting main loop!")
        while True:
            req = request_q.get()
            if req is None:
                break
            try:
                req_id = req.get("id")
                method = req.get("method")
                args = req.get("args", []) or []
                kwargs = req.get("kwargs", {}) or {}

                if method == "encode":
                    # Expect: list of unique token_tuples [(a, b), ...]
                    token_tuples = args[0] if args else kwargs.get("token_tuples")
                    batch_size = kwargs.get("batch_size", 512)

                    # Build forward/backward pairs
                    pairs = []
                    for (a, b) in token_tuples:
                        pairs.append((a, b))
                        pairs.append((b, a))

                    # model.to(device)
                    # model.eval()

                    torch.backends.cudnn.benchmark = True

                    all_preds: List[int] = []
                    with torch.inference_mode():
                        for i in range(0, len(pairs), batch_size):
                            batch_pairs = pairs[i:i + batch_size]
                            firsts = [p[0] for p in batch_pairs]
                            seconds = [p[1] for p in batch_pairs]

                            encoded = tokenizer(
                                firsts, seconds,
                                padding=True,
                                truncation=True,
                                return_tensors="pt"
                            )
                            encoded = {k: v.to(device, non_blocking=True) for k, v in encoded.items()}

                            logits = model(**encoded).logits

                            pred_ids = logits.argmax(dim=1).cpu().tolist()
                            # Map pred_id (0/1/2) -> pred_int (-1/0/1) by subtracting 1
                            # for pid in pred_ids:
                            #     all_pred_ints.append(int(pid) - 1)
                            # pred_ids = probs.argmax(dim=1).tolist()

                            # Convert directly to -1/0/1
                            for pid in pred_ids:
                                if pid == ent_id:
                                    all_preds.append(1)
                                elif pid == contra_id:
                                    all_preds.append(-1)
                                elif pid == neut_id:
                                    all_preds.append(0)
                                else:
                                    raise ValueError(
                                        f"Predicted id {pid} not in possible ids of classes ENTAILMENT {ent_id}, CONTRADICTION {contra_id}, NEUTRAL {neut_id}")

                    # Combine forward/backward predictions per token_tuple
                    results: List[int] = []
                    for u_idx in range(len(token_tuples)):
                        forward = all_preds[2 * u_idx]
                        backward = all_preds[2 * u_idx + 1]
                        combined = _combine_nli(forward, backward)
                        results.append(combined)

                    result_q.put({"id": req_id, "status": "ok", "result": results})

                elif method == "encode_full":
                    # Args: greedy_alternatives
                    greedy_alternatives = args[0] if args else kwargs.get("greedy_alternatives")
                    batch_size = kwargs.get("batch_size", 512)

                    greedy_alternatives_nli, nli_queue = _prepare_nli_lists(greedy_alternatives)
                    if nli_queue:
                        _process_nli_queue(
                            nli_queue, greedy_alternatives_nli,
                            model, tokenizer, device, batch_size,
                            ent_id, contra_id, neut_id,
                        )

                    result_q.put({"id": req_id, "status": "ok", "result": greedy_alternatives_nli})
                else:
                    result_q.put({
                        "id": req_id,
                        "status": "error",
                        "error": f"unknown_method:{method}"
                    })

            except Exception as e:
                tb = traceback.format_exc()
                result_q.put({
                    "id": req.get("id"),
                    "status": "error",
                    "error": str(e),
                    "traceback": tb
                })
    finally:
        shutdown()
