import asyncio
from collections import defaultdict
from typing import Iterable, Callable, List, Optional, Union, Dict

from ...utils.end_of_data import EndOfData


def sampling(
        generator: Callable,  # async generator function to wrap (async def g(item): ...)
        dependencies: Iterable[str],  # dependencies to sample (e.g. ["dep1", "dep2"])
        sample_size: int,  # number of iterations per sample
        total_iterations: int,  # total iterations per id (must be divisible by sample_size)
        mode: str = "first",  # "first" | "spread" | "extend"
        spread_keys: Optional[Union[List[str], "all"]] = None,  # required if mode == "spread"
        # flush_incomplete: bool = False  # if True, flush incomplete samples on EndOfData
):
    assert mode in ("first", "spread", "extend"), "mode must be one of 'first','spread','extend'"
    if mode == "spread":
        assert spread_keys is not None and spread_keys == "all" or len(
            spread_keys) > 0, "spread_keys required for 'spread' mode"
    assert total_iterations % sample_size == 0, "total_iterations must be divisible by sample_size"

    groups_per_id = total_iterations // sample_size
    iteration_groups = defaultdict(
        lambda: [[None] * sample_size for _ in range(groups_per_id)]
    )
    lock = asyncio.Lock()
    end_of_data_seen = False
    eod_emitted = False  # ensure we only emit EoD once
    variations_dict = dict()

    async def wrapped(item : Union[Dict, EndOfData]):
        nonlocal end_of_data_seen, eod_emitted

        items_to_process = []
        if isinstance(item, EndOfData):
            items_to_process.append(item)
        else:
            # If this is a normal item, place in group and possibly process completed group(s)
            id_ = item["id"]
            iter_idx: int = int(item.get("iter", 0))
            group_id = iter_idx // sample_size
            idx_within = iter_idx % sample_size

            completed_variation = None

            async with lock:
                iteration_groups[id_][group_id][idx_within] = item
                group = iteration_groups[id_][group_id]
                if all(e is not None for e in group):
                    # print(f"All samples for id {id_} seen!")
                    completed_variation = iteration_groups[id_][group_id]
                    del iteration_groups[id_][group_id]
                    # cleanup empty id entry
                    if not any(any(e is not None for e in g) for g in iteration_groups[id_]): # all are None - delete the entire dict for the id to save memory
                        del iteration_groups[id_]

            # process completed variation(s) outside lock
            if completed_variation is not None:
                variations_to_process = [completed_variation]
                if mode=="extend": # do all variations
                    for shift in range(1, len(completed_variation)):
                        rotated = completed_variation[shift:] + completed_variation[:shift]
                        variations_to_process.append(rotated)
                # variation abspeichern wenn spread, um sie später zurück spreaden zu können
                elif mode == "spread":
                    variations_dict[(completed_variation[0]["id"], completed_variation[0].get("iter", 0))] = completed_variation


                # variations to process um dependencies erweitern, dann zu items_to process hinzufügen
                for variation in variations_to_process:
                    combined = variation[0].copy()
                    for key in dependencies:
                        combined_key = "sampled_" + key
                        combined[combined_key] = [it[key] for it in variation]
                    items_to_process.append(combined)



        # This needs changing
        for item in items_to_process:
            async for out in generator(item):
                if isinstance(out, EndOfData):
                    yield out
                else:
                    if mode == "first" or mode == "extend":
                        yield out
                    else:  # spread
                        completed_variation = variations_dict.pop((out["id"], out.get("iter", 0)))

                        # out must contain keys in spread_keys mapping to lists of length sample_size
                        # scalar outputs will be copied to every produced item
                        # create per-iteration items based on original variation
                        for i, item in enumerate(completed_variation):
                            if spread_keys == "all":  # all means its a leaf node
                                base = dict()
                                base["id"] = item["id"]
                                base["iter"] = item["iter"]
                            else:
                                base = item.copy()
                            keys = list((set(out.keys()) - {"id", "iter"}) if spread_keys == "all" else spread_keys)
                            for k in keys:
                                val_list = out.get(k)
                                if isinstance(val_list, list) and len(val_list) == len(completed_variation):
                                    base[k] = val_list[i]
                                else:
                                    # if not a list, set same value for all iterations
                                    base[k] = out.get(k)
                            # print(f"[sampling] yielding id={base['id']}, iter={base['iter']}")
                            yield base


    return wrapped
