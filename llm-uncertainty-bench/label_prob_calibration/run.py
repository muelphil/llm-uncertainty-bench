import argparse
import gc
import os
from functools import partial
from importlib.metadata import version
from async_graph_bench import DiskCacheStore, NodeConfig, BenchmarkManager, visualize_graph, temporary_env, \
    CSVDataStore, ResourcePool
import torch

import logging
from tqdm import tqdm
import time
from models import MODELS, DEFAULTS
from benchmark_datasets import datasets
from data_sources import get_prompt_1, get_prompt_2, get_prompt_3, get_prompt_4
from nodes import LabelProbExtractor, MultipleChoiceLabelProbGenerator

NodeConfig.base_config = {"queue_size": 100, "prop_name": "estimations"}


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        # Format the record and write it using tqdm.write
        msg = self.format(record)
        tqdm.write(msg)


logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s [%(name)s] %(message)s",  # Include the logger name in brackets
    datefmt="%H:%M:%S",  # Time format in HH:MM:SS
    handlers=[TqdmLoggingHandler()]
)

prompts_dict = {
    'prompt_1': get_prompt_1,
    'prompt_2': get_prompt_2,
    'prompt_3': get_prompt_3,
    'prompt_4': get_prompt_4,
}

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Filter models by query and GPU requirements.")
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Comma-separated list of query strings to filter models by name (case-insensitive). If not provided, all models are used."
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="Comma-separated list of strings to filter prompts by indice (1,2,3,4). If not provided, all prompts are used."
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=['base', 'instruct', 'reasoning', None],
        help="Type of model ('base', 'instruct', 'reasoning') - default: all models will be used."
    )
    parser.add_argument(
        "--gpu",
        type=str,
        choices=["a100", "h100"],
        required=True,
        help="Specify the GPU type. Must be either 'a100' or 'h100'."
    )
    args = parser.parse_args()
    print("args=", args)

    # Prepare query list
    queries = [q.strip().lower() for q in args.query.split(",")] if args.query else []


    # Filter models
    def model_matches(model):
        name_match = any(q in model["name"].lower() for q in queries) if queries else True
        type_match = True if args.model_type is None else model["type"] == args.model_type
        return name_match and type_match


    models = [m for m in MODELS if model_matches(m)]

    print(f"Running Benchmark for models {[model['basename'] for model in models]}.")

    if args.prompts:
        # split "1,2,4" -> [1, 2, 4]
        indices = [int(x) for x in args.prompts.split(",")]
        prompts_dict = {
            f"prompt_{i}": prompts_dict[f"prompt_{i}"]
            for i in indices if f"prompt_{i}" in prompts_dict
        }
    print("Running Benchmark for prompts", list(prompts_dict.keys()))

    for prompt, get_prompt in prompts_dict.items():

        for dataset in datasets:  # TODO
            data_source = dataset["data_source"]()

            for model in models:
                if "vllm_version" in model:
                    assert model["vllm_version"] == version(
                        "vllm"), f"Incorrect vllm version ({version('vllm')}), expected: {model['vllm_version']}"

                print("=" * 100)
                print(f"Running Benchmark for prompt {prompt}, model {model['basename']} and dataset {dataset['id']}.")
                print("=" * 100)

                model_name = model["name"]
                use_chat_template = model["type"] != "base"

                result_path = f"data/data_{prompt}/{dataset['id']}/{model['basename']}"
                os.makedirs(result_path, exist_ok=True)
                overrides = model.get("kwargs", dict())
                overrides_gpu = model.get(f"kwargs_{args.gpu}", dict())
                llm_args = {**DEFAULTS, **overrides, **overrides_gpu}

                async def build_model(env):
                    #raise RuntimeError("This is a test!") # TODO
                    from async_graph_bench.models.multi_vllm_instances import start_workers, RemoteVLLMModel, WorkerClient
                    if not hasattr(env, "main_model_pool"):
                        print(f"GPUs detected for worker building: {torch.cuda.device_count()}")
                        # Note: this will throw the following error due to import/creation of cuda context by calling these functions - dont print it. Solution would be to share CUDA context somehow with subprocesses
                        # RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
                        # for i in range(torch.cuda.device_count()):
                        #     props = torch.cuda.get_device_properties(i)
                        #     free_mem = torch.cuda.mem_get_info(i)[0] / 1024 ** 3  # in GB
                        #     total_mem = props.total_memory / 1024 ** 3  # in GB
                        #     print(f"  GPU {i}: {props.name} | Free: {free_mem:.2f} GB / {total_mem:.2f} GB")
                        worker_clients, close = await start_workers(
                            model["name"],
                            llm_kwargs=llm_args,
                            gpus=list(range(torch.cuda.device_count())),
                            gpus_per_worker=llm_args["tensor_parallel_size"],
                        )
                        models = [RemoteVLLMModel(worker_client, True) for worker_client in worker_clients]
                        resource_pool = ResourcePool(models)
                        resource_pool.close = close
                        env.main_model_pool = resource_pool
                    return [env.main_model_pool]


                nodes = [
                    NodeConfig(
                        MultipleChoiceLabelProbGenerator(
                            get_prompt=partial(get_prompt, shots=dataset["shots"]),
                            model_type=model["type"],
                            end_of_reasoning_pattern=model.get("end_of_reasoning_pattern", None),
                            system_prompt=model.get("system_prompt", None)
                        ),
                        data_store=DiskCacheStore,
                        resource_builder=build_model,
                        greedy=True,
                        batch_size=150
                    ),
                    NodeConfig(
                        LabelProbExtractor(),
                        greedy=True,
                        data_store=CSVDataStore
                    )
                ]

                with temporary_env(getattr(model, "env", dict())):
                    man = BenchmarkManager(
                        iterations=1,
                        data_source=data_source,
                        nodes=nodes,
                        data_storage_path=result_path,
                        show_progress_bars=True
                    )
                    #if man.base_adg:
                    #    visualize_graph(man.base_adg, to_pdf=False)
                    start = time.time()
                    try:
                        result = man.run_benchmark()
                    except:
                        pass
                    end = time.time()
                    print("Benchmarking finished!")
                    print(man.get_formatted_report())
                    elapsed = end - start
                    human_readable = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                    end_h = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end))
                    # write to file
                    state = "  successful" if man.get_state() in ["skipped", "finished"] else "unsuccessful"
                    msg = f"[{end_h}] Benchmark {state} for {prompt}, {dataset['id']} and {model['basename']} took {human_readable} to finish - Run states={[run.state for run in man.runs]}\n"
                    with open("timing.log", "a") as f:
                        f.write(msg)
                    exceptions = [item for sublist in result["exceptions"].values() for item in sublist]
                    if exceptions:
                        message = 'Exceptions happened:' + str(exceptions)
                        print(message)

                    del man
                    gc.collect()
