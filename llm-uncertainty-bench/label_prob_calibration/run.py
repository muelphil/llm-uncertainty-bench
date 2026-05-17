import argparse
import asyncio
import gc
import logging
import os
from functools import partial

from async_graph_bench import DiskCacheStore, NodeConfig, BenchmarkManager, CSVDataStore

from benchmark_datasets import datasets
from builders import build_model_from_config
from data_sources import get_prompt_1, get_prompt_2, get_prompt_3, get_prompt_4
from util.models import MODELS, DEFAULTS
from nodes import LabelProbExtractor, MultipleChoiceLabelProbGenerator, MultipleChoiceLabelProbGeneratorMagistral

NodeConfig.base_config = {"queue_size": 100, "prop_name": "estimations"}


def make_header(t):
    s = "\033[1;96m";
    e = "\033[0m"
    w = len(t) + 4
    return f"{s}╔{'═' * w}╗\n║  {t}  ║\n╚{'═' * w}╝{e}"


logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s [%(name)s] %(message)s",  # Include the logger name in brackets
    datefmt="%H:%M:%S",  # Time format in HH:MM:SS
)

prompts_dict = {
    'prompt_1': get_prompt_1,
    'prompt_2': get_prompt_2,
    'prompt_3': get_prompt_3,
    'prompt_4': get_prompt_4,
}

close_resource = None
resource_pool = None

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Filter models by query and GPU requirements.")
    parser.add_argument(
        "--models",
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
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated list of query strings to filter datasets by name (case-insensitive). If not provided, all datasets are used."
    )
    args = parser.parse_args()
    print("args=", args)

    # Prepare query list
    queries = [q.strip().lower() for q in args.models.split(",")] if args.models else []

    dataset_queries = [q.strip().lower() for q in args.datasets.split(",")] if args.datasets else None


    # Filter datasets
    def dataset_matches(d):
        return any(q in d["id"].lower() for q in dataset_queries) if dataset_queries else True


    filtered_datasets = [d for d in datasets if dataset_matches(d)]


    # Filter models
    def model_matches(model):
        name_match = any(q in model["name"].lower() for q in queries) if queries else True
        type_match = args.model_type is None or model["type"] == args.model_type
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

        for model in models:
            def build_model(env):
                global close_resource, resource_pool  # <-- correct
                if not resource_pool:
                    overrides = model.get("kwargs", dict())
                    overrides_gpu = model.get(f"kwargs_{args.gpu}", dict())
                    llm_args = {**DEFAULTS, **overrides, **overrides_gpu}
                    resource_pool, close_resource = asyncio.run(  # close model manually instead of framework doing it
                        build_model_from_config(model, llm_args)
                    )
                return resource_pool


            for dataset in filtered_datasets:
                data_source = dataset["data_source"]()

                benchmark_terminal_header = make_header(
                    f"Running Benchmark for prompt {prompt}, model {model['basename']} and dataset {dataset['id']}.")
                print(benchmark_terminal_header)

                result_path = f"data_struct_output/data_{prompt}/{dataset['id']}/{model['basename']}"
                os.makedirs(result_path, exist_ok=True)

                nodes = [
                    NodeConfig(
                        MultipleChoiceLabelProbGenerator(
                            get_prompt=partial(get_prompt, shots=dataset["shots"]),
                            model_type=model["type"],
                            system_prompt=model.get("system_prompt", None)
                        ) if model['basename'] != "Magistral-Small-2507-Reasoning-Enabled"
                        else MultipleChoiceLabelProbGeneratorMagistral(
                            get_prompt=partial(get_prompt, shots=dataset["shots"]),
                            system_prompt=model.get("system_prompt", None),
                            max_tokens=4096
                        ),
                        data_store=DiskCacheStore,
                        resource_builder=build_model,
                        greedy=True,
                        batch_size=100
                    ),
                    NodeConfig(
                        LabelProbExtractor(),
                        greedy=True,
                        data_store=CSVDataStore
                    )
                ]

                man = BenchmarkManager(
                    iterations=1,
                    data_source=data_source,
                    nodes=nodes,
                    data_storage_path=result_path,
                    show_progress_bars=True
                )
                try:
                    man.run_benchmark()
                except Exception as e:
                    print(f"Benchmark failed for model {model['name']} and dataset {dataset['id']}: {e}")
                    # you can send a notification here; raising to interrupt the benchmark.
                    raise
                report = man.get_report()
                print(report.to_table())
                report.write_csv_to_file("benchmark_log.csv", extra_data={
                    "Model": model['name'],
                    "Dataset": dataset['id'],
                    "Prompt": prompt,
                })

                del man
                gc.collect()

            if close_resource is not None:
                print("Closing main model resource pool...")
                close_resource()
                usage = resource_pool.get_usage_distribution()
                with open("resource_usage.txt", "a") as f:
                    f.write(f"Main model {model['basename']} usage:\n{usage}\n")
                close_resource = None
                resource_pool = None
