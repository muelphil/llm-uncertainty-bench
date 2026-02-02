import argparse
import gc
import logging
import os

import torch
from async_graph_bench import DiskCacheStore, NodeConfig, BenchmarkManager, CSVDataStore, ResourcePool
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark_datasets import datasets
from models import MODELS
from nodes import LabelProbExtractor, QueryLevelUncertainty

NodeConfig.base_config = {"queue_size": 100, "prop_name": "estimations"}


def build_model_builder(model_name):
    def build_models(env):
        global close_resource, resource_pool  # <-- correct
        if not resource_pool:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                low_cpu_mem_usage=True,
                dtype=torch.float16,
                # device_map="cuda:0",
                device_map="auto",  # Let Transformers split the model across all GPUs
                attn_implementation="eager"
            )
            model.config.attention_impl = "eager"
            tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
            tokenizer.pad_token = tokenizer.eos_token
            model_pool = ResourcePool([model])
            tokenizer_pool = ResourcePool([tokenizer])

            def close_resource_func():
                del model
                del tokenizer
                gc.collect()
                # free GPU memory
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            resource_pool = [model_pool, tokenizer_pool]
            close_resource = close_resource_func
        return resource_pool

    return build_models


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
        "--model-type",
        type=str,
        default=None,
        choices=['base', 'instruct', 'reasoning', None],
        help="Type of model ('base', 'instruct', 'reasoning') - default: all models will be used."
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

    for model in models:
        builder = build_model_builder(model["name"])

        for dataset in filtered_datasets:
            data_source = dataset["data_source"]()

            benchmark_terminal_header = make_header(
                f"Running Benchmark for model {model['basename']} and dataset {dataset['id']}.")
            print(benchmark_terminal_header)

            result_path = f"data_query_level/{dataset['id']}/{model['basename']}"
            os.makedirs(result_path, exist_ok=True)

            nodes = [
                NodeConfig(
                    QueryLevelUncertainty(
                        target_tokens=model["yes_no_ids"]
                    ),
                    data_store=CSVDataStore,
                    resource_builder=builder,
                    greedy=True,
                    batch_size=50
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
            report.write_csv_to_file("data_query_level/benchmark_log.csv", extra_data={
                "Model": model['name'],
                "Dataset": dataset['id']
            })

            del man
            gc.collect()

        if close_resource is not None:
            print("Closing main model resource pool...")
            del resource_pool
            resource_pool = None
            close_resource()
            close_resource = None
