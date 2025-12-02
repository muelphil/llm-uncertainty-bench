import argparse
import os

import torch
from dotenv import load_dotenv

from async_graph_bench import BenchmarkManager, NodeConfig
from async_graph_bench.stores import JSONDataStore
from async_graph_bench.utils.visualize_graph import visualize_graph
from builders import Endpoint, get_openai_api_builder, get_vllm_builder, get_vllm_multi_instance_builder
from prompt_datasource import PromptDataSource
from query_model import QueryModel

load_dotenv()

print("===== ENV =====")
for i in [1, 2]:
    prefix = f"OPENAI_API_ENDPOINT_{i}"
    base_url = os.environ.get(f"{prefix}_BASE_URL")
    api_key = os.environ.get(f"{prefix}_API_KEY")
    model = os.environ.get(f"{prefix}_MODEL")
    print(f"{prefix}_BASE_URL =", base_url)
    print(f"{prefix}_API_KEY =", (api_key[:3] + '*' * (len(api_key) - 3) if api_key else None))
    print(f"{prefix}_MODEL =", model)
print("\n\n")

NodeConfig.base_config = {"queue_size": 100, "prop_name": "estimations"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run a simple benchmark comparing LLM inference across different resource configurations. "
            "You can use online OpenAI-compatible endpoints or local vLLM instances."
        )
    )
    parser.add_argument(
        "--resources",
        choices=['endpoint-1', 'endpoint-2', 'both-endpoints', 'offline-vllm', 'offline-vllm-multi-instance'],
        required=True,
        help=(
            "Select which model resources to use:\n"
            "  endpoint-1 / endpoint-2   - Use a single OpenAI-style API endpoint defined in the .env file.\n"
            "  both-endpoints             - Run the benchmark across both configured endpoints.\n"
            "  offline-vllm               - Use a single local vLLM instance.\n"
            "  offline-vllm-multi-instance - Launch multiple vLLM instances (based on available GPUs)."
        )
    )
    parser.add_argument(
        "--model",
        default="mistralai/Ministral-8B-Instruct-2410",
        help="Model name or identifier to use for offline vLLM benchmarks (default: mistralai/Ministral-8B-Instruct-2410)."
    )
    parser.add_argument(
        "--llm-args",
        type=dict,
        default={"tokenizer_mode": "mistral", "tensor_parallel_size": 1},
        help=(
            "Dictionary of additional keyword arguments for vLLM initialization, e.g. "
            '{"tokenizer_mode": "mistral", "tensor_parallel_size": 1}. '
            "Ignored for online endpoints."
        )
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of prompts processed per batch during the benchmark (default: 50)."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=2,
        help="Number of iterations computed per item (10 items total) during the benchmark (default: 2)."
    )

    args = parser.parse_args()
    data_source = PromptDataSource()

    models = []
    resource_builder = None

    if args.resources.startswith(("endpoint", "both-endpoints")):
        ids = {"endpoint-1": [1], "endpoint-2": [2], "both-endpoints": [1, 2]}.get(args.resources, [1])
        endpoints = []
        for i in ids:
            model = os.environ.get(f"OPENAI_API_ENDPOINT_{i}_MODEL")
            models.append(model)
            endpoints.append(
                Endpoint(
                    base_url=os.environ.get(f"OPENAI_API_ENDPOINT_{i}_BASE_URL"),
                    api_key=os.environ.get(f"OPENAI_API_ENDPOINT_{i}_API_KEY"),
                    model=model,
                )
            )
        resource_builder = get_openai_api_builder(endpoints)

    elif args.resources == "offline-vllm":
        models.append(args.model)
        resource_builder = get_vllm_builder(
            args.model, args.llm_args, use_chat_template=True, reasoning_parser_model=None
        )

    else:  # offline vllm multi instance
        models.append(args.model)
        amount_models = torch.cuda.device_count() / args.llm_args["tensor_parallel_size"]
        models.append(f"x{amount_models}")
        resource_builder = get_vllm_multi_instance_builder(
            args.model, llm_args=args.llm_args, use_chat_template=True, reasoning_parser_mode=None
        )

    result_path = f"data/{args.resources}-{'--'.join(models)}-{args.batch_size}"
    os.makedirs(result_path, exist_ok=True)

    available_stat_calculators = [
        NodeConfig(
            QueryModel(max_tokens=1024),
            data_store=JSONDataStore,
            resource_builder=resource_builder,
            greedy=True,
            batch_size=args.batch_size,
            always_recompute=True,
        ),
    ]

    man = BenchmarkManager(
        iterations=args.iterations,  # Adjust as needed (e.g., 50 iterations * 10 prompts = 500 total queries per run)
        data_source=data_source,
        nodes=available_stat_calculators,
        data_storage_path=result_path,
        show_progress_bars=True,
        raise_exceptions=True,
    )

    if man.base_adg:
        visualize_graph(man.base_adg, format="svg")

    man.run_benchmark()

    store = man.store_per_node["QueryModel"]
    token_lengths = [item["token_lengths"] for item in store.iter_items()]

    report = man.get_report()
    print(report.to_table())

    node_delta = report.nodes["QueryModel"].delta  # Total amount of queries in this run
    time_per_query = report.total_time / node_delta

    report.write_csv_to_file(
        "benchmark.csv",
        extra_data={
            "Resources": args.resources,
            "Models": ",".join(models),
            "Batch Size": args.batch_size,
            "Time per Query (in s)": time_per_query,
            "Average Token Length": sum(token_lengths) / len(token_lengths),
        },
    )
