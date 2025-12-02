import argparse
import asyncio
import gc
import json
import logging
import os
import time
from functools import partial

from async_graph_bench import NodeConfig, BenchmarkManager, DiskCacheStore, CSVDataStore, visualize_graph, \
    SamplingConfig
from async_graph_bench.stores.serializers import PickleSerializer, ZLibCompressionSerializer

from benchmark_datasets import DATASETS
from builders import build_encoders_builder, build_model_from_config_builder, build_model_from_config
from leaf_nodes import AnsweredCorrectlyArithmeticSimple, AnsweredCorrectlyMC, ConclusionProbabilityExtractor, \
    SciBenchAnswerFrequency, Verbalized2SUEExtractor
from models import MODELS, DEFAULTS
from nodes import MCQAAPriCoTResponseGenerator, MCQAAPriCoTResponseGeneratorPost, ArithmeticResponseGeneratorPre, ArithmeticResponseGeneratorPost, \
    OriginalPTrueApricot, OriginalPTrueArithmetic, VerbalizedArithmetic, Verbalized2SApricot, \
    ClaimConditionedProbability, GreedyAlternativesSimpleNLICalculator
from nodes.response_compressor import ResponseCompressorSerializer

logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s [%(name)s] %(message)s",  # Include the logger name in brackets
    datefmt="%H:%M:%S",  # Time format in HH:MM:SS
)

NodeConfig.base_config = {"queue_size": 100, "prop_name": "estimations"}


def make_header(t):
    s = "\033[1;96m";
    e = "\033[0m"
    w = len(t) + 4
    return f"{s}╔{'═' * w}╗\n║  {t}  ║\n╚{'═' * w}╝{e}"


close_resource = None
resource_pool = None

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Filter models by query matching their name.")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of query strings to filter models by name (case-insensitive). If not provided, all models are used."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated list of query strings to filter datasets by name (case-insensitive). If not provided, all datasets are used."
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
        "--main-model-steps-only",
        action="store_true",
        default=False,
        help=(
            "If set, build the main model only once for all datasets and skip "
            "steps that require separate models."
        ),
    )
    args = parser.parse_args()

    # Prepare query list
    model_queries = [q.strip().lower() for q in args.models.split(",")] if args.models else None
    dataset_queries = [q.strip().lower() for q in args.datasets.split(",")] if args.datasets else None


    # Filter models
    def model_matches(model):
        name_match = any(q in model["name"].lower() for q in model_queries) if model_queries else True
        type_match = True if args.model_type is None else model["type"] == args.model_type
        return name_match and type_match


    # Filter datasets
    def dataset_matches(dataset):
        return any(q in dataset["id"].lower() for q in dataset_queries) if dataset_queries else True


    models = [m for m in MODELS if m["type"] != "base" and model_matches(m)]
    clustering_model = next(
        m for m in MODELS if "mistralai/Mistral-Small-3.2-24B-Instruct-2506" in m["name"])
    datasets = [d for d in DATASETS if dataset_matches(d)]

    for model in models:
        model_name = model["name"]

        if args.main_model_steps_only:  # share model across multiple benchmarks
            def build_model(env):
                global close_resource, resource_pool  # <-- correct
                if not resource_pool:
                    t0 = time.perf_counter()
                    overrides = model.get("kwargs", dict())
                    overrides_gpu = model.get(f"kwargs_{args.gpu}", dict())
                    llm_args = {**DEFAULTS, **overrides, **overrides_gpu}
                    print(f"Launching model with ", llm_args)
                    resource_pool, close_resource = asyncio.run(  # close model manually instead of framework doing it
                        build_model_from_config(model, llm_args)
                    )
                    startup_time = time.perf_counter() - t0
                    with open("resource_startup_times.ndjson", "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            'model': model['name'],
                            'startup_time': startup_time
                        }) + '\n')
                return [resource_pool]

        else:
            build_model = build_model_from_config_builder(model, args.gpu, DEFAULTS)

        build_clustering_model = build_model_from_config_builder(
            m=clustering_model, gpu_type=args.gpu,
            default_kwargs=DEFAULTS,
            additional_kwargs={"max_model_len": 4096})

        for dataset in datasets:
            data_source = dataset["data_source"]()
            is_arithmetic = not dataset["is_multiple_choice"]

            result_path = f"data/{dataset['id']}/{os.path.basename(model_name)}"
            os.makedirs(result_path, exist_ok=True)

            benchmark_terminal_header = make_header(f"Running Benchmark for {model_name} and {dataset['id']}")
            print(benchmark_terminal_header)

            verbalized_prompt = """Provide the probability that your guess is correct. Give ONLY the probability, no other words or explanation.\n\nFor example:\n\nProbability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>\n"""  # https://arxiv.org/pdf/2305.14975
            # verbalized_response_format = {"type": "regex", "regex": "^Probability: (1\\.0|0\\.\\d+)$"}

            encoders_builder = build_encoders_builder()

            nodes = [
                NodeConfig(
                    (ArithmeticResponseGeneratorPre if is_arithmetic else MCQAAPriCoTResponseGenerator)(
                        max_tokens=10240, is_reasoning=model["type"] == "reasoning",
                        system_prompt=model.get("system_prompt", None)),
                    data_store=partial(DiskCacheStore, serializers=[
                        ResponseCompressorSerializer(prefix="assistant"),
                        ResponseCompressorSerializer(prefix="reasoning"),
                        PickleSerializer(),
                        ZLibCompressionSerializer()
                    ]),
                    resource_builder=build_model,
                    greedy=True,
                    batch_size=100,
                ),
                NodeConfig(
                    (ArithmeticResponseGeneratorPost if is_arithmetic else MCQAAPriCoTResponseGeneratorPost)(is_reasoning=model["type"] == "reasoning"),
                    data_store=partial(DiskCacheStore, serializers=[
                        ResponseCompressorSerializer(prefix="conclusion"),
                        ResponseCompressorSerializer(prefix="conclusion_reasoning"),
                        PickleSerializer(),
                        ZLibCompressionSerializer()
                    ]),
                    resource_builder=build_model,
                    greedy=True,
                    batch_size=100,
                ),
                NodeConfig(
                    (VerbalizedArithmetic if is_arithmetic else Verbalized2SApricot)(
                        confidence_prompt=verbalized_prompt,
                        max_new_tokens=4096 if model["type"] == "reasoning" else 15,
                        # response_format=verbalized_response_format
                    ),
                    id="Verbalized2SAnswer",
                    data_store=DiskCacheStore,
                    resource_builder=build_model,
                    queue_size=200,
                    batch_size=200,
                ),
                NodeConfig(
                    (OriginalPTrueArithmetic if is_arithmetic else OriginalPTrueApricot)(
                        max_new_tokens=4096 if model["type"] == "reasoning" else 15),
                    id="PTrueOriginalAnswers",
                    data_store=DiskCacheStore,
                    resource_builder=build_model,
                    queue_size=200,
                    batch_size=200,
                ),
            ]

            if dataset["id"] == "SciBench" and not args.main_model_steps_only:
                nodes.append(NodeConfig(
                    SciBenchAnswerFrequency(10),
                    data_store=DiskCacheStore,
                    id="AnsweredCorrectly",
                    resource_builder=build_clustering_model,
                    sampling_config=SamplingConfig(sampling_size=10),
                    batch_size=50,
                    greedy=True,
                    step=3
                ))
            if not dataset["id"] == "SciBench":
                nodes.append(NodeConfig(
                    (AnsweredCorrectlyArithmeticSimple if is_arithmetic else AnsweredCorrectlyMC)(),
                    sampling_config=SamplingConfig(sampling_size=10),
                    data_store=CSVDataStore,
                    greedy=True,
                    batch_size=200,
                    id="AnsweredCorrectly"
                ))

            if not args.main_model_steps_only:
                nodes.append(
                    NodeConfig(
                        GreedyAlternativesSimpleNLICalculator(),
                        resource_builder=encoders_builder,
                        data_store=DiskCacheStore,
                        step=2,
                        batch_size=50,
                        greedy=True
                    ),
                )

            consumer_nodes = [
                ClaimConditionedProbability(),
                NodeConfig(
                    ConclusionProbabilityExtractor(
                        dependency_name="p_true_original_alternatives",
                        valid_answers=["A", "B"],
                        # Note: not (A) and (B) as the ptrue calculator will prepend '(' already
                        alternative_equal_or_anywhere_okay_valid_answer="anywhere"
                    ),
                    batch_size=200,
                    id="PTrueOriginal"
                ),
                Verbalized2SUEExtractor(),
            ]

            man = BenchmarkManager(
                iterations=10,
                iterations_first=True,
                data_source=data_source,
                nodes=nodes,
                consumer_nodes=consumer_nodes,
                data_storage_path=result_path,
                show_progress_bars=True,
                halt_on_exception=True
            )
            if man.base_adg:
                visualize_graph(man.base_adg, format="pdf", include_removed=True, output_file_name="cot_graph")

            try:
                man.run_benchmark()
            except Exception as e:
                print(f"Benchmark failed for model {model['name']} and dataset {dataset['id']}: {e}")
                # you can send a notification here; raising to interrupt the benchmark.
                continue
            report = man.get_report()
            print(report.to_table())
            if man.get_state() != "skipped":
                report.write_csv_to_file("benchmark_log.csv", extra_data={
                    "Model": model['name'],
                    "Dataset": dataset['id'],
                })
            del man
            gc.collect()

        if close_resource is not None:
            print("Closing main model resource pool...")
            close_resource()
            close_resource = None
            usage = resource_pool.get_usage_distribution()
            with open("resource_usage.txt", "a") as f:
                f.write(f"Main model {model['basename']} usage:\n{usage}\n")
            resource_pool = None
