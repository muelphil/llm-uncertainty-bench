import argparse
import gc
import logging
import os
import random
import time
from functools import partial

from async_graph_bench import NodeConfig, BenchmarkManager, DiskCacheStore, CSVDataStore, visualize_graph, \
    SamplingConfig, ResourcePool
from async_graph_bench.models.vllm_model import VLLMModel
from async_graph_bench.stores.serializers import PickleSerializer, ZLibCompressionSerializer
from tqdm import tqdm

from benchmark_datasets import DATASETS
from builders import build_model_from_config, build_encoders
from leaf_nodes import AnsweredCorrectlyArithmeticSimple, AnsweredCorrectlyMC, ConclusionProbabilityExtractor, \
    Verbalized2SUEExtractor, SciBenchAnswerFrequency
from models import MODELS, DEFAULTS
from nodes import MCQAAPriCoTResponseGenerator, ArithmeticResponseGenerator, OriginalPTrueApricot, \
    OriginalPTrueArithmetic, VerbalizedArithmetic, Verbalized2SApricot, \
    ClaimConditionedProbability, GreedyAlternativesSimpleNLICalculator
from nodes.response_compressor import ResponseCompressorSerializer


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

NodeConfig.base_config = {"queue_size": 100, "prop_name": "estimations"}

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
    args = parser.parse_args()
    print("args=", args)

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
    datasets = [d for d in DATASETS if dataset_matches(d)]

    for dataset in datasets:
        data_source = dataset["data_source"]()
        is_arithmetic = not dataset["is_multiple_choice"]

        for model in models:
            model_name = model["name"]
            result_path = f"data/{dataset['id']}/{os.path.basename(model_name)}"
            os.makedirs(result_path, exist_ok=True)

            print(f"Running Benchmark for {model_name} and {dataset['id']}")

            build_model = partial(build_model_from_config, m=model, gpu_type=args.gpu, default_kwargs=DEFAULTS)
            # clustering_model = m = next(m for m in MODELS if "meta-llama/Llama-3.3-70B-Instruct" in m["name"])
            clustering_model = m = next(m for m in MODELS if "mistralai/Mistral-Small-3.2-24B-Instruct-2506" in m["name"])
            # build_clustering_model = partial(build_model_from_config, m=clustering_model, gpu_type=args.gpu,
            #                                  default_kwargs=DEFAULTS, additional_kwargs={"max_model_len": 4096})


            def build_clustering_model(env):
                from vllm import LLM
                overrides = m.get("kwargs", dict())
                overrides_gpu = m.get(f"kwargs_{args.gpu}", dict())
                llm_args = {**DEFAULTS, **overrides, **overrides_gpu, "max_model_len": 4096}
                if not hasattr(env, "main_model_pool"):
                    llm = LLM(m.get("name"), **llm_args)
                    model = VLLMModel(llm, True, reasoning_parser_mode=m.get("reasoning_parser_mode"))
                    print("Successfully built single LLM instances!")
                    resource_pool = ResourcePool([model])
                    resource_pool.close = model.close
                    env.main_model_pool = resource_pool
                return [env.main_model_pool]


            verbalized_prompt = """Provide the probability that your guess is correct. Give ONLY the probability, no other words or explanation.\n\nFor example:\n\nProbability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>\n"""  # https://arxiv.org/pdf/2305.14975

            nodes = [
                NodeConfig(
                    (ArithmeticResponseGenerator if is_arithmetic else MCQAAPriCoTResponseGenerator)(max_tokens=4096,
                                                                                                     is_reasoning=model[
                                                                                                                      "type"] == "reasoning"),
                    data_store=partial(DiskCacheStore, serializers=[
                        ResponseCompressorSerializer(prefix="assistant"),
                        ResponseCompressorSerializer(prefix="reasoning"),
                        ResponseCompressorSerializer(prefix="conclusion"),
                        ResponseCompressorSerializer(prefix="conclusion_reasoning"),
                        PickleSerializer(),
                        ZLibCompressionSerializer()
                    ]),
                    resource_builder=build_model,
                    greedy=True,
                    batch_size=200,
                ),
                NodeConfig(
                    (VerbalizedArithmetic if is_arithmetic else Verbalized2SApricot)(
                        confidence_prompt=verbalized_prompt,
                        max_new_tokens=512 if model["type"] == "reasoning" else 15),
                    id="Verbalized2SAnswer",
                    data_store=DiskCacheStore,
                    resource_builder=build_model,
                    queue_size=200,
                    batch_size=200,
                ),
                NodeConfig(
                    (OriginalPTrueArithmetic if is_arithmetic else OriginalPTrueApricot)(
                        max_new_tokens=512 if model["type"] == "reasoning" else 15),
                    id="PTrueOriginalAnswers",
                    data_store=DiskCacheStore,
                    resource_builder=build_model,
                    queue_size=200,
                    batch_size=200,
                ),
                NodeConfig(
                    GreedyAlternativesSimpleNLICalculator(),
                    resource_builder=build_encoders,
                    data_store=DiskCacheStore,
                    step=2,
                    batch_size=50,
                    greedy=True
                ),
           ]
            if dataset["id"] == "SciBench":
                nodes.append(NodeConfig(
                    SciBenchAnswerFrequency(),
                    data_store=DiskCacheStore,
                    id="AnsweredCorrectly",
                    resource_builder=build_clustering_model,
                    sampling_config=SamplingConfig(sampling_size=10),
                    batch_size=50,
                    greedy=True,
                    step=3
                ))
                pass
            else:
                nodes.append(NodeConfig(
                    (AnsweredCorrectlyArithmeticSimple if is_arithmetic else AnsweredCorrectlyMC)(),
                    data_store=CSVDataStore,
                    greedy=True,
                    batch_size=200,
                    id="AnsweredCorrectly"
                ))

            consumer_nodes = [
                ClaimConditionedProbability(),
                NodeConfig(
                    ConclusionProbabilityExtractor(
                        dependency_name="p_true_original_alternatives",
                        valid_answers=["A", "B"],
                        alternative_equal_or_startswith_valid_answer="starts_with"
                    ),
                    batch_size=200,
                    id="PTrueOriginal"
                ),
                Verbalized2SUEExtractor()
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
                visualize_graph(man.base_adg, to_pdf=False, include_removed=True, output_file="cot_graph")

            start = time.time()
            result = man.run_benchmark()
            end = time.time()
            print("Benchmarking finished!")
            elapsed = end - start
            human_readable = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            end_h = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end))
            # write to file
            with open("timing.log", "a") as f:
                f.write(
                    f"[{end_h}] Benchmark for {dataset['id']} and {model['name']} took {human_readable} to finish\n")

            report = man.get_formatted_report()
            print(report)
            exceptions = [item for sublist in result["exceptions"].values() for item in sublist]
            print("Benchmarking finished!")
            if exceptions:
                print("Exceptions happened! Raising exceptions...")
                raise exceptions[0]
            else:
                print("Benchmarking finished!")
            del man
            gc.collect()
