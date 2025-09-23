import argparse
import gc
import os
import time

from dotenv import load_dotenv

from async_graph_bench import BenchmarkManager
from async_graph_bench import NodeConfig
from async_graph_bench.models.openai_api_model import OpenAIAPIModel
from async_graph_bench.stores import JSONDataStore
from async_graph_bench.utils.visualize_graph import visualize_graph
from dummy_datasource import DummyDataSource
from query_model import QueryModel

load_dotenv()
for prefix in ['OPENAI', 'BLABLADOR', 'SCADS']:
    print(f"{prefix}_BASE_URL=", os.environ.get(f"{prefix}_BASE_URL", None))  # They need to be set, throw otherwise
    api_key = os.environ.get(f"{prefix}_API_KEY", None)  # They need to be set, throw otherwise
    print(f"{prefix}_API_KEY=", (api_key[:3] + '*' * (len(api_key) - 3) if api_key else None))

NodeConfig.base_config = {"queue_size": 100, "prop_name": "estimations"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select model type.")
    parser.add_argument("--models",
                        choices=['one', 'two', 'both'], required=True,
                        help="Choose between 'one', 'two', 'both'")

    args = parser.parse_args()

    # models = model_dict.get(
    #     args.models) if args.models != 'all' else llama_models + mistral24b_models + mistral7b_models

    data_source = DummyDataSource()

    result_path = f"data/{args.models}"
    os.makedirs(result_path, exist_ok=True)


    def build_resources(env):
        models = []
        if args.models == 'one' or args.models == 'both':
            models.append(OpenAIAPIModel(
                model_path="10 - DeepSeek-R1-Distill-Llama-8B - the best fast model as of January 2025",
                openai_endpoint=os.environ.get(f"BLABLADOR_BASE_URL"),
                openai_api_key=os.environ.get(f"BLABLADOR_API_KEY")
            ))
        if args.models == 'two' or args.models == 'both':
            models.append(OpenAIAPIModel(
                model_path="deepseek-ai/DeepSeek-R1",
                openai_endpoint=os.environ.get(f"SCADS_BASE_URL"),
                openai_api_key=os.environ.get(f"SCADS_API_KEY")
            ))
        return models


    available_stat_calculators = [
        NodeConfig(
            QueryModel(max_tokens=1024),
            data_store=JSONDataStore,
            resource_builder=build_resources,
            greedy=True,
            batch_size=1,
        ),
    ]

    man = BenchmarkManager(
        iterations=2,
        data_source=data_source,
        nodes=available_stat_calculators,
        # consumers=[LabelProbExtractor()],  # TODO
        verbose=False,
        data_storage_path=result_path,
        show_progress_bars=True
    )
    if man.adg:
        visualize_graph(man.adg, to_pdf=False)
    start = time.time()
    result = man()
    end = time.time()
    print(f"Execution time: {end - start:.6f} seconds")
    print("Benchmarking finished!")
    exceptions = [item for sublist in result["exceptions"].values() for item in sublist]
    if exceptions:
        message = 'Exceptions happened:' + str(exceptions)
        print(message)

    del man
    gc.collect()
