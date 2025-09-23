from async_graph_bench.models.openai_api_model import OpenAIAPIModel
from async_graph_bench.models import GenerationParameters
import os
import asyncio
from tqdm import tqdm
import re
import argparse
import gc
import traceback
import os
import time
from collections import Counter
from dotenv import load_dotenv

from async_graph_bench import CSVDataStore, DataSource, NodeConfig, SamplingConfig, BenchmarkManager, Node, \
    visualize_graph, DiskCacheStore

for prefix in ['BLABLADOR']:
    print(f"{prefix}_BASE_URL=", os.environ.get(f"{prefix}_BASE_URL", None))  # They need to be set, throw otherwise
    api_key = os.environ.get(f"{prefix}_API_KEY", None)  # They need to be set, throw otherwise
    print(f"{prefix}_API_KEY=", (api_key[:3] + '*' * (len(api_key) - 3) if api_key else None))


class DummyDataSource(DataSource):
    stats = ['idx']

    def __len__(self):
        return 1

    async def iter_items(self):
        yield {"id": 0, "idx": 0}

    def iter_keys(self):
        yield 0


class ResponseGenerator:
    dependencies = ['idx']

    def __init__(self, prompt, stat):
        self.prompt = prompt
        self.stats = [stat]
        self.params = GenerationParameters(max_tokens=400)
        self.id = f"ResponseGenerator[{stat}]"

    async def __call__(self, item_stats, model):
        res = await model.query(self.prompt, self.params)
        return {
            self.stats[0]: res.get_messages()
        }


# class CommaSeparatedCounter:
#     dependencies = ['comma_separated']
#     stats = ['comma_separated_length']
#
#     def __call__(self, item_stats):
#         responses = item_stats["comma_separated"]
#         length = [len(r.split(",")) for r in responses]
#         return {
#             "comma_separated_length": length
#         }

class CommaSeparatedCounter:
    dependencies = ['comma_separated']
    stats = ['comma_separated_length']

    def __call__(self, item_stats):
        responses = item_stats["comma_separated"]
        lengths = []

        # Regex: find sequences like "12, 34, 56"
        pattern = re.compile(r"(?:\d+\s*,\s*)+\d+")

        for r in responses:
            match = pattern.search(r)
            if match:
                # Extract the sequence of numbers
                seq = match.group(0)
                # Split by comma and strip spaces
                nums = [n.strip() for n in seq.split(",")]
                lengths.append(len(nums))
            else:
                # No valid sequence found
                lengths.append(0)

        return {"comma_separated_length": lengths}


class NumberedListCounter:
    dependencies = ['numbered_list']
    stats = ['numbered_list_length']
    pattern = re.compile(r"\d+\. \d+")

    def __call__(self, item_stats):
        responses = item_stats["numbered_list"]
        length = [len(self.pattern.findall(r)) for r in responses]
        return {
            "numbered_list_length": length
        }


class LengthCounter:

    def __init__(self, dependency):
        self.dep = "sampled_" + dependency
        self.dependencies = [self.dep]
        self.id = f"LengthCounter[{self.dep}]"

    def __call__(self, item_stats):
        samples = item_stats[self.dep]
        return {
            'counts': [dict(Counter(s)) for s in samples]
        }


if __name__ == "__main__":
    print("Running Benchmark")
    data_source = DummyDataSource()

    model = OpenAIAPIModel(
        model_path="1 - Ministral 8b - the fast model",
        openai_endpoint=os.environ.get(f"BLABLADOR_BASE_URL"),
        openai_api_key=os.environ.get(f"BLABLADOR_API_KEY")
    )
    PROMPT_NUMBERED_1 = "Output 27 random integers between 1 and 100, comma-separated, without any additional text or introduction."
    PROMPT_NUMBERED_2 = """Produce exactly 27 lines. Each line must be:
<n>. <N>
where <n> is 1..27 in ascending order, a literal dot, one space, and <N> is a random integer 1–100.  
Output only those 27 lines and nothing else."""

    nodes = [
        NodeConfig(
            ResponseGenerator(
                prompt="Output 27 random integers between 1 and 100, comma-separated.",
                stat="comma_separated"),
            data_store=DiskCacheStore,
            resource_builder=lambda env: model,
            greedy=True,
            id="ResponseGenerator[comma_separated]"
        ),
        NodeConfig(
            ResponseGenerator(prompt="Output 27 random integers between 1 and 100 in a numbered list.",
                              stat="numbered_list"),
            data_store=DiskCacheStore,
            resource_builder=lambda env: model,
            greedy=True,
            id="ResponseGenerator[numbered_list]"
        ),

    ]

    consumer_nodes = [
        # in the consumer_nodes array, all NodeConfigs will receive greedy=True und data_store=CSVDataStore
        # you can also simply provide them in nodes with these parameters
        NodeConfig(CommaSeparatedCounter(), always_recompute=True),
        NodeConfig(NumberedListCounter(), always_recompute=True),
        NodeConfig(
            LengthCounter(dependency="numbered_list_length"),
            sampling_config=SamplingConfig(sampling_size=100),
            always_recompute=True
        ),
        NodeConfig(
            LengthCounter(dependency="comma_separated_length"),
            sampling_config=SamplingConfig(sampling_size=100),
            always_recompute=True
        )
    ]

    man = BenchmarkManager(
        iterations=100,
        iterations_first=True,
        data_source=data_source,
        nodes=nodes,
        consumer_nodes=consumer_nodes,
        data_storage_path=f"data",
        show_progress_bars=True,
        halt_on_exception=True
    )
    print("Created Manager")
    visualize_graph(man.base_adg, to_pdf=True)
    try:
        result = man.run_benchmark()
        print("Benchmarking finished!")
        exceptions = [item for sublist in result["exceptions"].values() for item in sublist]
        if exceptions:
            message = 'Exceptions happened:' + str(exceptions)
            print(message)
        else:
            print("Successfully finished")
    except Exception as e:
        traceback.print_exc()
    finally:
        gc.collect()
        # visualize_graph(
        #     man.runs[-1].adg,
        #     to_pdf=True,
        #     output_file="run_graph",
        #     show_descriptions=True,
        #     resolved_counts=man.runs[-1].resolved_at_start
        # )

# Give me 27 random integers between 1 and 100, comma separated.
# 27, 27, 28, 28, 29, 27, 28, 28, 29, 27, 27, 28, 27, 28, 28, 26, 28, 28, 28, 28, 28, 27, 27, 29, 30, 27, 29, 27, 30, 28, 28, 28, 29, 26, 28, 27, 29, 28, 27, 28, 51, 28, 27, 27, 29, 27, 30, 35, 27, 28, 28, 27, 27, 28, 28, 29, 29, 28, 29, 27, 28, 28, 29, 29, 27, 28, 26, 29, 28, 27, 26, 28, 27, 27, 27, 30, 28, 29, 28, 27, 28, 26, 29, 27, 28, 27, 28, 28, 27, 29, 28, 26, 28, 28, 29, 28, 26, 28, 29, 30
# Give me 27 random integers between 1 and 100 in a numbered list.
# 27, 27, 27, 27, 26, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 0, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 0, 27, 27, 27, 27, 27, 27, 26, 26, 26, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 0, 27, 27, 27, 26, 27, 27, 27, 27, 27, 27, 27, 0, 27, 27, 0, 27, 27, 27, 27, 27, 27, 27, 27, 27, 26, 27, 27, 27, 27, 27, 27, 27, 26, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27
