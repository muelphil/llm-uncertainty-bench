import os
import re
from collections import Counter

from dotenv import load_dotenv

from async_graph_bench import DataSource, NodeConfig, SamplingConfig, BenchmarkManager, visualize_graph, ResourcePool, \
    JSONDataStore, Model
from async_graph_bench.models import GenerationParameters
from async_graph_bench.models.openai_api_model import OpenAIAPIModel

# Load environment variables for OpenAI API
load_dotenv()
print("===== ENV =====")
print(f"OPENAI_API_ENDPOINT_BASE_URL=",
      os.environ.get(f"OPENAI_API_ENDPOINT_BASE_URL", None))  # Ensure this is set
api_key = os.environ.get(f"OPENAI_API_ENDPOINT_API_KEY", None)  # Ensure this is set
print(f"OPENAI_API_ENDPOINT_API_KEY=", (api_key[:3] + '*' * (len(api_key) - 3) if api_key else None))
print(f"OPENAI_API_ENDPOINT_MODEL=", os.environ.get(f"OPENAI_API_ENDPOINT_MODEL", None))
print("\n\n")

# Regex patterns to parse model responses
pattern_comma_separated = re.compile(r"(?:\d+\s*,\s*)+\d+")
pattern_numbered_list = re.compile(r"\d+\. \d+")


def extract_comma_separated(response: str):
    """Extracts the number of items from a comma-separated string of numbers."""
    match = pattern_comma_separated.search(response)
    if match:
        seq = match.group(0)
        nums = [n.strip() for n in seq.split(",")]
        return len(nums)
    return 0


def extract_from_numbered_list(response: str):
    """Extracts the number of items from a numbered list."""
    return len(pattern_numbered_list.findall(response))


# Prompts to instruct the model
PROMPT_COMMA_SEPARATED = "Output {number_items} random integers between 1 and 100, comma-separated, without any additional text or introduction."
PROMPT_NUMBERED_LIST = "Output {number_items} random integers between 1 and 100 in a numbered list."

amount_items = 28


class PromptSource(DataSource):
    """
    Provides two items with prompts to generate random integers:
    - Comma-separated integers
    - Numbered list of integers
    Each item also provides the corresponding extractor function.
    """
    provides = ["number_items", "prompt", "extractor"]
    items = [
        {
            "id": ("comma_separated", amount_items),
            "number_items": amount_items,
            "prompt": PROMPT_COMMA_SEPARATED.format(number_items=amount_items),
            "extractor": extract_comma_separated
        },
        {
            "id": ("numbered_list", amount_items),
            "number_items": amount_items,
            "prompt": PROMPT_NUMBERED_LIST.format(number_items=amount_items),
            "extractor": extract_from_numbered_list
        },
    ]

    def __len__(self):
        return len(self.items)

    def iter_ids(self):
        for item in self.items:
            yield item["id"]

    async def iter_items(self):
        for item in self.items:
            yield item


class ResponseGenerator:
    """
    Calls a language model to generate responses for a given prompt.
    Produces 'response' output for each input item.
    """
    requires = ["prompt"]
    provides = ["response"]
    description = "Generates a response for a prompt using an LLM"

    def __init__(self):
        self.params = GenerationParameters(max_tokens=400)

    async def __call__(self, item_stats, model: Model):
        # The model is provided by the framework via resource_builder
        res = await model.query(item_stats["prompt"], self.params)
        return {"response": res.get_messages()}


class LengthExtractor:
    """
    Extracts the number of items from the model's response using the provided extractor.
    Produces 'length' output.
    """
    requires = ["extractor", "response"]
    provides = ["length"]
    description = "Extracts the number of items from the model's response"

    def __call__(self, item_stats):
        responses = item_stats["response"]
        length = [
            extractor(response)
            for response, extractor
            in zip(item_stats["response"], item_stats["extractor"])
        ]
        return {"length": length}


class SampleLengthCounter:
    """
    Counts occurrences of each length across sampled runs and computes accuracy.
    Requires sampled lengths and expected number of items.
    Produces 'counts' and 'accuracy'.
    """
    requires = ["sampled_length", "number_items"]
    description = "Counts occurrences of each length across sampled runs and computes accuracy"

    def __call__(self, item_stats):
        samples = item_stats["sampled_length"]
        number_items = item_stats["number_items"]
        counts = [dict(Counter(s)) for s in samples]
        accuracy = [
            c.get(n, 0) / len(s) if len(s) > 0 else 0
            for c, s, n in zip(counts, samples, number_items)
        ]
        return {
            "counts": counts,
            "accuracy": accuracy
        }


if __name__ == "__main__":
    data_source = PromptSource()

    def resource_builder(env):
        if not hasattr(env, "main_pool"):
            model = OpenAIAPIModel(
                model_id=os.environ.get(f"OPENAI_API_ENDPOINT_MODEL"),
                openai_endpoint=os.environ.get(f"OPENAI_API_ENDPOINT_BASE_URL"),
                openai_api_key=os.environ.get(f"OPENAI_API_ENDPOINT_API_KEY")
            )
            env.main_pool = ResourcePool([model])
        return env.main_pool

    nodes = [
        NodeConfig(
            ResponseGenerator(),
            data_store=JSONDataStore,
            resource_builder=resource_builder,
            greedy=True,
        ),
    ]

    iterations = 10

    consumer_nodes = [
        NodeConfig(LengthExtractor()),
        # SampleLengthCounter samples multiple iterations to evaluate accuracy, recomputes every run
        NodeConfig(
            SampleLengthCounter(),
            always_recompute=True,
            sampling_config=SamplingConfig(sampling_size=iterations)
        )
    ]

    man = BenchmarkManager(
        iterations=iterations,
        iterations_first=True,
        data_source=data_source,
        nodes=nodes,
        consumer_nodes=consumer_nodes,
        data_storage_path="data",
        show_progress_bars=True,
        halt_on_exception=True,
        raise_exceptions=True
    )

    # Generate an SVG of the execution graph
    visualize_graph(man.base_adg, format="svg")

    # Run the benchmark
    man.run_benchmark()
    report = man.get_report()
    print(report.to_table())

    # Print accuracy results per prompt
    store = man.store_per_node["SampleLengthCounter"]
    for row in store.iter_items():
        print(
            f"Prompt with id {row['id']}\t provided requested number of random numbers with an accuracy of {row['accuracy'] * 100:.1f}% (Counts={row['counts']})"
        )
