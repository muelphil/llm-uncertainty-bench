import asyncio
from typing import List, Dict
from datasets import load_dataset
from async_graph_bench import (
    DataSource,
    NodeConfig,
    BenchmarkManager,
    JSONDataStore,
    CSVDataStore,
    ResourcePool,
)
from async_graph_bench.models import GenerationParameters
from async_graph_bench.models.openai_api_model import OpenAIAPIModel


# ==========================
# Data Source
# ==========================

class MMLUDataSource(DataSource):
    """Loads MMLU dataset and provides question, choices, and correct index."""
    provides = ["question", "choices", "correct_choice_index"]

    def __init__(self, subset: str = "abstract_algebra", split: str = "test", limit: int | None = None):
        self.dataset = load_dataset("cais/mmlu", subset, split=split)
        if limit:
            self.dataset = self.dataset.select(range(limit))

    def __len__(self):
        return len(self.dataset)

    def iter_ids(self):
        for i in range(len(self.dataset)):
            yield i

    async def iter_items(self):
        for i, item in enumerate(self.dataset):
            yield {
                "id": i,
                "question": item["question"],
                "choices": item["choices"],
                "correct_choice_index": item["answer"],
            }


# ==========================
# Computation Nodes
# ==========================

class QueryAnswer:
    """Formats MMLU question into a multiple-choice prompt and queries model."""
    requires = ["question", "choices"]
    provides = ["response", "prompt"]

    def __init__(self):
        self.params = GenerationParameters(
            max_tokens=32,
            logprobs=5,  # retrieve logprobs for each token
            temperature=0.0
        )

    async def __call__(self, item_stats: Dict[str, List], model):
        prompts = []
        for q, choices in zip(item_stats["question"], item_stats["choices"]):
            formatted_choices = "\n".join(
                [f"{chr(65+i)}. {c}" for i, c in enumerate(choices)]
            )
            prompt = f"Answer the following multiple-choice question by selecting one option (A, B, C, or D):\n\n{q}\n\n{formatted_choices}\n\nAnswer:"
            prompts.append(prompt)

        responses = await model.query(prompts, self.params)
        messages = responses.get_assistant_messages()
        return {"response": messages, "prompt": prompts}


class Extractor:
    """Extracts per-choice log-probabilities and correctness information."""
    requires = ["response", "choices", "correct_choice_index"]
    provides = ["choice_logprobs", "predicted_index", "is_correct"]

    def __call__(self, item_stats: Dict[str, List]):
        import re
        responses = item_stats["response"]
        choices = item_stats["choices"]
        corrects = item_stats["correct_choice_index"]

        # Extract predicted label (A/B/C/D)
        predicted_labels = []
        for r in responses:
            match = re.search(r"\b([A-D])\b", r.strip())
            if match:
                predicted_labels.append(match.group(1))
            else:
                predicted_labels.append("A")  # fallback

        # Convert to index
        predicted_indices = [ord(label) - 65 for label in predicted_labels]
        is_correct = [int(pred == corr) for pred, corr in zip(predicted_indices, corrects)]

        # Placeholder probabilities (models exposing token logprobs can map here)
        choice_logprobs = [
            [0.25 for _ in c] for c in choices
        ]

        return {
            "choice_logprobs": choice_logprobs,
            "predicted_index": predicted_indices,
            "is_correct": is_correct,
        }


# ==========================
# Resource Builder
# ==========================

def resource_builder(env):
    if not hasattr(env, "model_pool"):
        model = OpenAIAPIModel(
            openai_endpoint="https://api.helmholtz-blablador.fz-juelich.de/v1",
            openai_api_key="glpat-F28KzK9GF9u8xZYN-sK6",
            model_id="Ministral-8B-Instruct-2410"
        )
        env.model_pool = ResourcePool([model])
    return env.model_pool


# ==========================
# Benchmark Execution
# ==========================

if __name__ == "__main__":
    data_source = MMLUDataSource(subset="abstract_algebra", limit=50)

    nodes = [
        NodeConfig(
            node=QueryAnswer(),
            data_store=JSONDataStore,
            resource_builder=resource_builder,
            greedy=True,
        ),
        NodeConfig(
            node=Extractor(),
            data_store=CSVDataStore,
            greedy=True,
        ),
    ]

    manager = BenchmarkManager(
        data_source=data_source,
        nodes=nodes,
        iterations=1,
        data_storage_path="data/mmlu_benchmark",
    )

    manager.run_benchmark()

    report = manager.get_report()
    print(report.to_table())

    # ==========================
    # Post-Benchmark: Accuracy
    # ==========================
    store = manager.store_per_node["Extractor"]
    items = list(store.iter_items())
    accuracies = [row["is_correct"] for row in items if "is_correct" in row]
    overall_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0

    print(f"\nMMLU Accuracy ({len(accuracies)} items): {overall_accuracy * 100:.2f}%")
