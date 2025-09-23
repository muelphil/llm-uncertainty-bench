import gc
import logging
import traceback
from typing import List, Dict

from tqdm import tqdm

from async_graph_bench import CSVDataStore, DataSource, NodeConfig, SamplingConfig, BenchmarkManager, Node, \
    visualize_graph


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


class DummyDataSource(DataSource):
    numbers = [(1, 2), (5, 3), (2, 4), (9, 1)]
    stats = ["first_number", "second_number"]

    def iter_items(self):
        for idx, row in enumerate(self.numbers):
            yield {
                "id": idx,
                "first_number": row[0],
                "second_number": row[1]
            }

    def iter_keys(self):
        return range(len(self.numbers))

    def __len__(self):
        return len(self.numbers)


import random


class DummyNoiseResource:
    def __init__(self, stddev: float = 1.0):
        self.rng = random.Random()
        self.stddev = stddev

    def generate_noise(self, length):
        # time.sleep(1 + (length // 50))  # TODO simulating load, resource can take up to 50 items concurrently
        return [self.rng.gauss(0.0, self.stddev) for _ in range(length)]


class NoiseAdder:
    description = "Adds noise to first number using DummyNoiseResource"
    dependencies = ["first_number"]
    stats = ["noisy_first_number"]

    async def __call__(self, item_stats: Dict[str, List], resource: DummyNoiseResource) -> Dict[str, List]:
        first_numbers = item_stats["first_number"]
        noise = resource.generate_noise(len(first_numbers))
        noisy_first_number = [x + n for x, n in zip(first_numbers, noise)]
        return {
            "noisy_first_number": noisy_first_number  # TODO
        }


class SquareCalculator:
    dependencies = ["second_number"]
    stats = ["second_number_squared"]

    def __call__(self, item_stats: Dict[str, list]) -> Dict[str, List]:
        second_numbers = item_stats["second_number"]
        second_number_squared = [pow(number, 2) for number in second_numbers]
        return {
            "second_number_squared": second_number_squared
        }


class AdditionCalculator:
    dependencies = ["noisy_first_number", "second_number_squared"]
    stats = ["addition"]

    async def __call__(self, item_stats: Dict[str, list]) -> Dict[str, List]:
        first_summand = item_stats["noisy_first_number"]
        second_summand = item_stats["second_number_squared"]
        addition = [a + b for a, b in zip(first_summand, second_summand)]
        return {
            "addition": addition
        }


import statistics


class DiffFromMeanSpread:
    dependencies = ["sampled_score"]
    spread = True  # <-- True

    async def __call__(self, dependencies: Dict[str, list]) -> Dict[str, List]:
        return {
            "diff_from_mean": [
                [abs(score - statistics.mean(sample)) for score in sample]
                # <-- returning list of scores, one for each item in the sample batch
                for sample
                in dependencies["sampled_score"]
            ]
        }


class MySpreadEstimator(Node):
    dependencies = ["sampled_addition"]
    spread = True

    async def __call__(self, dependencies: Dict[str, list]) -> List:
        sampled_addition = dependencies["sampled_addition"]
        diff_from_mean = [
            [x - (sum(inner) / len(inner)) for x in inner]
            for inner in sampled_addition
        ]
        return {"diff_from_mean": diff_from_mean}


NodeConfig.base_config = {"queue_size": 100, "prop_name": "estimations"}

if __name__ == "__main__":
    print("Running Benchmark")
    data_source = DummyDataSource()


    def build_noise_resource(env):
        if not hasattr(env, "noise_resource") or env.noise_resource is None:
            env.noise_resource = DummyNoiseResource()
        return env.noise_resource


    nodes = [
        NodeConfig(
            NoiseAdder(),
            data_store=CSVDataStore,
            resource_builder=build_noise_resource,
            greedy=True,
            batch_size=50,
        ),
        # in the nodes array, this automatically gets wrapper in a NodeConfig -> NodeConfig(SquareCalculator())
        SquareCalculator(),
        AdditionCalculator()
    ]

    consumer_nodes = [
        # in the consumer_nodes array, all NodeConfigs will receive greedy=True und data_store=CSVDataStore
        # you can also simply provide them in nodes with these parameters
        NodeConfig(
            VarianceEstimator(),
            always_recompute=True,
            sampling_config=SamplingConfig(sampling_size=5),
            batch_size=50
        ),
        NodeConfig(
            VarianceEstimator(),
            id="VarianceEstimator[extend]",
            always_recompute=True,
            sampling_config=SamplingConfig(sampling_size=5, all_variations=True),
            step=2,
            batch_size=50
        ),
        NodeConfig(
            MySpreadEstimator(),
            always_recompute=True,
            sampling_config=SamplingConfig(sampling_size=5),
            batch_size=50
        ),

    ]

    man = BenchmarkManager(
        iterations=5,
        iterations_first=True,
        data_source=data_source,
        nodes=nodes,
        consumer_nodes=consumer_nodes,
        data_storage_path=f"data",
        show_progress_bars=False,
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
        visualize_graph(
            man.runs[-1].adg,
            to_pdf=True,
            output_file="run_graph",
            show_descriptions=True,
            resolved_counts=man.runs[-1].resolved_at_start
        )
        gc.collect()
