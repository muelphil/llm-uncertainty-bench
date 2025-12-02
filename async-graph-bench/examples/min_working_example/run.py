import gc
import gc
import logging
import random
import statistics
import time
import traceback
from typing import List, Dict

from async_graph_bench import CSVDataStore, DataSource, NodeConfig, SamplingConfig, BenchmarkManager, \
    ResourcePool, visualize_graph
from async_graph_bench.utils import BuilderEnvironment

log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s [%(name)s] %(message)s",  # Include the logger name in brackets
    datefmt="%H:%M:%S",  # Time format in HH:MM:SS
)


class NumberTupleDataSource(DataSource):
    numbers = [(1, 2), (5, 3), (2, 4), (9, 1)]
    provides = ["first_number", "second_number"]

    def iter_items(self):
        for idx, row in enumerate(self.numbers):
            yield {
                "id": idx,
                "first_number": row[0],
                "second_number": row[1]
            }

    def iter_ids(self):
        return range(len(self.numbers))

    def __len__(self):
        return len(self.numbers)


class DummyNoiseResource:
    def __init__(self, stddev: float = 1.0):
        self.rng = random.Random()
        self.stddev = stddev

    def generate_noise(self, length):
        time.sleep(1 + (length // 50))  # simulating load, resource can take up to 50 items concurrently
        return [self.rng.gauss(0.0, self.stddev) for _ in range(length)]


class NoiseAdder:
    description = "Creates first_number_noisy by adding noise to first number using DummyNoiseResource"
    requires = ["first_number"]
    provides = ["first_number_noisy"]

    async def __call__(self, item_stats: Dict[str, List], resource: DummyNoiseResource) -> Dict[str, List]:
        first_numbers = item_stats["first_number"]
        noise = resource.generate_noise(len(first_numbers))
        first_number_noisy = [x + n for x, n in zip(first_numbers, noise)]
        return {
            "first_number_noisy": first_number_noisy
        }


class SquareCalculator:
    requires = ["second_number"]
    provides = ["second_number_squared"]
    description = "Creates second_number_squared by squaring the second number"

    def __call__(self, item_stats: Dict[str, list]) -> Dict[str, List]:
        second_numbers = item_stats["second_number"]
        second_number_squared = [pow(number, 2) for number in second_numbers]
        return {
            "second_number_squared": second_number_squared
        }


class AdditionCalculator:
    requires = ["first_number_noisy", "second_number_squared"]
    provides = ["sum"]
    description = "Adds first_number_noisy to second_number_squared to create sum"

    async def __call__(self, item_stats: Dict[str, list]) -> Dict[str, List]:
        first_summand = item_stats["first_number_noisy"]
        second_summand = item_stats["second_number_squared"]
        sum = [a + b for a, b in zip(first_summand, second_summand)]
        return {
            "sum": sum
        }


# -------------------------- Sampling Nodes --------------------------

class VarianceEstimator:
    requires = ["sampled_sum"]
    description = "Samples sum to determine the variance introduced by the noise"

    async def __call__(self, item_stats, **kwargs):
        # raise ValueError("My funny error for testing purposes - in step 2")
        return {
            "variance": [
                statistics.variance(sample)
                for sample in item_stats["sampled_sum"]
            ]
        }


class DiffFromMeanEstimator:
    requires = ["sampled_sum"]
    description = "Samples sum to determine the difference per sum from the mean of sampled sums"

    async def __call__(self, item_stats: Dict[str, list], **kwargs) -> Dict[str, List]:
        # raise ValueError("My funny error for testing purposes - in step 1")
        return {
            "diff_from_mean": [
                abs(sample[0] - statistics.mean(sample))
                # sample is a list of additions, the first element belonging to the item, the rest to the sampled variations from other items
                for sample
                in item_stats["sampled_sum"]
            ]
        }


class DiffFromMeanSpreadEstimator:
    requires = ["sampled_sum"]
    spread = True  # <-- spreading result across items in the sampling batch
    description = "Samples sum to determine the difference per sum from the mean of sampled sums - using spread"

    async def __call__(self, item_stats: Dict[str, list], **kwargs) -> Dict[str, List]:
        return {
            "diff_from_mean": [
                [abs(addition - statistics.mean(sample)) for addition in sample]
                # instead of returning a single result for a single item, return all results as a list for all sampled item in the sampling batch
                # this effectively does the same as DiffFromMeanEstimator, but spreads the result across all sampled items, making it more efficient
                # in scenarios where many items can be processed as a batch to compute the desired dependency
                for sample
                in item_stats["sampled_sum"]
            ]
        }


NodeConfig.base_config = {"queue_size": 100, "prop_name": "estimations"}

if __name__ == "__main__":
    data_source = NumberTupleDataSource()


    def build_noise_resource(env: BuilderEnvironment):
        if not hasattr(env, "noise_resource") or env.noise_resource is None:
            env.noise_resource = DummyNoiseResource()
        return ResourcePool([env.noise_resource])


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

    iterations = 25

    consumer_nodes = [
        # in the consumer_nodes array, all NodeConfigs will receive greedy=True und data_store=CSVDataStore
        # you can also simply provide them in nodes with these parameters
        NodeConfig(
            VarianceEstimator(),
            always_recompute=True,
            sampling_config=SamplingConfig(sampling_size=iterations),
            step=2,
            batch_size=50
        ),
        NodeConfig(
            DiffFromMeanEstimator(),
            always_recompute=True,
            sampling_config=SamplingConfig(sampling_size=iterations, all_variations=True),
            batch_size=50
        ),
        NodeConfig(
            DiffFromMeanSpreadEstimator(),
            always_recompute=True,
            sampling_config=SamplingConfig(sampling_size=iterations),
            batch_size=50
        ),

    ]

    man = BenchmarkManager(
        iterations=iterations,
        iterations_first=True,
        data_source=data_source,
        nodes=nodes,
        consumer_nodes=consumer_nodes,
        data_storage_path=f"data",
        show_progress_bars=True,
        halt_on_exception=True,
        raise_exceptions=True
    )
    visualize_graph(man.base_adg, output_file_name="execution_graph", format="svg", show_descriptions=True)
    try:
        man.run_benchmark()
    except Exception as e:
        print("Execution halted due to exception: ", e)
        traceback.print_exc()
    finally:
        report = man.get_report()
        print(report.to_table())
        report.write_csv_to_file("report.csv", extra_data={"Batch Size": 50})
        gc.collect()
