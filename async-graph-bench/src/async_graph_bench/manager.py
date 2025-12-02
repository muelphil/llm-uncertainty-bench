import gc
import logging
import os
from typing import Callable, Union
from typing import Dict, Optional

from async_graph_bench.utils import ExceptionInfo
from .acyclic_directed_graph import AcyclicDirectedGraph
from .benchmark_run import BenchmarkRun
from .data_source import DataSource
from .node_config import NodeConfig
from .report import BenchmarkReport
from .stores import CSVDataStore
from .utils.helpers import *

log = logging.getLogger(__name__)

NodeState = Union["pruned", "unreachable", "active"]
BenchmarkState = Union["pending", "finished", "crashed", "skipped"]


class BenchmarkManager:
    """Orchestrates the setup, validation, and execution of benchmark workflows.

    The BenchmarkManager coordinates multiple computational nodes arranged
    in a directed acyclic graph (ADG). Each node represents a distinct
    calculation step with defined dependencies and outputs.

    Responsibilities include:
      * Building and validating the dependency graph between nodes.
      * Managing data stores and caching across runs.
      * Executing one or more BenchmarkRuns in correct step order.
      * Handling exceptions, reporting, and benchmark state tracking.
    """

    def __init__(
            self,
            data_source: DataSource,
            nodes: List[NodeConfig | Callable],
            data_storage_path: str,
            consumer_nodes: Optional[List[NodeConfig | Callable]] = None,
            show_progress_bars: bool = True,
            halt_on_exception: bool = True,
            raise_exceptions: bool = True,
            iterations: int = 1,
            iterations_first: Optional[bool] = None,
    ):
        """Initializes a BenchmarkManager instance.

        Args:
            data_source: The input data provider for the benchmark.
            nodes: List of computational nodes or callables forming the main graph.
            data_storage_path: Root path for caching intermediate and final results.
            consumer_nodes: Optional greedy or output nodes for evaluation.
            show_progress_bars: Whether to display progress bars during execution.
            halt_on_exception: If True, stops execution when an exception occurs.
            raise_exceptions: If True, raises exceptions instead of storing them.
            iterations: Number of iterations to execute per benchmark run.
            iterations_first: Whether iteration order precedes sampling order.
        """
        self.data_source: DataSource = data_source
        self.iterations: int = iterations
        self.iterations_first: bool = iterations_first
        self.halt_on_exception: bool = halt_on_exception
        self.raise_exceptions = raise_exceptions
        self.show_progress_bars: bool = show_progress_bars
        self.exceptions: List[ExceptionInfo] = []

        self.data_storage_path: str = os.path.abspath(data_storage_path)
        self.store_per_node: Dict[str, DataStore] = dict()
        self.base_adg: Optional[AcyclicDirectedGraph] = None
        self.node_configs: List[NodeConfig] = []
        self.runs: List[BenchmarkRun] = []
        # self.resources: Set = set()

        if not os.path.exists(self.data_storage_path):
            os.makedirs(self.data_storage_path)
            log.warning(f"Data Storage directory ({self.data_storage_path}) does not exist, creating it...")

        def ensure_consumer_config(c):
            if not isinstance(c, NodeConfig):
                c = NodeConfig(c, greedy=True, data_store=CSVDataStore)
            c.greedy = True
            c.data_store = c.data_store or CSVDataStore
            return c

        consumer_nodes = [ensure_consumer_config(c) for c in consumer_nodes] if consumer_nodes else []
        calculators = [
            calc if isinstance(calc, NodeConfig) else NodeConfig(calc)
            for calc in nodes
        ]

        self.node_configs = consumer_nodes + calculators

        # check duplicate ids
        check_unique_strings([config.id for config in self.node_configs])
        assert all(self.iterations % node.sampling_config.sampling_size == 0 for node in self.node_configs if
                   node.sampling_config is not None)
        for sampling_node in [node for node in self.node_configs if node.sampling_config is not None]:
            if sampling_node.always_recompute:
                clear_metadata(self.data_storage_path, sampling_node.id)
            metadata = get_metadata(self.data_storage_path, sampling_node.id)
            provided = {
                "sampling_size": sampling_node.sampling_config.sampling_size,
                "all_variations": sampling_node.sampling_config.all_variations,
                "sampling_mode": sampling_node.sampling_mode
            }
            for key in provided.keys():
                if key in metadata and metadata[key] != provided[key]:
                    raise Exception(
                        f"Mismatch in provided sampling config and sampling config stored in metadata file {self.data_storage_path}/{sampling_node.id}.metadata.json for node {sampling_node.id}. Please adjust the provided sampling config or delete the metadata and possible previously stored data in the corresponding datastore to prevent id mismatches. Key: {key}, Metadata: {metadata[key]}, provided SamplingConfig: {provided[key]}")
            update_metadata(self.data_storage_path, sampling_node.id, {
                "sampling_size": sampling_node.sampling_config.sampling_size,
                "all_variations": sampling_node.sampling_config.all_variations,
                "sampling_mode": sampling_node.sampling_mode
            })

        # build base ADG (this is the graph template we will copy for each run)
        self.base_adg = AcyclicDirectedGraph(self.data_source, self.node_configs)
        self.base_adg.build_graph()
        removed = self.base_adg.treeshake()
        if removed:
            log.info("Removed nodes during initial treeshaking: %s", ", ".join(r.id for r in removed))
        self.base_adg.remove_transient_edges()

        # create stores for greedy nodes and clear ones flagged as always_recompute once up-front
        for greedy_node_config in list(self.base_adg.greedy_nodes_configs):
            store = greedy_node_config.data_store(self.data_storage_path, greedy_node_config.id, create_okay=True)
            if greedy_node_config.always_recompute:
                store.clear()
            self.store_per_node[greedy_node_config.id] = store

        # Determine steps: default is single run (step 0). If any NodeConfig has `step` attribute > 0,
        # we will run steps 0..max_step inclusive.
        steps = set(cfg.step for cfg in self.node_configs)
        max_step = max(steps) if steps else 1
        missing = [s for s in range(2, max_step + 1) if
                   not any(getattr(cfg, "step", 1) == s for cfg in self.node_configs)]
        if missing:
            raise ValueError(f"Step configuration is not continuous — missing steps: {missing}")
        self.total_steps = max_step

        if self.iterations_first is None:
            self.iterations_first = any(node.is_sampling() for node in
                                        self.base_adg.optional_nodes_configs | self.base_adg.greedy_nodes_configs)

        duplicates = get_duplicates(self.data_source.iter_ids())
        if len(duplicates):
            raise AssertionError(f"Duplicate IDs in provided DataSource. Duplicate IDs found: {duplicates}")

        # Build combined ids for this run
        all_ids = build_combined_keys(
            self.data_source.iter_ids(),
            iterations=self.iterations,
            iterations_first=self.iterations_first
        )
        self.data_source_item_index: Dict[Tuple, int] = {  # TODO a hashlist may be more appropriate/ efficient
            item_id: index
            for index, item_id
            in enumerate(all_ids)
        }

    def _prepare_adg_for_step(self, step: int) -> AcyclicDirectedGraph:
        """Prepares a step-specific ADG by deactivating nodes not relevant to the step.

        Args:
            step: The benchmark step number to prepare.

        Returns:
            AcyclicDirectedGraph: A step-filtered copy of the base ADG containing only
            nodes active for the given step.
        """
        adg_copy = self.base_adg.copy()
        # Deactivate nodes that should not be active in this step
        for cfg in (adg_copy.optional_nodes_configs | adg_copy.greedy_nodes_configs):
            if cfg.step > step:
                adg_copy.remove_node(cfg)
        return adg_copy

    def run_benchmark(self):
        """Executes the full benchmark workflow across all configured steps.

        For each step, prepares a step-specific ADG, deactivates irrelevant nodes,
        and launches a corresponding BenchmarkRun. Results and exceptions are collected
        after each step. Subsequent steps are aborted if any exception occurs.
        """
        assert len(self.runs) == 0, "Cannot rerun same benchmark instance!"
        last_stores = self.store_per_node

        for step in range(1, self.total_steps + 1):
            log.info(f"Starting benchmark step {step}/{self.total_steps}")

            adg_for_step = self._prepare_adg_for_step(step)

            run = BenchmarkRun(
                adg=adg_for_step,
                step=step,
                data_source=self.data_source,
                iterations=self.iterations,
                iterations_first=self.iterations_first,
                data_source_item_index=self.data_source_item_index,
                store_per_node=self.store_per_node,
                data_storage_path=self.data_storage_path,
                show_progress_bars=self.show_progress_bars,
                halt_on_exception=self.halt_on_exception,
                raise_exceptions=self.raise_exceptions
            )
            run.init_graph()
            self.runs.append(run)

            try:
                if self.show_progress_bars or log.getEffectiveLevel() <= logging.INFO:
                    title = f" Running Benchmark Step {step}/{self.total_steps} "
                    bar = "═" * len(title)
                    print(f"\n╔{bar}╗\n║{title}║\n╚{bar}╝\n")
                run.run()
                for exception in run.exceptions:
                    self.exceptions.append(
                        ExceptionInfo(exception.exception, exception.originator, step=step)
                    )
                if self.exceptions:
                    print(f"Exception during benchmark run at step {step}; aborting subsequent steps.")
                    if self.raise_exceptions:
                        raise ExceptionGroup(
                            "Exception during benchmark run at step {step}; aborting subsequent steps.",
                            [exc.exception for exc in self.exceptions]
                        )
                    else:
                        break  # don't execute subsequent steps if this one resulted in an error
                else:
                    print(f"Finished benchmark run at step {step} - run state: {run.state}")
            finally:
                gc.collect()  # force garbage collection

    def get_node_state(self, node, adg) -> NodeState:
        """Determines the execution state of a node within a given ADG.

        Args:
            node: The NodeConfig or node ID to inspect.
            adg: The ADG instance containing the node.

        Returns:
            str: One of "active", "pruned", or "unreachable".
        """
        return "pruned" if node in adg.pruned_nodes else "unreachable" if node in adg.unreachable_nodes else "active"

    def get_state(self) -> BenchmarkState:
        """Returns the current overall benchmark state.

        Possible states:
          * "pending" — Not yet executed or in progress.
          * "finished" — All steps completed successfully.
          * "skipped" — All steps skipped due to fully resolved caches.
          * "crashed" — One or more steps failed with exceptions.

        Returns:
            str: The global benchmark state.
        """
        if any(run.state == "crashed" for run in self.runs):
            return "crashed"
        if len(self.runs) == self.total_steps:
            if all(run.state == "skipped" for run in self.runs):
                return "skipped"
            if all(run.state in ["finished", "skipped"] for run in self.runs) and any(
                    run.state == "finished" for run in self.runs):
                return "finished"
        return "pending"

    def get_report(self) -> BenchmarkReport:
        """Generates and returns a summary report for the completed benchmark.

        Includes runtime statistics, node-level performance data,
        and exception summaries for post-execution analysis.

        Returns:
            BenchmarkReport: A structured report object containing benchmark metrics.
        """
        return BenchmarkReport(self)
