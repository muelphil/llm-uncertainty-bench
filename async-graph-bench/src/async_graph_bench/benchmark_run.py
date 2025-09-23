import time
import asyncio
import gc
import inspect
import logging
import sys
from functools import reduce
from typing import List, Dict, Optional, Set

from async_graph_data_flow import AsyncExecutor, AsyncGraph
from tqdm import tqdm

from .acyclic_directed_graph import AcyclicDirectedGraph
from .data_source import DataSource
from .execution_nodes import data_cache, batching, skip_indices, skip_indices_data_source, \
    multi_incoming_node, with_resources, progress_wrapper, NodeExecutionWrapper, sampling, coordinated_end_of_data
from .execution_nodes.data_source_execution_wrapper import DataSourceExecutionWrapper
from .stores import DataStore
from .utils.builder_enviroment_stat_calculator import BuilderEnvironment

log = logging.getLogger(__name__)

from .utils.helpers import *


class BenchmarkRun:
    def __init__(
            self,
            adg: AcyclicDirectedGraph,
            data_source: DataSource,
            iterations: int,
            iterations_first: bool,
            store_per_node: Dict[str, DataStore],
            data_storage_path: str,
            show_progress_bars: bool,
            halt_on_exception: bool,
            step: int = 1,
    ):
        self.adg = adg
        self.step = step

        self.data_source = data_source
        self.iterations = iterations
        self.iterations_first = iterations_first
        self.store_per_node = store_per_node
        self.data_storage_path = data_storage_path
        self.resource_builder_env: BuilderEnvironment = BuilderEnvironment()
        self.show_progress_bars = show_progress_bars
        self.halt_on_exception = halt_on_exception
        self.resolved_at_start = dict()

        # per-run state
        self.data_source_item_index: Dict[tuple, int] = {}
        self.consumer_skip_indices: Dict[str, bitarray] = {}
        self.resources: Set = set()
        self._progress_bars: List[tqdm] = []
        self.data_source_idx_skip_table: Optional[bitarray] = None
        self.state = "pending"
        self.elapsed = 0

    def get_store_and_resolved_ids(self, node_config):
        store = self.store_per_node.get(node_config.id)
        if store is None:
            store = node_config.data_store(self.data_storage_path, node_config.id, create_okay=True)
            if node_config.always_recompute:
                store.clear()
            self.store_per_node[node_config.id] = store

        resolved_ids = set(store.iter_indices())
        if node_config.is_sampling() and node_config.sampling_mode == "first":
            resolved_ids = expand_resolved_ids(resolved_ids, self.iterations,
                                               node_config.sampling_config.sampling_size)
        return store, resolved_ids

    def init_graph(self):
        """
        Per-run initialization (previously 'init', but now for a single run/adg).
        Prepares data source item index and consumer skip indices according to stores.
        """
        log.info("=" * 100)
        log.info("Initializing benchmarking run (step=%s) ...", self.step)
        log.info("=" * 100)

        # Build combined ids for this run
        all_ids = build_combined_ids(
            self.data_source.iter_keys(),
            iterations=self.iterations,
            iterations_first=self.iterations_first
        )
        self.data_source_item_index = {item_id: index for index, item_id in enumerate(all_ids)}

        # For greedy nodes in this ADG (note: ADG may have been modified to exclude nodes)
        for greedy_node_config in list(self.adg.greedy_nodes_configs):
            store, resolved_ids = self.get_store_and_resolved_ids(greedy_node_config)
            ba = resolved_ids_to_bitarray(self.data_source_item_index, resolved_ids)
            # Note: this initial deviates from the initial variable in _build_execution_graph_nodes - here it represents the state of the individual node, in _build_execution_graph_nodes its represents fully resolved data items based on the leaf nodes
            initial = ba.count(1)
            total = len(self.data_source_item_index)
            self.resolved_at_start[greedy_node_config.id] = (initial, total)
            if ba.all():
                self.adg.turn_consumer_non_greedy(greedy_node_config)
                log.warning(
                    f"Node {greedy_node_config.id:<25} has fully resolved ({initial:>6} / {total:<7}) - setting it to non greedy")
            else:
                self.consumer_skip_indices[greedy_node_config.id] = ba
                log.info(
                    f"Node {greedy_node_config.id:<25} has resolved {initial:>6} / {len(self.data_source_item_index):<7}")

        # If there are no greedy nodes reachable in the adg, nothing to do for this run
        if not self.adg.greedy_nodes_configs:
            self.state = "skipped"
            log.warning("No greedy nodes defined or reachable for this run, or all greedy nodes are already resolved.")
            return

        # combine skip tables (items already computed by all consumers)
        if self.consumer_skip_indices:
            computed_ids = self.consumer_skip_indices.values()
            self.data_source_idx_skip_table = reduce(lambda x, y: x & y, computed_ids)
            log.warning(
                f"Fully resolved items upon start (step={self.step}): {self.data_source_idx_skip_table.count(1)} / {len(self.data_source_item_index)}"
            )

        removed = self.adg.treeshake()
        if removed:
            log.info("Removed nodes during treeshaking: %s", ", ".join(r.id for r in removed))

    def _build_execution_graph_data_source(self, execution_graph: AsyncGraph):
        generator = DataSourceExecutionWrapper(
            data_source=self.adg.data_source.iter_items,
            iterations=self.iterations,
            iterations_first=self.iterations_first
        ).execute

        if self.data_source_idx_skip_table:
            log.info(f"DataSource will skip {self.data_source_idx_skip_table.count(1)} items")
            generator = skip_indices_data_source(generator, self.data_source_idx_skip_table)

        execution_graph.add_node(generator, name=self.adg.data_source.id)

    def _build_execution_graph_nodes(self, execution_graph: AsyncGraph):
        consumers_by_calculator = self.adg.track_consumers_by_calculator()

        all_node_configs, _ = self.adg.get_nodes_and_edges_in_topological_order()

        for node_config in all_node_configs:

            args = {
                "name": node_config.id,
                "unpack_input": False,
                "max_tasks": 1
                # node_config.max_tasks or 1 TODO more than 1 currently not supported due to issues arising with EndOfData signal
            }
            if node_config.queue_size:
                args["queue_size"] = node_config.queue_size

            generator = NodeExecutionWrapper(node_config.node).execute

            store = None
            fully_resolved = False
            if node_config.data_store:
                store, resolved_ids = self.get_store_and_resolved_ids(node_config)
                fully_resolved = len(resolved_ids) >= len(self.data_source_item_index) and \
                                 is_fully_resolved(self.data_source_item_index, resolved_ids)

            if node_config.resource_builder and not fully_resolved:
                log.info("Building Resources for Node %s", node_config.id)
                if inspect.iscoroutinefunction(node_config.resource_builder):
                    resources = asyncio.run(node_config.resource_builder(env=self.resource_builder_env))
                else:
                    resources = node_config.resource_builder(env=self.resource_builder_env)
                log.info("Successfully built resources")
                # resources = node_config.resource_builder(env=self.resource_builder_env)
                args["max_tasks"] *= min([pool.total() for pool in resources])
                self.resources.update(
                    set(flatten_recursive(resources) if isinstance(resources, Iterable) else [resources]))
                generator = with_resources(generator, resources)

            if args["max_tasks"] > 1: # TODO BIG - put in place again after testing
                generator = coordinated_end_of_data(generator, node_config.id)

            if node_config.batch_size and node_config.batch_size > 1:
                generator = batching(generator, batch_size=node_config.batch_size)

            if node_config in self.adg.sampling:
                generator = sampling(
                    generator=generator,
                    dependencies=self.adg.sampling[node_config],
                    sample_size=node_config.sampling_config.sampling_size or self.iterations,
                    total_iterations=self.iterations,
                    mode=node_config.sampling_mode,
                    spread_keys=None if node_config.sampling_mode != "spread" else "all" if not hasattr(
                        node_config.node, "stats") else node_config.node.stats
                )

            if store is not None:
                generator = data_cache(
                    generator,
                    store=store,
                    properties=node_config.node.stats if hasattr(node_config.node, "stats") else "all"
                )

            skip_table = None
            initial = 0
            if self.adg.is_leaf_node(node_config):
                skip_table = self.consumer_skip_indices.get(node_config.id)
                initial = skip_table.count(1) if skip_table else 0
            else:
                consumer_skip_indices = [
                    self.consumer_skip_indices[consumer_config.id]
                    for consumer_config in consumers_by_calculator[node_config]
                    if consumer_config.id in self.consumer_skip_indices
                ]
                skip_table = reduce(lambda x, y: x & y, consumer_skip_indices) if consumer_skip_indices else None
                initial = skip_table.count(1) if skip_table else 0

            total = len(self.adg.data_source) * (
                self.iterations // node_config.sampling_config.sampling_size
                if node_config.is_sampling() and node_config.sampling_mode == "first"
                else self.iterations
            )

            if self.show_progress_bars:
                bar = tqdm(
                    initial=initial,
                    total=total,
                    desc=adjust_string_length(node_config.id, 25),
                    delay=1,
                    smoothing=0.0,
                    file=sys.stdout,
                    position=len(self._progress_bars),
                    # when using batching, a lot of items will come in in less than 1/10 of a second - this prohibits the bar from displaying the updates individually
                    mininterval=1.0
                )
                bar.disable = True
                self._progress_bars.append(bar)
                generator = progress_wrapper(generator, bar)

            if skip_table:
                if skip_table.any() and skip_table != self.data_source_idx_skip_table:
                    generator = skip_indices(generator, skip_table)


            if self.adg.count_parents(node_config) > 1:
                generator = multi_incoming_node(generator, self.adg.count_parents(node_config))

            execution_graph.add_node(generator, **args)

    def build_execution_graph(self) -> AsyncGraph:
        execution_graph = AsyncGraph(halt_on_exception=self.halt_on_exception)
        self._build_execution_graph_data_source(execution_graph)
        self._build_execution_graph_nodes(execution_graph)

        _, edges = self.adg.get_nodes_and_edges_in_topological_order()
        for (producer_config, consumer_config) in edges:
            execution_graph.add_edge(producer_config.id, consumer_config.id)

        return execution_graph

    def run(self) -> Dict[str, Any]:
        """
        Execute this run. Returns same shape dict as original UEManager: {exceptions, stores}
        """
        if self.state == "skipped":
            log.warning(
                "Skipping execution (run): nothing to calculate for this step - graph does not contain any greedy nodes.")
            return {"exceptions": {}, "stores": self.store_per_node}

        log.info(f"Building execution graph for step {self.step}")
        try:
            execution_graph = self.build_execution_graph()
            executor = AsyncExecutor(execution_graph)
        except Exception as e:
            self.state = "crashed"
            log.exception(f"Unexpected error during run initialization: {e}")
            raise
        log.info(f"Built execution graph for step {self.step}")
        try:
            for pb in self._progress_bars:
                pb.disable = False
                pb.refresh()
            self.start_time = time.time()
            executor.execute()
        except KeyboardInterrupt:
            log.warning("Received KeyboardInterrupt, cleaning up... please wait!")
            raise
        finally:
            self.state = "crashed" if any(len(excp) for excp in executor.exceptions.values()) else "finished"
            self.end_time = time.time()
            self.elapsed = self.end_time - self.start_time
            for resource in self.resources:  # TODO it would be nice if there is the option for the user to manually handle resource closing
                if hasattr(resource, 'close') and callable(resource.close):
                    asyncio.run(resource.close()) if inspect.iscoroutinefunction(resource.close) else resource.close()
            self.resources = set()
            # flush shared stores (manager will also flush at the end, but flush here after each run too)
            for store in self.store_per_node.values():
                store.flush()
            del execution_graph
            gc.collect()

        return {
            "exceptions": {k: v for k, v in executor.exceptions.items() if v},
            "stores": self.store_per_node
        }
