import gc
import logging
import os
from typing import Any, List, Dict, Callable, Optional, Set

from async_graph_bench import acyclic_directed_graph

from .acyclic_directed_graph import AcyclicDirectedGraph
from .benchmark_run import BenchmarkRun
from .data_source import DataSource
from .node_config import NodeConfig
from .stores import DataStore, CSVDataStore
from .utils.helpers import check_unique_strings, get_metadata, update_metadata, clear_metadata
import time

log = logging.getLogger(__name__)


class BenchmarkManager:
    def __init__(
            self,
            data_source: DataSource,
            nodes: List[NodeConfig | Callable],
            data_storage_path: str,
            consumer_nodes: Optional[List[NodeConfig | Callable]] = None,
            show_progress_bars: bool = True,
            halt_on_exception: bool = True,
            iterations: int = 1,
            iterations_first: Optional[bool] = None,
    ):
        self.data_source: DataSource = data_source
        self.iterations: int = iterations
        self.iterations_first: bool = iterations_first
        self.halt_on_exception: bool = halt_on_exception
        self.show_progress_bars: bool = show_progress_bars

        self.data_storage_path: str = os.path.abspath(data_storage_path)
        self.store_per_node: Dict[str, DataStore] = dict()
        self.base_adg: Optional[AcyclicDirectedGraph] = None
        self.node_configs: List[NodeConfig] = []
        self.runs: List[BenchmarkRun] = []
        # self.resources: Set = set()

        if not hasattr(self.data_source, "id"):
            self.data_source.id = self.data_source.__class__.__name__

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

    def _prepare_adg_for_step(self, step: int) -> AcyclicDirectedGraph:
        """
        Returns a copy of base_adg for this step and deactivates nodes whose NodeConfig.step > step.
        """
        adg_copy = self.base_adg.copy()
        # Deactivate nodes that should not be active in this step
        for cfg in (adg_copy.optional_nodes_configs | adg_copy.greedy_nodes_configs):
            if cfg.step > step:
                adg_copy.remove_node(cfg)
        return adg_copy

    def run_benchmark(self) -> Dict[str, Any]:
        assert len(self.runs) == 0, "Cannot rerun same benchmark instance!"
        aggregated_exceptions = {}
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
                store_per_node=self.store_per_node,
                data_storage_path=self.data_storage_path,
                show_progress_bars=self.show_progress_bars,
                halt_on_exception=self.halt_on_exception,
            )
            run.init_graph()
            self.runs.append(run)

            try:
                result = run.run()
                # self.resources.update(run.resources)
            except Exception:  # TODO this should be put into the exceptions still and returned, not raised - raising it will interrupt and in run.py the report will never be printed
                log.exception(f"Exception during benchmark run at step {step}; aborting subsequent steps.")
                raise
            else:  # try block succeeded
                aggregated_exceptions.update(result.get("exceptions", {}))
                last_stores = result.get("stores", last_stores)

            gc.collect()  # force garbage collection

        return {"exceptions": {k: v for k, v in aggregated_exceptions.items() if v}, "stores": last_stores}

    def get_node_state(self, node, adg):
        return "pruned" if node in adg.pruned_nodes else "unreachable" if node in adg.unreachable_nodes else "active"

    def get_formatted_report(self):
        """
        Node 		Initial			Step 1 (finished, 1h:5min)			Step2(crashed, 1h:5min)		  End      Delta
        NodeName	0				54							        100				  		               100
        NodeName2	10				pruned						        100				   		                 90
        NodeName3	unreachable 	unreachable					        100				    	                0
        """
        report_data = self.get_report()
        total_time = time.strftime("%H:%M:%S", time.gmtime(report_data["total_time"]))
        end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(report_data["finish_time"])) if report_data[
                                                                                                         "finish_time"] is not None else None
        result = f"Benchmark {report_data['state']}{' ' + end_time if end_time is not None else ''} ({len(report_data['run_states'])}/{self.total_steps} steps initiated, total runtime: {total_time})\n"
        is_pending = report_data["state"] == "pending"
        table_header = [
            "Node",
            "Initial", *[
                f"Step {run_idx} ({state}, {time.strftime('%H:%M:%S', time.gmtime(run_time))})"
                for run_idx, (state, run_time) in
                enumerate(zip(report_data['run_states'], report_data['step_times']))
            ]
        ]
        if not is_pending:
            table_header += ["End", "Delta"]
        table = [table_header]
        for node_id, node_report_data in report_data["nodes"].items():
            row = [
                node_id,
                node_report_data["start_count"],
                *[(count if state == "active" else state)
                  for count, state in
                  zip(node_report_data["step_counts"], node_report_data["step_states"])],
            ]
            if not is_pending:
                row += [
                    node_report_data["end_count"],
                    node_report_data["end_count"] - node_report_data["start_count"]
                ]
            table.append(row)
        # Convert to strings
        table = [[str(c) for c in row] for row in table]
        # Compute widths
        widths = [max(map(len, col)) + 2 for col in zip(*table)]

        for row in table:
            result += ("".join(cell.ljust(w) for cell, w in zip(row, widths))) + "\n"
        return result

    def get_state(self):
        if any(run.state == "crashed" for run in self.runs):
            return "crashed"
        if len(self.runs) == self.total_steps:
            if all(run.state == "skipped" for run in self.runs):
                return "skipped"
            if all(run.state in ["finished", "skipped"] for run in self.runs) and any(
                    run.state == "finished" for run in self.runs):
                return "finished"
        return "pending"

    def get_report(self):
        end_stores = self.store_per_node
        # TODO this depends on base_adg being present -
        state = self.get_state()
        step_times = [run.elapsed if run.state in ["finished", "crashed"] else 0 for run in self.runs]
        report_data = {
            "state": state,  # TODO
            "run_states": [run.state for run in self.runs],
            "total_time": sum(step_times),
            "step_times": step_times,
            "finish_time": next((run.end_time for run in reversed(self.runs) if hasattr(run, "end_time")), None),
            "nodes": {},
        }
        all_nodes = (
            # {man.base_adg.data_source}|
                self.base_adg.optional_nodes_configs
                | self.base_adg.greedy_nodes_configs
                | self.base_adg.unreachable_nodes
                | self.base_adg.pruned_nodes
        )

        for node in all_nodes:
            base_state = self.get_node_state(node, self.base_adg)
            step_states = [
                self.get_node_state(node, run.adg)
                for run in self.runs
            ]

            start_count = next(
                (run.resolved_at_start[node.id][0] for run in self.runs if node.id in run.resolved_at_start),
                0) if len(self.runs) == self.total_steps else len(
                self.store_per_node[node.id]) if node.id in self.store_per_node else 0
            end_count = len(self.store_per_node[node.id]) if node.id in self.store_per_node else start_count

            step_counts = [
                run.resolved_at_start[node.id][0] if node.id in run.resolved_at_start else None
                for run in self.runs
            ]
            step_counts.append(end_count)

            report_data["nodes"][node.id] = {
                "base_state": base_state,
                "step_states": step_states,
                "start_count": start_count,
                "end_count": end_count,
                "step_counts": step_counts,
            }

        return report_data
