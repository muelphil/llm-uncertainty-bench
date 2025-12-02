import csv
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from .utils.helpers import get_resolved_keys, resolved_ids_to_bitarray


@dataclass
class NodeReport:
    node_id: str
    base_state: str
    step_states: List[str]
    start_count: int
    end_count: int
    step_counts: List[Optional[int]]
    delta: int


class BenchmarkReport:
    """
    Encapsulates the full benchmark report:
    - Builds data from a BenchmarkManager
    - Provides rendering methods: to_markdown_table(), to_json(), etc.
    """

    def __init__(self, manager: "BenchmarkManager"):
        self.manager = manager
        self.state: str = manager.get_state()
        self.total_steps: int = manager.total_steps
        self.run_states: List[str] = [run.state for run in manager.runs]
        self.step_times: List[float] = [
            run.elapsed if run.state in ["finished", "crashed"] else 0
            for run in manager.runs
        ]
        self.total_time: float = sum(self.step_times)
        self.finish_time: Optional[float] = next(
            (run.end_time for run in reversed(manager.runs) if hasattr(run, "end_time")), None
        )

        self.nodes: Dict[str, NodeReport] = self._collect_node_reports()

    def _collect_node_reports(self) -> Dict[str, NodeReport]:
        adg = self.manager.base_adg
        runs = self.manager.runs

        topological_ordered_nodes, _ = adg.get_nodes_and_edges_in_topological_order()
        topological_greedy_nodes = [n for n in topological_ordered_nodes if n in adg.greedy_nodes_configs]
        nodes = topological_greedy_nodes

        report_nodes: Dict[str, NodeReport] = {}

        for node in nodes:
            base_state = self.manager.get_node_state(node, adg)
            step_states = [self.manager.get_node_state(node, run.adg) for run in runs]

            # Determine start and end counts
            start_count = next(
                (run.resolved_at_start[node.id][0] for run in runs if node.id in run.resolved_at_start),
                0
            )

            store = self.manager.store_per_node.get(node.id, None)
            if store is not None:
                resolved_ids = get_resolved_keys(store=store, node_config=node, iteration_count=self.manager.iterations)
                ba = resolved_ids_to_bitarray(self.manager.data_source_item_index, resolved_ids)
                end_count = ba.count(1)
            else:
                end_count = 0

            step_counts = [run.resolved_at_start.get(node.id, ("",))[0] for run in runs] + [end_count]
            delta = end_count - start_count
            report_nodes[node.id] = NodeReport(
                node_id=node.id,
                base_state=base_state,
                step_states=step_states,
                start_count=start_count,
                end_count=end_count,
                step_counts=step_counts,
                delta=delta
            )

        return report_nodes

    def to_json(self) -> Dict[str, Any]:
        """Return the report as a JSON-serializable dict."""
        return {
            "state": self.state,
            "total_steps": self.total_steps,
            "run_states": self.run_states,
            "step_times": self.step_times,
            "total_time": self.total_time,
            "finish_time": self.finish_time,
            "nodes": {nid: asdict(nr) for nid, nr in self.nodes.items()},
        }

    def _build_table_data(self):
        """Collect header, columns, and rows for table rendering."""
        import time

        total_time = time.strftime("%H:%M:%S", time.gmtime(self.total_time))
        end_time_str = (
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.finish_time))
            if self.finish_time
            else "N/A"
        )

        header = (
            f"Benchmark {self.state} — {len(self.run_states)}/{self.total_steps} steps, "
            f"Total Runtime: {total_time}, Finished on {end_time_str}"
        )

        # Column headers
        columns = ["Node", "Initial"] + [
            f"Step {i + 1} ({state}, {time.strftime('%H:%M:%S', time.gmtime(t))})"
            for i, (state, t) in enumerate(zip(self.run_states, self.step_times))
        ]
        if self.state != "pending":
            columns += ["End", "Δ"]

        # Rows
        rows = []
        for node in self.nodes.values():
            row = [node.node_id, str(node.start_count)]
            row += [
                str(count) if state == "active" else state
                for count, state in zip(node.step_counts[1:], node.step_states)
            ]
            if self.state != "pending":
                delta = node.end_count - node.start_count
                row += [str(node.end_count), str(delta)]
            rows.append(row)

        return header, columns, rows

    def to_markdown_table(self) -> str:
        """Render the report as a markdown table."""
        header, columns, rows = self._build_table_data()

        md_header = (
                f"**{header}**\n\n"
                + " | ".join(columns)
                + "\n"
                + " | ".join(["---"] * len(columns))
                + "\n"
        )
        lines = [
            " | ".join(row)
            for row in rows
        ]
        return md_header + "\n".join(lines) + "\n"

    def to_table(self) -> str:
        """Render the report as a formatted UTF-8 table for terminal output."""
        header, columns, rows = self._build_table_data()

        # Compute column widths
        col_widths = [
            max(len(str(cell)) for cell in [col] + [r[i] for r in rows])
            for i, col in enumerate(columns)
        ]

        def fmt_row(row):
            return "│ " + " │ ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)) + " │"

        top = "┌" + "┬".join("─" * (w + 2) for w in col_widths) + "┐"
        sep = "├" + "┼".join("─" * (w + 2) for w in col_widths) + "┤"
        bottom = "└" + "┴".join("─" * (w + 2) for w in col_widths) + "┘"

        lines = [top, fmt_row(columns), sep] + [fmt_row(r) for r in rows] + [bottom]

        return f"{header}\n" + "\n".join(lines)

    def to_csv_row(self):
        """Return CSV-compatible header and single row of benchmark results."""
        # one column per node (each column named after the node_id)
        header = list(self.nodes.keys())

        # deltas for each node
        row = [
            str(node.end_count - node.start_count)
            for node in self.nodes.values()
        ]

        # additional columns for benchmark meta info
        header += ["State", "RunTime", "FinishTime"]

        total_time = time.strftime("%H:%M:%S", time.gmtime(self.total_time))
        finish_time_str = (
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.finish_time))
            if self.finish_time
            else "N/A"
        )

        row += [self.state, total_time, finish_time_str]

        return header, row

    def write_csv_to_file(self, path="report.csv", extra_data=dict()):
        """Append or merge benchmark data into a CSV file."""

        import pandas as pd

        new_header, new_row = self.to_csv_row()
        new_extra_keys = list(extra_data.keys())
        new_extra_vals = list(extra_data.values())

        # Insert new extra-data into header + row
        new_header_full = new_header + new_extra_keys
        new_row_full = new_row + new_extra_vals

        # CASE 1: File does not exist → write fresh
        if not os.path.exists(path):
            df = pd.DataFrame([new_row_full], columns=new_header_full)
            df.to_csv(path, index=False)
            return

        # CASE 2: File exists → load it
        existing_df = pd.read_csv(path)
        existing_header = list(existing_df.columns)

        # If headers match → normal append
        if existing_header == new_header_full:
            new_df = pd.DataFrame([new_row_full], columns=new_header_full)
            pd.concat([existing_df, new_df], ignore_index=True).to_csv(path, index=False)
            return

        # -------- HEADER MISMATCH → MERGE MODE --------

        # Parse existing header into sections
        def split_sections(header):
            if "State" not in header:
                raise ValueError("Existing CSV missing required 'State' column")

            idx_state = header.index("State")
            idx_finish = header.index("FinishTime")

            node_cols = header[:idx_state]
            meta_cols = header[idx_state:idx_finish + 1]  # ["State", "RunTime", "FinishTime"]
            extra_cols = header[idx_finish + 1:]

            return node_cols, meta_cols, extra_cols

        old_node_cols, old_meta_cols, old_extra_cols = split_sections(existing_header)
        new_node_cols, new_meta_cols, new_extra_cols = split_sections(new_header_full)

        # Sanity: meta columns always identical order by design
        # Merge node columns while keeping original ordering
        merged_node_cols = old_node_cols + [c for c in new_node_cols if c not in old_node_cols]

        # Merge extra columns while keeping original ordering
        merged_extra_cols = old_extra_cols + [c for c in new_extra_cols if c not in old_extra_cols]

        # Final unified column order
        final_columns = merged_node_cols + old_meta_cols + merged_extra_cols

        # Rebuild full dataframe
        # 1) Align existing data to final columns
        existing_df = existing_df.copy()
        for col in final_columns:
            if col not in existing_df.columns:
                existing_df[col] = ""

        existing_df = existing_df[final_columns]

        # 2) Create new single-row dataframe aligned to final columns
        new_series = {col: "" for col in final_columns}

        # Fill node & meta data
        for col, val in zip(new_header_full, new_row_full):
            new_series[col] = val

        new_df = pd.DataFrame([new_series], columns=final_columns)

        # 3) Combine and rewrite
        merged_df = pd.concat([existing_df, new_df], ignore_index=True)
        merged_df.to_csv(path, index=False)