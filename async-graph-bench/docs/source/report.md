# BenchmarkReport

The `BenchmarkReport` class provides a structured summary of a benchmark execution managed by a [`BenchmarkManager`](./api/benchmarkmanager.md). It collects per-node results, timing data, and global benchmark states, and can render them in multiple output formats such as JSON, Markdown, tables, or CSV.

A report is typically built automatically after running a benchmark and serves as the main entry point for inspection, exporting, or logging of benchmark outcomes.

---

## Overview

When initialized, `BenchmarkReport` extracts metadata and results from the given [`BenchmarkManager`](./api/benchmarkmanager.md):

* Global benchmark state (`finished`, `pending`, `crashed`, …)
* Per-step run states and execution times
* Per-node statistics (state changes, number of items resolved, deltas, etc.)
* Total runtime and finish time

The resulting object can be serialized or formatted using one of the provided rendering functions.

---

## `__init__(manager: BenchmarkManager)`

Initializes the report from a completed (or partially completed) [`BenchmarkManager`](./api/benchmarkmanager.md).

### Parameters

* **`manager`** ([`BenchmarkManager`](./api/benchmarkmanager.md)):
  The benchmark manager instance from which to build the report.

### Internal Behavior

* Reads overall and per-step states (`state`, `run_states`).
* Calculates elapsed time per step and the total runtime.
* Collects per-node reports via `_collect_node_reports()`.

---

## `to_json() -> Dict[str, Any]`

Returns a fully JSON-serializable representation of the benchmark report, suitable for logging or exporting.

**Example output structure:**

```
{
    "state": "finished",
    "total_steps": 2,
    "run_states": [
        "finished",
        "finished"
    ],
    "step_times": [ // in seconds
        8.062489986419678,
        0.036429643630981445
    ],
    "total_time": 8.09891963005066,
    "finish_time": 1759996152.3264494, // epoch timestamp
    "nodes": {
        "NoiseAdder": {
            "node_id": "NoiseAdder",
            "base_state": "active",
            "step_states": [
                "active",
                "active"
            ],
            "start_count": 0,
            "end_count": 200,
            "step_counts": [
                0, // start
                200, // after step 1
                200 // after step 2 == end
            ]
        },
        // other nodes omitted
    }
}
```

---

## `to_markdown_table() -> str`

Renders the benchmark summary as a Markdown table. This format is ideal for reports in documentation, notebooks, or web dashboards.

**Example output:**

```markdown
**Benchmark finished — 2/2 steps, Total Runtime: 00:00:08, Finished on 2025-10-09 09:49:12**

Node | Initial | Step 1 (finished, 00:00:08) | Step 2 (finished, 00:00:00) | End | Δ
--- | --- | --- | --- | --- | ---
NoiseAdder | 0 | 200 | 200 | 200 | 200
DiffFromMeanEstimator | 0 | 200 | pruned | 200 | 200
DiffFromMeanSpreadEstimator | 0 | 200 | pruned | 200 | 200
VarianceEstimator | 0 | pruned | 200 | 200 | 200
```

---

## `to_table() -> str`

Produces a formatted UTF-8 table for terminal output. The layout resembles `psql` or `rich`-style ASCII tables, making it suitable for console logs.

**Example output:**

```
Benchmark finished — 2/2 steps, Total Runtime: 00:00:08, Finished on 2025-10-09 09:49:12
┌─────────────────────────────┬─────────┬─────────────────────────────┬─────────────────────────────┬─────┬─────┐
│ Node                        │ Initial │ Step 1 (finished, 00:00:08) │ Step 2 (finished, 00:00:00) │ End │ Δ   │
├─────────────────────────────┼─────────┼─────────────────────────────┼─────────────────────────────┼─────┼─────┤
│ NoiseAdder                  │ 0       │ 200                         │ 200                         │ 200 │ 200 │
│ DiffFromMeanEstimator       │ 0       │ 200                         │ pruned                      │ 200 │ 200 │
│ DiffFromMeanSpreadEstimator │ 0       │ 200                         │ pruned                      │ 200 │ 200 │
│ VarianceEstimator           │ 0       │ pruned                      │ 200                         │ 200 │ 200 │
└─────────────────────────────┴─────────┴─────────────────────────────┴─────────────────────────────┴─────┴─────┘
```

---

## `to_csv_row() -> Tuple[List[str], List[str]]`

Returns a compact CSV-compatible representation of the benchmark results, with one column per node and additional metadata columns for the overall state, total runtime, and finish time.

**Example result:**

```python
(
    ['NoiseAdder', 'DiffFromMeanEstimator', 'DiffFromMeanSpreadEstimator', 'VarianceEstimator', 'State', 'RunTime', 'FinishTime'],  # header
    ['200', '200', '200', '200', 'finished', '00:00:08', '2025-10-09 09:49:12']  # row
)
```

---

## `write_csv_to_file(path="benchmarks.csv", extra_data=dict())`

Appends the benchmark’s results to a CSV file, optionally including extra metadata columns.

### Parameters

* **`path`** (`str`, default `"benchmarks.csv"`):
  Path to the target CSV file. The header is written if the file does not exist.
* **`extra_data`** (`dict`, optional):
  Key–value pairs to add as extra columns to the CSV output (e.g., environment info, version tags).

### Behavior

* Writes one row per benchmark run.
* Ensures header consistency across writes. Raises a `ValueError` if the existing header does not match the new one.

**Example**

```python
report.write_csv_to_file(
    path="benchmarks.csv",
    extra_data={"Commit": "a91f4b7", "Hardware": "Raspberry Pi 5"}
)
```

### Example `benchmarks.csv`

| NoiseAdder | DiffFromMeanEstimator | DiffFromMeanSpreadEstimator | VarianceEstimator | State    | RunTime  | FinishTime          | Batch Size | Commit  | Hardware       |
|------------|-----------------------|-----------------------------|-------------------|----------|----------|---------------------|------------|---------|----------------|
| 180        | 190                   | 195                         | 210               | finished | 00:00:07 | 2025-10-09 09:49:12 | 50         | b3c8d2e | MacBook Air M2 |
| 200        | 200                   | 200                         | 200               | finished | 00:00:08 | 2025-10-10 14:22:45 | 50         | a91f4b7 | Raspberry Pi 5 |

---

## Summary

The `BenchmarkReport` class provides a flexible and structured way to extract and visualize benchmark results:

* Use `to_json()` for programmatic logging.
* Use `to_table()` or `to_markdown_table()` for human-readable summaries.
* Use `write_csv_to_file()` for long-term tracking or automated aggregation.
