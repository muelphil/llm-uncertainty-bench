# BenchmarkManager

The [`BenchmarkManager`](./api/benchmarkmanager.md) is the central orchestration component of the benchmarking framework. It coordinates the setup, validation, and execution of a full benchmark workflow consisting of multiple computational nodes, defined as [`NodeConfig`](./api/nodeconfig.md) objects. Each node encapsulates a specific calculation step with defined dependencies and outputs, forming a directed acyclic graph (ADG).

The manager is responsible for:

* Constructing and validating the dependency graph between nodes.
* Managing data storage paths and DataStores for caching results.
* Preparing and executing one or more **BenchmarkRuns** in the correct order.
* Handling exceptions and providing overall benchmark state and reporting utilities.

---

## Initialization

```python
BenchmarkManager(
    data_source: DataSource,
nodes: List[NodeConfig | Callable],
data_storage_path: str,
consumer_nodes: Optional[List[NodeConfig | Callable]] = None,
show_progress_bars: bool = True,
halt_on_exception: bool = True,
raise_exceptions: bool = True,
iterations: int = 1,
iterations_first: Optional[bool] = None,
)
```

### Parameters

* **`data_source`** ([`DataSource`](./api/datasource.md))
  Provides the input data items used throughout the benchmark. Must implement an iterator over identifiable data items.

* **`nodes`** (`List[NodeConfig | Callable]`)
  The list of computational nodes (or callables that will be wrapped as [`NodeConfig`](./api/nodeconfig.md)) forming the main benchmark graph.

* **`data_storage_path`** (`str`)
  Path to the root directory where all intermediate and final results will be cached.

* **`consumer_nodes`** (`Optional[List[NodeConfig | Callable]]`)
  Additional “greedy” nodes that consume results but are not dependencies for other nodes. These are often evaluation or reporting nodes. They are automatically wrapped in [`NodeConfig`](./api/nodeconfig.md) (if not already) with `greedy=True` and a default `CSVDataStore`.

* **`show_progress_bars`** (`bool`, default: `True`)
  Whether to display progress bars for each node during execution.

* **`halt_on_exception`** (`bool`, default: `True`)
  Whether to stop the benchmark immediately when an exception occurs in any node.

* **`raise_exceptions`** (`bool`, default: `True`)
  Whether caught exceptions should be raised. Will otherwise silently be stored in the `exceptions` property.

* **`iterations`** (`int`, default: `1`)
  Number of full benchmark iterations to run. Used together with sampling nodes to control repeated evaluation.

* **`iterations_first`** (`Optional[bool]`)
  Determines whether iteration order precedes sampling order. If not provided, it is inferred based on the presence of sampling nodes.

---

## Initialization Logic

During instantiation, the manager performs several key setup steps:

1. **Node Configuration** — All nodes and consumer nodes are converted to [`NodeConfig`](./api/nodeconfig.md) instances and validated (ensuring unique IDs and consistent sampling configurations).
2. **Graph Construction** — An `AcyclicDirectedGraph` (ADG) is created to represent dependencies between nodes. Unreachable or redundant nodes are pruned.
3. **Store Initialization** — For every greedy node, a [`DataStore`](./api/datastore.md) is initialized for caching. If the node is marked `always_recompute`, its store is cleared upfront.
4. **Step Determination** — Nodes may define an execution `step`. The manager determines the total number of steps to execute, ensuring no missing intermediate steps.
5. **Iteration Expansion** — Builds a complete mapping of all data item identifiers across all iterations, used to track which items are resolved or pending.

---

## Methods

### `run_benchmark()`

Executes the full benchmark workflow across all defined steps. For each step, it prepares a copy of the base ADG, deactivates nodes not relevant for that step, and launches a corresponding `BenchmarkRun`.

Execution proceeds in sequence, and results or exceptions are collected after each run. If any exception occur in one step, subsequent steps are aborted.

**Key behaviors:**

* Automatically prints progress bars and step headers when enabled.
* Collects and stores exceptions in `self.exceptions`.
* Ensures garbage collection between steps for memory efficiency.

---

### `_prepare_adg_for_step(step: int) -> AcyclicDirectedGraph`

Creates and returns a step-specific copy of the base ADG. Nodes that are not active in the current step (i.e. their `NodeConfig.step` is greater than the given step) are deactivated.

This method ensures that only the appropriate subset of nodes is executed at each stage of a multi-step benchmark.

---

### `get_node_state(node, adg) -> NodeState`

Returns the current state of a given node within the specified ADG. Possible states include:

* `"active"` — The node is scheduled for execution.
* `"pruned"` — The node was removed because it is not required.
* `"unreachable"` — The node cannot be reached from the data source.

---

### `get_state() -> BenchmarkState`

Evaluates and returns the global state of the benchmark as one of:

* `"pending"` — Benchmark not yet executed or still in progress.
* `"finished"` — All steps completed successfully.
* `"skipped"` — All steps were skipped (e.g. due to resolved caches).
* `"crashed"` — One or more steps failed with exceptions.

---

### `get_report() -> BenchmarkReport`

Generates a summarized report object for the completed benchmark. The report includes runtime statistics, node performance data, and exception summaries. This is the primary entry point for downstream evaluation and analysis of results. Refer to the `BenchmarkReport` documentation for details on available metrics and export options.

---

## Typical Usage Example

The usage example shows both ways of exception handling. When using `raise_exceptions=True`, exceptions will be raised immediately - use `try` - `except` then. When using `raise_exceptions=False`, exceptions will be collected and can be inspected after the run using `man.exceptions`. Opt for either way.

```python
man = BenchmarkManager(
    iterations=50,
    iterations_first=True,
    data_source=data_source,
    nodes=nodes,
    consumer_nodes=consumer_nodes,
    data_storage_path=f"data",
    show_progress_bars=True,
    halt_on_exception=True,
    raise_exceptions=False
)
visualize_graph(man.base_adg, to_pdf=True, output_file="base_graph", show_descriptions=True, to_console=False)
try:
    man.run_benchmark()
    report = man.get_report()
    print(report.to_table())
    report.write_csv_to_file("benchmarks.csv")

    if man.exceptions:
        print("Exceptions happened:")
        for exc in man.exceptions:
            details = []
            if exc.originator is not None:
                details.append(f"originated in {exc.originator}")
            if exc.step is not None:
                details.append(f"during benchmark step {exc.step}")

            suffix = f" ({', '.join(details)})" if details else ""
            print(f"* Exception {repr(exc.exception)}{suffix}")
    else:
        print("No Exceptions happened during run.")
except Exception as e:
    print("Execution halted due to exception: ", e)
    traceback.print_exc()
```

---

## Accessing DataStores

The manager provides a `store_per_node` property (`Dict[str, DataStore]`) that is populated after the benchmark completes. Each key corresponds to a node’s ID (by default, the class name) and maps to its associated [`DataStore`](./api/datastore.md). This allows immediate analysis of node outputs directly after execution. For later or external analysis (e.g., in Jupyter notebooks), the same [`DataStore`](./api/datastore.md) instances can be recreated by specifying the original result directory in the manager’s `data_storage_path` and the IDs of the individual nodes.

---

## Summary

The [`BenchmarkManager`](./api/benchmarkmanager.md) is the high-level coordinator that translates a declarative node configuration into an executable, stepwise benchmark pipeline. It ensures reproducibility, manages caching, handles sampling and iteration logic, and produces consistent results and reports across runs.
