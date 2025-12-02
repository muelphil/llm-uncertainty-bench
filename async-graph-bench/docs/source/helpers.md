# Helpers

Helper functions provide additional functionality for working with the benchmarking framework.

---

## [`visualize_graph`](./api/visualize_graph.md)

The [`visualize_graph`](./api/visualize_graph.md) function provides a way to visualize the **acyclic directed graph (ADG)** of node dependencies that define the execution flow of a benchmark.
It is implemented as a standalone helper to avoid requiring optional dependencies like `graphviz` for users who do not need visualization.

This function is **not implemented as part of** [`BenchmarkManager`](./api/benchmarkmanager.md) or `BenchmarkReport` to keep the `graphviz` dependency optional.
If you intend to use visualization, install Graphviz manually:

```bash
pip install graphviz
```

---

### Functionality

[`visualize_graph`](./api/visualize_graph.md) uses the `graphviz` library to generate a visual representation of the benchmark graph.
The output format determines whether the result is printed as text or rendered to a file.

Supported formats:

| Format    | Description                                        |
| :-------- | :------------------------------------------------- |
| `console` | Prints the Graphviz DOT source directly to stdout. |
| `pdf`     | Renders the graph as a PDF file.                   |
| `png`     | Renders the graph as a PNG image.                  |
| `svg`     | Renders the graph as an SVG image.                 |

You can take the printed DOT source and paste it into [GraphvizOnline](https://dreampuf.github.io/GraphvizOnline/) to view it interactively without installing additional software.

The visualization includes:

* **Nodes** representing benchmark components ([`NodeConfig`](./api/nodeconfig.md) instances)
* **Edges** representing data dependencies between nodes
* Optional **descriptions** of each node (if available)
* Optional **resolution statistics** (e.g., `3/7` to indicate resolved vs. total results)
* Distinct color-coding for **pruned** or **unreachable** nodes

---

### Example: Visualizing the Base Graph

The base graph shows the *planned* execution structure — all nodes and dependencies are shown, even if some are already resolved and skipped during execution.

```python
from your_package import BenchmarkManager, visualize_graph

man = BenchmarkManager(
    # ...
)
visualize_graph(man.base_adg, format="pdf")
```

This will generate an `execution_graph.pdf` file showing the complete dependency layout.

---

### Example: Visualizing a Specific Benchmark Run

You can visualize the graph of a specific run to inspect which nodes were recalculated, pruned, or skipped:

```python
man = BenchmarkManager(
    # ...
)
man.run_benchmark()

visualize_graph(
    man.runs[-1].adg,
    format="svg",
    output_file_name="graph", # without extension
    show_descriptions=True,
    resolved_counts=man.runs[-1].resolved_at_start
)
```

This will render an `execution_graph.svg` file, including node descriptions and resolution information.

---

### Example: Printing the Graph Source to Console

To inspect or debug the DOT source directly:

```python
visualize_graph(man.base_adg, format="console")
```

The DOT definition will be printed to the terminal.
You can copy it into [GraphvizOnline](https://dreampuf.github.io/GraphvizOnline/) to view the structure interactively.

---

### Notes

* [`visualize_graph`](./api/visualize_graph.md) requires the **Graphviz** Python package (`pip install graphviz`) for rendering.
* The output format controls whether rendering or console output is used.
* If an unsupported format is specified, a warning is logged and no output is generated.
