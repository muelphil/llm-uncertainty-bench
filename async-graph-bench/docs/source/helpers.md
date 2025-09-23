# Helpers

here helpers are listed for additional functionality

## `visualize_graph`

see [`visualize_graph`](api/)

Examples:
visualizing base graph (this graph shows the planned execution; if nodes are fully resolved and not needed to be recalculated they will still show up in this graph, but not in the graph for the specific benchmark run)
```python
man = BenchmarkManager(
    # ...
)
visualize_graph(man.base_adg, to_pdf=True)
```
visualizing graph for specific runs 
```python
man = BenchmarkManager(
    # ...
)
try:
    result = man.run_benchmark()
except Exception as e:
    # exception handling
    pass
visualize_graph(
    man.runs[-1].adg,
    to_pdf=True,
    show_descriptions=True,
    resolved_counts=man.runs[-1].resolved_at_start
)
```

