# Nodes

Nodes are the basis of this framework and responsible for calculation of intermediate and final steps.

You will find the following information:

- [Data Source](nodes/data_sources.md) - singular nodes in the benchmark that provide the initial data, most of the times datasets, item by item
- [Nodes](nodes/nodes.md) - on the implementation of nodes responsible for carrying out the calculation of intermediate and final results
- [Node Configuration](nodes/node_configuration.md) - information about all possible configuration options for the nodes as they are provided to the
  `BenchmarkManager`
- [Sampling](nodes/sampling.md) - remarks on how to implement sampling nodes, that calculate scores based on multiple iterations of the same data
