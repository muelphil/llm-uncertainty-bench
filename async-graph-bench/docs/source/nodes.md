# Nodes

Nodes form the core of this framework and are responsible for computing both intermediate and final results in a benchmark.

The following sections provide detailed information:

## Overview of Node Concepts

- **[Data Source](nodes/data_sources.md)**  
  Nodes that provide the initial data, typically datasets, processed item by item.

- **[Nodes](nodes/nodes.md)**  
  Implementation of nodes responsible for performing the calculations for intermediate and final results.

- **[Node Configuration](nodes/node_configuration.md)**  
  Details on all configuration options available for nodes when used with the [`BenchmarkManager`](./api/benchmarkmanager.md).

- **[Sampling](nodes/sampling.md)**  
  Guidelines for implementing sampling nodes, which compute scores based on multiple iterations over the same data.
