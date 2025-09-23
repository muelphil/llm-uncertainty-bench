# Nodes

Nodes are the fundamental building blocks of the framework. Each node encapsulates a calculation and defines which values it requires from other nodes (its
**dependencies**) and which values it produces (its **statistics
**). The framework uses this information to connect nodes into an execution graph and manage their execution.

Conceptually, a node is a small, self-contained calculator. It takes in specific values from the items moving through the graph, performs a computation, and outputs new values that can be consumed by downstream nodes. The
`Node` interface is defined as a protocol, ensuring a consistent contract for all implementations.

## Dependencies and Statistics

Every node must declare the dependencies it requires in order to run as a `dependencies` attribute of type
`List[str]`. Dependencies are keys that must be present in the items it receives. A node can also declare the statistics it produces as a
`stat` attribute. Declared statistics are made available to other nodes further down the graph.

* If a node does not declare `stats`, it is treated as a **leaf node
  **: it still computes values, but those values are not considered dependencies for other nodes.
* Both `dependencies` and
  `stats` can be declared statically at the class level or set dynamically in the node’s constructor.

## Calculation

The core logic of a node lives in its
`__call__` method, which can be defined as synchronous or asynchronous. When the benchmark manager executes a node, it calls this method with a dictionary containing the required dependencies.

This dictionary maps each dependency name to a list of values, one for each item in the batch currently being processed. This enables nodes to compute results for multiple items at once in a streamlined way. A common pattern is to use Python’s
`for` and `zip` to iterate over corresponding values.

The
`__call__` method must return a dictionary mapping each statistic name to a list of results, again one for each input item. If the node is a leaf node (i.e. no
`stats` were defined), there are two alternatives:

1. The method can return a dictionary with custom statistics. These values are stored in the benchmark by the defined data stores (see [Node Configuration](node_configuration.md)) but are not exposed as dependencies.
2. The method can return a plain list of values, one per item. The framework will automatically wrap this list in a dictionary using the default key
   `value` (or a custom key provided via `NodeConfig.base_config["prop_name"]`).

It is important to note that the dependency data received in `__call__` as a parameter should never be modified.

**Example of a Node**

```python
class AdditionCalculator:
    dependencies = ["number_a", "number_b"]
    stats = ["sum_of_numbers"]

    def __call__(self, item_stats: Dict[str, list]) -> Dict[str, List]:
        sum_of_numbers = [
            a + b for a, b in zip(item_stats["number_a"], item_stats["number_b"])
        ]
        return {
            "sum_of_numbers": sum_of_numbers
        }
```

## Identifiers

When the framework builds the execution graph, each node is assigned an identifier. By default, the node’s class name is used. A custom identifier can be specified via the
`id` attribute or in the node’s configuration. This is useful when multiple instances of the same node class appear in the same graph.

## Descriptions

Nodes may optionally provide a human-readable description in the attribute
`description`. If present, this description can be included in graph visualizations (see [Graph Visualization](../helpers.md)), making the structure of the benchmark easier to inspect and understand.