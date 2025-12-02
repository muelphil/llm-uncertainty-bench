# Nodes

Nodes are the fundamental building blocks of the framework. Each node encapsulates a calculation and defines which values it requires from other nodes and which values it provides. The framework uses this information to connect nodes into an execution graph and manage their execution.

Conceptually, a node is a small, self-contained calculator. It takes in specific values from the items moving through the graph, performs a computation, and outputs new values that can be consumed by downstream nodes. The [`Node`](../api/node.md) interface is defined as a protocol, ensuring a consistent contract for all implementations.

## Dependencies and Statistics

Dependencies use strings as identifiers. Every node must declare the dependencies it requires in order to run as a `requires` attribute of type `List[str]`. It will then receive these dependencies packed in a dictionary as a parameter. A node can also declare the dependencies it produces and provides to other nodes using the `provides` attribute. Provided dependencies are made available to other nodes further down the graph.

* If a node does not declare `provides`, it is treated as a **leaf node**: it still computes values, that can be stored for later analysis, but those values are not considered available as dependencies for other nodes.
* Both `requires` and `provides` can be declared statically at the class level or set dynamically in the node’s constructor.

## Calculation

The core logic of a node lives in its `__call__` method, which can be defined as synchronous or asynchronous. By default, a batch of items will be provided to the nodes, and nodes as a result must define their computations on batches. When the benchmark manager executes a node, it calls the `__call__` method with a dictionary containing the required dependencies.

This dictionary maps each dependency name to a list of values — one for each item in the batch currently being processed. This enables nodes to compute results for multiple items at once in a streamlined way. A common pattern is to use Python’s `for` and `zip` to iterate over corresponding values.

For example, if the following three items are batched together for the computation of a node:

```python
items = [
    {'id': 0, 'a': 0, 'b': 1},
    {'id': 1, 'a': 2, 'b': 7},
    {'id': 2, 'a': 6, 'b': 5},
]
```

the node will receive the following dictionary as the first argument to the `__call__` function:

```python
{
    'a': [0, 2, 6],
    'b': [1, 7, 5]
}
```

A common implementation pattern looks like this:

```python
def __call__(item_stats):
    sum_of_values = [
        a + b
        for a, b
        in zip(item_stats['a'], item_stats['b'])
    ]
    return {'sum': sum_of_values}
```

The `__call__` method must return a dictionary mapping each statistic name to a list of results, again one for each input item. If the node is a leaf node (i.e. no
provided dependencies were defined using `provides`), there are two alternatives:

1. The method can return a dictionary with custom keys and values. These values are stored in the benchmark by the defined data stores (see [Node Configuration](node_configuration.md)) but are not exposed as dependencies.
2. The method can return a plain list of values, one per item. The framework will automatically wrap this list in a dictionary as value to the key `'value'` (or a custom key provided via `NodeConfig.base_config["prop_name"]`).

It is important to note that the dependency data received in `__call__` as a parameter should never be modified.

**Example of a Node**

```python
class AdditionCalculator:
    requires = ["number_a", "number_b"]
    provides = ["sum_of_numbers"]

    def __call__(self, item_stats: Dict[str, list]) -> Dict[str, List]:
        sum_of_numbers = [
            a + b for a, b in zip(item_stats["number_a"], item_stats["number_b"])
        ]
        return {
            "sum_of_numbers": sum_of_numbers
        }
```

## Identifiers

When the framework builds the execution graph, each node is assigned an identifier. By default, the node’s class name is used. A custom identifier can be specified as the
`id` attribute of the class/ object or in the [`NodeConfig`](../api/nodeconfig.md) configuration. This is useful when multiple instances of the same node class with different parameters appear in the same graph.

## Descriptions

Nodes may optionally provide a human-readable description in the attribute
`description`. If present, this description can be included in graph visualizations (see [Graph Visualization](../helpers.md)), making the structure of the benchmark easier to inspect and understand.