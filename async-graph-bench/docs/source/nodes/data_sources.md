# DataSources

A [`DataSource`](../api/datasource.md) is a special kind of node responsible for providing the input data to the benchmark. Each benchmark must have exactly one [`DataSource`](../api/datasource.md), as it defines the items that will flow through the computation graph.

Unlike regular nodes, a DataSource does not consume dependencies from other nodes. Instead, it produces the initial values that other nodes can depend on. To do this, it declares the set of values it provides in its `stats` attribute. Each emitted item must supply all of these values alongside a unique `id`.

## Implementing a DataSource

To implement a custom DataSource, inherit from the [`DataSource`](../api/datasource.md) base class. At a minimum, three methods must be defined:

* **`iter_items`**
  The most important method. It must be either a synchronous or asynchronous iterator that yields items one by one.
  Each item is a dictionary with two requirements:

  1. It must have a unique `"id"` identifying the item. This can be an `int`, `str`, or a tuple of such values.
  2. It must contain values for all keys listed in the `stats` attribute.

* **`iter_ids`**
  Returns an iterator over all unique item ids. This allows the framework to build indices for the [`DataSource`](../api/datasource.md) without having to load the entire data on initialization.

* **`__len__`**
  Returns the total number of items available.

The easiest way to assign ids is to use the index of the item. However, if you expect the set or order of items to change over time (for example, when using different dataset subsets), it is recommended to use stable, data-derived ids. This ensures that cached intermediate results remain valid and reusable across runs.

## Example

A simple DataSource yielding pairs of numbers:

```python
class DummyDataSource(DataSource):
    numbers = [(1, 2), (5, 3), (2, 4), (9, 1)]
    stats = ["first_number", "second_number"]

    def iter_items(self):
        for idx, row in enumerate(self.numbers):
            yield {
                "id": idx, # <-- bad
                "first_number": row[0],
                "second_number": row[1]
            }

    def iter_ids(self):
        return range(len(self.numbers))

    def __len__(self):
        return len(self.numbers)
```

In this version, the index serves as the item id. If the order of items changes, cached intermediate results may no longer match correctly. A more robust approach is to use a data driven id, in this example it can even be the data itself:

```python
    def iter_items(self):
        for idx, row in enumerate(self.numbers):
            yield {
                "id": row, # <-- good!
                "first_number": row[0],
                "second_number": row[1]
            }

    def iter_ids(self):
        return self.numbers # <-- also needs adjustment

```

Here, even if the order of number pairs is shuffled, or new pairs are added, the cache will correctly match ids to their cached items.

> :warning: **Reserved dependency names**: Do only use the "id" dependency in [`DataSource`](../api/datasource.md)! Do not use the dependencies "iter" and "_idx", as they are reserved.