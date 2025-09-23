# DataSources

A **DataSource** is a special kind of node responsible for providing the input data to the benchmark. Each benchmark must have exactly one DataSource, as it defines the items that will flow through the computation graph.

Unlike regular nodes, a DataSource does not consume dependencies from other nodes. Instead, it produces the initial values that other nodes can depend on. To do this, it declares the set of values it provides in its `stats` attribute. Each emitted item must supply all of these values alongside a unique `id`.

## Implementing a DataSource

To implement a custom DataSource, inherit from the [`DataSource`](../api/datasource.md) base class. At a minimum, three methods must be defined:

* **`iter_items`**
  The most important method. It must be either a synchronous or asynchronous iterator that yields items one by one.
  Each item is a dictionary with two requirements:

  1. It must have a unique `"id"` identifying the item. This can be an `int`, `str`, or a tuple of such values.
  2. It must contain values for all keys listed in the `stats` attribute.

* **`iter_keys`**
  Returns an iterator over all item ids. This allows the framework to look up items or subsets efficiently.

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

    def iter_keys(self):
        return range(len(self.numbers))

    def __len__(self):
        return len(self.numbers)
```

In this version, the index serves as the item id. If the order of items changes, cached intermediate results may no longer match correctly. A more robust approach is to use a data driven id, in this example it can even be the data itself:

```python
yield {
    "id": row,
    "first_number": row[0],
    "second_number": row[1]
}
```

Here, even if the order of number pairs is shuffled, or new pairs are added, the cache will correctly match results to their items.