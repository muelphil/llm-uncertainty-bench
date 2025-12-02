 # DataStores

DataStores define how intermediate and final results of nodes are stored, cached, and later retrieved. They provide a unified interface for managing persistent data across benchmark runs. By handling serialization, caching, and retrieval, DataStores make it possible to re-run or evaluate benchmarks efficiently without recomputing existing results.

Each DataStore implementation follows a shared interface described in the [DataStore API](./data_stores.md). During execution, nodes write their results to a DataStore instance, which can later be used to load these results for evaluation or further analysis.

A DataStore is specified in a node’s configuration through the `data_store` attribute of [`NodeConfig`](./api/nodeconfig.md). This attribute should be a callable (for example, a class or factory function) that returns a configured DataStore instance. Once the benchmark has finished, the same store can be opened again to inspect or analyze the stored data.

While basic implementations are provided, users can also define their own DataStore by inheriting the [`DataStore`](./api/datastore.md) class. This makes it easy to tailor storage performance, compression, or format to specific workloads.

## InMemoryStore

The `InMemoryStore` class serves as an abstract base class for DataStores that keep all items in memory during computation. Once a node finishes, the data is serialized and written to disk. This approach is simple and efficient for smaller sets of data or when quick iteration is more important than storage optimization.
Subclasses of `InMemoryStore` must implement the actual serialization and deserialization of the data into a file.

### CSVDataStore

The `CSVDataStore` is a concrete implementation of `InMemoryStore`. It writes each node’s results to a `.csv` file, using the [`NodeConfig`](./api/nodeconfig.md)’s identifier as the filename. This makes it particularly convenient for quick inspection and debugging, as the results can be viewed directly in a spreadsheet editor.

**Example**

```python
# In Node configuration
NodeConfig(
    # ...
    data_store=CSVDataStore
)

# For later evaluation
store = CSVDataStore("path/to/store", "STORE_ID")
```

### JSONDataStore

The `JSONDataStore` is similar in spirit to the CSV variant but stores data in a `.json` file. It offers better compatibility with structured or nested data while maintaining human readability.

## DiskCacheStore

The `DiskCacheStore` is designed for handling large datasets efficiently, such as LLM responses. It uses the [`diskcache`](https://grantjenks.com/docs/diskcache/) library internally to cache results on disk, allowing computations to scale without exceeding memory limits. To optimize storage and performance, it supports a **serialization pipeline**, which defines how data is converted to bytes and optionally compressed before writing to disk.

Each serializer in the pipeline implements the [`Serializer`](./api/serializer.md) protocol:

```python
class Serializer(Protocol):
    def serialize(self, item: Any) -> bytes:
        pass

    def deserialize(self, data: bytes) -> Any:
        pass
```

A pipeline can contain multiple serializers, which are applied in order during serialization and reversed during deserialization. For example, a pipeline consisting of `[PickleSerializer(), ZLibCompressionSerializer()]` will first pickle an object and then compress the resulting bytes with zlib. When reading, zlib decompression happens first, followed by unpickling.

The following Serializers are implemented. By default, the `DiskCacheStore` uses `serializers=[PickleSerializer(), ZLibCompressionSerializer()]` as default binary serialization pipeline.

* **`PickleSerializer`** — Uses Python’s built-in `pickle` module to convert arbitrary Python objects to and from a byte representation. It supports almost any Python object.
* **`ZLibCompressionSerializer`** — Applies compression using the `zlib` library. It can significantly reduce storage size, especially for repetitive or structured data. The compression level can be configured via its `level` argument (default `-1` lets zlib choose an optimal balance between speed and compression ratio).

  ```python
  class ZLibCompressionSerializer:
      def __init__(self, level=-1):
          self.level = level

      def serialize(self, item: bytes) -> bytes:
          return zlib.compress(item, level=self.level)

      def deserialize(self, data: bytes) -> bytes:
          return zlib.decompress(data)
  ```
* **`MessagePackSerializer`** — Serializes data using the [MessagePack](https://msgpack.org/) format, a compact binary alternative to JSON. It is well-suited for structured data and offers faster encoding and decoding than pickle for many data types.
* **`ZstdCompressionSerializer`** — Compresses data using the [Zstandard](https://facebook.github.io/zstd/) algorithm. It generally achieves higher compression ratios and better speed than zlib, making it a good alternative for large-scale or high-performance workloads.

Custom serializers can be implemented by defining a class that adheres to the [`Serializer`](./api/serializer.md) protocol. This allows developers to integrate specialized serialization formats, domain-specific compression schemes, or even encrypted storage layers.

**Example (custom serializer pipeline)**

```python
from functools import partial

NodeConfig(
    # ...
    data_store=partial(DiskCacheStore, serializers=[
        ResponseCompressorSerializer(),  # custom serializer
        PickleSerializer(),
        ZLibCompressionSerializer(),
    ]),
)
```

When reopening a stored dataset after a benchmark run, the same serializer configuration must be supplied to ensure compatibility between serialization and deserialization:

```python
store = DiskCacheStore(
    "path/to/store",
    "STORE_ID",
    serializers=[...]  # must match NodeConfig
)
```

---

In summary, DataStores serve as a flexible layer for result persistence and caching. Lightweight options such as `CSVDataStore` and `JSONDataStore` are suitable for quick experiments and transparent storage, while `DiskCacheStore` provides robust handling for larger-scale or production-level benchmarks. Custom implementations allow full control over serialization and compression, ensuring that storage can adapt to different performance and data requirements.
