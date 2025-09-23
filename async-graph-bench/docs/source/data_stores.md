from async_graph_bench import CSVDataStorefrom async_graph_bench import CSVDataStore

# DataStores

* see base class [DataStore API](TODO)
* are responsible for caching and loading results for the intermediate nodes
* DataStores must be configured in the `NodeConfig` using the `data_store` variable - this should be a function that generates a datastore
* after the benchmark has been run the datastores may be used to evaluate the results
* helper functions to load the data are available
* different implementations are available currently
* you may implement your own implementation, that need to implement the `Serializer` protocol

## Implementations
### CSVDataStore
* will store items in a csv file, with the node id as filename
* best for quick overview

Example:
```python
# in Node config
NodeConfig(
    # ...
    data_store=CSVDataStore
)
# for evaluation
store = CSVDataStore("path/to/store", "STORE_ID")
```

### JSONDataStore
* similar to csv datastore
* will store items in a json file, with the node id as filename

### DiskCacheStore
* optimized for large amounts of data
* uses `diskcache` module internally
* Additionally, for optimal disk store usage, several Serializers are implemented for serialization and compression of data
* currently implemented:
  * binary format: `PickleSerializer`
  * compression: `MessagePackSerializer`, `ZLibCompressionSerializer`, `ZstdCompressionSerializer`
* by default, PickleSerializer and ZLibCompressionSerializer will be used
* custom serializers may be implemented and applied

Examples:
in Node config
```python
# 
NodeConfig(
    # ...
    data_store=DiskCacheStore
)
```
specifying custom serializers:
```python
from functools import partial

NodeConfig(
    # ...
    data_store=partial(DiskCacheStore, serializers=[
        ResponseCompressorSerializer(),  # custom serializer
        PickleSerializer(),
        ZLibCompressionSerializer()
    ]),
)
```
for later evaluation: (note: must use the same serializers as provided in the NodeConfig)
```python
store = DiskCacheStore(
    "path/to/store",
    "STORE_ID",
    serializers=[...] # optional
)
```