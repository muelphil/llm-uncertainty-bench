# Resources

* resources may be used for the calculation of statistics within nodes
* resources are provided as part of the node configuration detailed in [Node Configuration](nodes/node_configuration.md)
* the resource_builder must be a function that provides a resource
* resource_builder functions get provided an environment so that a resource may be shared across multiple nodes.
* multiple resources required for a single execution of a node may be specified
* additionally, if multiple sets of these resources required by the nodes are available (for example 2 api endpoints), multiple resource sets may be provided, leading to multiple simultaneous runs of the node with the resources. For example, if 2 async endpoints are available and provided to the node, 2 instances of the node will be ran asynchronously using the resources when they are available, leading to a big improvement in benchmarking time.

Abstract Examples:

```python
class MyAPIResource:
    def __init__(self, url: str):
        self.url = url

    async def get_result(self, query):
        # use async fetch for url to get result
        return result
```

Having multiple endpoints 
```python
def build_resources(env):
    if hasattr(env, "my_resources"):
        return getattr(env, "my_resources")
    resources = [
        MyAPIResource("http://api.endpoint1.com"),
        MyAPIResource("http://api.endpoint2.com")
    ]
    return resources
```
Example of Node
```python
class MyNode:
    dependencies = ["query"]
    stats = ["result"]

    async def __call__(self, item_stats: Dict[str, list], resource: MyAPIResource) -> Dict[str, List]:
        return {
            "result": [
                await resource.get_result(number)
                for number in item_stats["number"]
            ]
        }
```
Configuration of Node
```python
NodeConfig(
    MyNode(),
    resource_builder=build_resources
)
```

This will result in 2 instances of MyNode running, getting provided the 2 `MyAPIResource` resources as they are available as parameters to `__call__`

### If a node requires more than 1 resource
Example:
```python
def build_resources(env):
    if hasattr(env, "resource_set"):
        return getattr(env, "resource_set")
    resource_set = [
        (ResourceA(), ResourceB()) # <-- put resource sets in in tuple to provide multiple resources to a single node
    ]
    return resources
```
```python
class MyMultiResourceNode:
    dependencies = ["query"]
    stats = ["result"]

    async def __call__(self, item_stats: Dict[str, list], resourceA: ResourceA, resourceB: ResourceB) -> Dict[str, List]:
        # use resourceA and resourceB to calculate stat "result"
```

### Note:
* combining multiple resources per node and multiple resource sets per node has not yet extensively been tested, as there was no use case that required both in parallel. It is advised to not use resources in different resource sets over different nodes, as this may result in conflict.