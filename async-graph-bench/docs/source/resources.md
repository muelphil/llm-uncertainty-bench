# Resources

Resources provide controlled access to external or shared components used during node execution — for example, API clients, GPU handles, model instances, or database connections. They allow nodes to use expensive or limited objects efficiently and concurrently through a pooling mechanism.

Each node that depends on one or more resources declares them through a **resource builder** in its [`NodeConfig`](./api/nodeconfig.md). Resource builders construct and configure the resources, wrap them in `ResourcePool` instances, and return them to the framework for coordinated use.

---

## Resource Builders

A **resource builder** is a callable assigned to `NodeConfig.resource_builder`. It is responsible for creating one or multiple resources and returning them as one or more `ResourcePool` instances.

Builders are invoked with a single argument — a **builder environment** (`env`) — which acts as a shared context. This environment allows resources to be constructed once and reused across multiple nodes that share the same dependencies. A builder may return a single `ResourcePool` or a list of pools if multiple independent resource types are needed. Builders may also be asynchronous if resource initialization requires async setup (e.g., opening connections).

---

## Example: Single Resource Type

The following example defines an asynchronous API resource and a resource builder that creates a pool of two such resources.

```python
class MyAPIResource:
    def __init__(self, url: str):
        self.url = url

    async def get_result(self, query):
        # Perform an async request to the API
        return await fetch_from_api(self.url, query)

    def close(self):
        # Close any persistent connections if necessary
        pass
```

A builder that creates two endpoints and registers cleanup:

```python
from async_graph_bench import ResourcePool


def build_resources(env):
    if not hasattr(env, "my_resources"):
        resources = [
            MyAPIResource("http://provider-a.com"),
            MyAPIResource("http://provider-b.com"),
        ]
        pool = ResourcePool(resources)

        def close():
            for r in resources:
                r.close()

        pool.on_close(close)
        env.my_resources = pool
    return env.my_resources
```

This builder ensures resources are created only once per environment and are properly closed when the pool is terminated.

A node that uses this resource might look like:

```python
class MyNode:
    dependencies = ["query"]
    stats = ["result"]

    async def __call__(self, item_stats: dict[str, list], resource: MyAPIResource) -> dict[str, list]:
        return {
            "result": [
                await resource.get_result(query)
                for query in item_stats["query"]
            ]
        }
```

Configuration example:

```python
NodeConfig(
    MyNode(),
    resource_builder=build_resources,
)
```

If two resources are available, up to two concurrent instances of `MyNode` may execute — one per resource — significantly improving throughput.

---

## Example: Multiple Resource Types

A node may depend on more than one resource type. In this case, the builder returns a list of `ResourcePool` instances. The framework ensures that a node instance acquires one resource from each pool before execution.

```python
from async_graph_bench import ResourcePool


def build_resources(env):
    if not hasattr(env, "resource_a_pool"):
        env.resource_a_pool = ResourcePool([ResourceA(), ResourceA()])  # 2 instances
    if not hasattr(env, "resource_b_pool"):
        env.resource_b_pool = ResourcePool([ResourceB(), ResourceB(), ResourceB()])  # 3 instances
    return [env.resource_a_pool, env.resource_b_pool]
```

If two `ResourceA` and three `ResourceB` instances are available, at most `min(2, 3) = 2` concurrent node executions can take place, since each run requires both resources.

Example node using both:

```python
class MyMultiResourceNode:
    dependencies = ["query"]
    stats = ["result"]

    async def __call__(self, item_stats: dict[str, list], resourceA: ResourceA, resourceB: ResourceB) -> dict[str, list]:
        # Perform combined computation using both resources
        pass
```

Configuration:

```python
NodeConfig(
    MyMultiResourceNode(),
    resource_builder=build_resources,
)
```

---

## Summary

Resources allow nodes to make efficient, concurrent use of shared or external objects. They are defined declaratively through **builders** that create and manage **pools**, ensuring reusability, proper cleanup, and scalability.

* Builders are configured per node via `NodeConfig.resource_builder`.
* Each builder returns one or more `ResourcePool` instances.
* Pools automatically coordinate concurrent access and cleanup.
* Multiple asynchronous resources enable concurrent execution of nodes.