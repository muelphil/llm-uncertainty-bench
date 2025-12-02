# Multiple Instances of VLLM

The **multi-instance vLLM setup** allows launching multiple `vLLM` models across available GPUs, managing them in a `ResourcePool` to maximize throughput during benchmarks. Each model runs in its own subprocess, and the framework handles communication transparently using the same [`ResponseWrapper`](../api/responsewrapper.md)` interface as `VLLMModel`.

This approach is particularly useful for **high-throughput benchmarking**, where a single model may become a bottleneck. Depending on your system and workload, not all created models may be used simultaneously; you can tune the number of active models versus GPUs to achieve optimal performance.

---

## Key Features

* Launch multiple vLLM instances in separate processes.
* Each instance can be placed on a different GPU.
* Access and manage models via a `ResourcePool`.
* Use the same [`ResponseWrapper`](../api/responsewrapper.md)` and [`Model`](../api/model.md) interface as single `VLLMModel`.
* Supports reasoning parsing through the `reasoning_parser_mode` parameter.

---

## Resource Builder Example

```python
import torch
from async_graph_bench import ResourcePool
from async_graph_bench.models.multi_instances.vllm import start_workers, RemoteVLLMModel

async def build_model_from_config(env, model_name: str, llm_args: dict, reasoning_parser_mode=None):
    if not hasattr(env, "main_model_pool"):
        print("GPUs detected for usage of worker building: ", torch.cuda.device_count())
        
        # Start multiple vLLM workers across available GPUs
        worker_clients, close = await start_workers(
            model_name,
            llm_kwargs=llm_args,
            gpus=list(range(torch.cuda.device_count())),
            gpus_per_worker=llm_args["tensor_parallel_size"],
        )
        
        # Wrap each worker client in RemoteVLLMModel
        models = [
            RemoteVLLMModel(
                worker_client,
                use_chat_template=True,
                reasoning_parser_mode=reasoning_parser_mode
            )
            for worker_client in worker_clients
        ]
        
        print("Successfully built", len(models), "LLM instances!")
        
        # Manage all models in a ResourcePool
        resource_pool = ResourcePool(models)
        resource_pool.on_close(close)
        env.main_model_pool = resource_pool

    return [env.main_model_pool]
```

This function can be provided as a `resource_builder` for a node:

```python
NodeConfig(
    MyNode(),
    resource_builder=build_model_from_config
)
```

---

## Notes on Optimal Usage

* The number of active models does **not need to match the number of GPUs** exactly. Throughput may be gated by the speed of the main benchmarking threads.
* For example, on an 8-GPU system, creating 8 models may not result in all 8 models being used simultaneously, as the main benchmarking threads evaluation is the bottleneck.
* Test different configurations to determine the **optimal number of GPUs** for your workload.
* Each `RemoteVLLMModel` instance behaves like a normal `VLLMModel` with all methods available, including reasoning parsing.

This setup provides a flexible way to scale benchmarks while fully utilizing available GPU resources.
