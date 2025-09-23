import logging

import torch
from async_graph_bench import ResourcePool

log = logging.getLogger(__name__)


async def build_model_from_config(env, m, gpu_type, default_kwargs={}, additional_kwargs={}):
    from async_graph_bench.models.multi_vllm_instances import start_workers, RemoteVLLMModel
    overrides = m.get("kwargs", dict())
    overrides_gpu = m.get(f"kwargs_{gpu_type}", dict())
    llm_args = {**default_kwargs, **overrides, **overrides_gpu, **additional_kwargs}
    if not hasattr(env, "main_model_pool"):
        print("GPUs detected for usage of worker building: ", torch.cuda.device_count())
        worker_clients, close = await start_workers(
            m["name"],
            llm_kwargs=llm_args,
            gpus=list(range(torch.cuda.device_count())),
            gpus_per_worker=llm_args["tensor_parallel_size"],
        )
        models = [
            RemoteVLLMModel(
                worker_client,
                use_chat_template=True,
                reasoning_parser_mode=m.get("reasoning_parser", None)
            )
            for worker_client in worker_clients
        ]
        print("Successfully built", len(models), "LLM instances!")
        resource_pool = ResourcePool(models)
        resource_pool.close = close
        env.main_model_pool = resource_pool
    return [env.main_model_pool]


async def build_encoders(env):
    from async_graph_bench.models.multi_nli_instances import start_encoder_workers, RemoteEncoderModel
    if not hasattr(env, "main_model_pool"):
        print("GPUs detected for usage of worker building: ", torch.cuda.device_count())
        worker_clients, close = await start_encoder_workers(
            model_name="microsoft/deberta-large-mnli",
            model_kwargs={},
            gpus=list(range(torch.cuda.device_count())),
            models_per_gpu=1,
            cache_path="~/nli_cache"
        )
        models = [RemoteEncoderModel(worker_client) for worker_client in worker_clients]
        print("Successfully built", len(models), "Encoder instances!")
        resource_pool = ResourcePool(models)
        resource_pool.close = close
        env.main_model_pool = resource_pool
    return [env.main_model_pool]
