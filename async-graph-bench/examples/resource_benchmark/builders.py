from dataclasses import dataclass
from typing import Optional, List, Dict

from async_graph_bench import ResourcePool


@dataclass
class Endpoint:
    base_url: str
    model: str
    api_key: Optional[str] = None


def get_openai_api_builder(endpoints: List[Endpoint]):
    from async_graph_bench.models.openai_api_model import OpenAIAPIModel
    def build_resources(env):
        if not hasattr(env, "main_resource"):
            models = [
                OpenAIAPIModel(
                    model_id=endpoint.model,
                    openai_endpoint=endpoint.base_url,
                    openai_api_key=endpoint.api_key
                )
                for endpoint in endpoints
            ]
            env.main_resource = ResourcePool(models)
        return env.main_resource

    return build_resources


def get_vllm_builder(model_id: str, llm_args: Dict, use_chat_template=True, reasoning_parser_model=None):
    from vllm import LLM  # lazy importing so packages are only required if the resource builder is generated
    from async_graph_bench.models.vllm_model import VLLMModel
    def build_resources(env):
        if not hasattr(env, "main_resource"):
            llm = LLM(model=model_id, **llm_args)
            vllm_model = VLLMModel(llm, use_chat_template=use_chat_template,
                                   reasoning_parser_mode=reasoning_parser_model)
            env.main_resource = ResourcePool([vllm_model])
        return env.main_resource

    return build_resources


def get_vllm_multi_instance_builder(model_id: str, llm_args: dict, use_chat_template=True,
                                          reasoning_parser_mode=None):
    import torch
    from async_graph_bench.models.multi_instances.vllm import start_workers, RemoteVLLMModel
    async def builder(env):
        if not hasattr(env, "main_model_pool"):
            print("GPUs detected for usage of worker building: ", torch.cuda.device_count())

            # Start multiple vLLM workers across available GPUs
            worker_clients, close = await start_workers(
                model_id,
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

    return builder
