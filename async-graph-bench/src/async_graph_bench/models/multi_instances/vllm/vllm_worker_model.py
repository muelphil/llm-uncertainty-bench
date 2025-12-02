try:
    from vllm.distributed.parallel_state import (
        destroy_model_parallel,
        destroy_distributed_environment,
    )
    from vllm.sampling_params import SamplingParams
except ImportError as e:
    raise ImportError(
        "To use this functionality, you need to install the 'vllm' module"
    ) from e
try:
    import torch
except ImportError as e:
    raise ImportError("To use this functionality, you need to install the 'torch' module") from e

from ...vllm_model import sampling_params_from_generation_params
from ...vllm_response_wrapper import VLLMResponseWrapper


class RemoteVLLMModel:
    def __init__(self, worker_client: "WorkerClient", use_chat_template: bool = True, reasoning_parser_mode=None):
        self.worker_client = worker_client
        self.use_chat_template = use_chat_template
        self.reasoning_parser_mode = reasoning_parser_mode

    async def query(self, prompt, generation_params):
        sampling_params = sampling_params_from_generation_params(generation_params)

        response = await self.worker_client.call(
            "generate" if not self.use_chat_template else "chat",
            prompt,
            sampling_params=sampling_params
        )
        return VLLMResponseWrapper(response, n_logprobs=getattr(generation_params, "logprobs", None),
                                   reasoning_parser_mode=self.reasoning_parser_mode)
