try:
    from vllm.distributed.parallel_state import (
        destroy_model_parallel,
        destroy_distributed_environment,
    )
    from vllm.sampling_params import SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
except ImportError as e:
    raise ImportError(
        "To use this functionality, you need to install the 'vllm' module"
    ) from e
try:
    import torch
except ImportError as e:
    raise ImportError("To use this functionality, you need to install the 'torch' module") from e

import contextlib
import gc
from typing import List, Union, Dict, Any

from . import Model, GenerationParameters
from .vllm_response_wrapper import VLLMResponseWrapper, ReasoningParserMode


def normalize_chat_input(
        user_input: Union[str, List[str], List[Dict[str, Any]], List[List[Dict[str, Any]]]]
) -> List[List[Dict[str, Any]]]:
    """
    Wrap user input in chat pattern if necessary.
    """
    if isinstance(user_input, str):
        return [[{"role": "user", "content": user_input}]]

    if isinstance(user_input, list) and all(isinstance(item, str) for item in user_input):
        return [[{"role": "user", "content": input}] for input in user_input]

    return user_input


def sampling_params_from_generation_params(generation_params: GenerationParameters) -> SamplingParams:
    params_dict = generation_params.to_dict()
    if "response_format" in params_dict:
        response_format = params_dict["response_format"]
        if response_format["type"] == "json_schema":
            guided_decoding_params = GuidedDecodingParams(json=response_format["json_schema"])
        elif response_format["type"] == "json_object":
            guided_decoding_params = GuidedDecodingParams(json_object=True)
        elif response_format["type"] == "json_object":
            guided_decoding_params = GuidedDecodingParams(json_object=True)
        elif response_format["type"] == "choice":
            guided_decoding_params = GuidedDecodingParams(choice=response_format["choice"])
        elif response_format["type"] == "regex":
            guided_decoding_params = GuidedDecodingParams(regex=response_format["regex"])
        else:
            raise NotImplementedError(f"GenerationsParameter Response Format Type {response_format} not supported")
        del params_dict["response_format"]
        params_dict["guided_decoding"] = guided_decoding_params
    return SamplingParams(**params_dict)


class VLLMModel(Model):
    def __init__(self, model, use_chat_template=True, reasoning_parser_mode: ReasoningParserMode = None):
        self.model = model
        self.use_chat_template = use_chat_template
        self.reasoning_parser_mode = reasoning_parser_mode
        self.query_model = self.model.chat if self.use_chat_template else self.model.generate

    async def query(self, prompt, generation_params: GenerationParameters) -> VLLMResponseWrapper:
        if self.use_chat_template:
            prompt = normalize_chat_input(prompt)

        response = self.query_model(
            prompt,
            sampling_params=sampling_params_from_generation_params(generation_params),
            use_tqdm=False
        )

        return VLLMResponseWrapper(response=response,
                                   n_logprobs=generation_params.logprobs if hasattr(generation_params,
                                                                                    'logprobs') else None,
                                   reasoning_parser_mode=self.reasoning_parser_mode)

    def close(self):
        # TODO https://github.com/vllm-project/vllm/issues/1908
        try:  # Note: try except necessary as different version of vllm provide different ways to close
            destroy_model_parallel()
        except Exception as e:
            pass
        try:
            destroy_distributed_environment()
        except Exception as e:
            pass
        try:
            if hasattr(self, 'model') and self.model is not None:
                model_tmp = self.model
                del self.model  # Isn't necessary for releasing memory, but why not
                del model_tmp.llm_engine.model_executor.driver_worker
                del model_tmp.llm_engine.model_executor
        except Exception as e:
            pass
        try:
            with contextlib.suppress(AssertionError):
                torch.distributed.destroy_process_group()
        except Exception as e:
            pass
        gc.collect()
        torch.cuda.empty_cache()
