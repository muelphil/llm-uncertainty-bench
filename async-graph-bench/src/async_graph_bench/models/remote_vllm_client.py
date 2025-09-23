# import aiohttp
# import pickle
#
# try:
#     from vllm.distributed.parallel_state import (
#         destroy_model_parallel,
#         destroy_distributed_environment,
#     )
#     from vllm.sampling_params import SamplingParams
# except ImportError as e:
#     raise ImportError(
#         "To use this functionality, you need to install the 'vllm' module"
#     ) from e
# try:
#     import torch
# except ImportError as e:
#     raise ImportError("To use this functionality, you need to install the 'torch' module") from e
#
# import contextlib
# import gc
# from typing import List, Tuple, Union, Dict, Any
# from . import Model, GenerationParameters, ResponseWrapper
#
#
# def normalize_chat_input(
#         user_input: Union[str, List[str], List[Dict[str, Any]], List[List[Dict[str, Any]]]]
# ) -> List[List[Dict[str, Any]]]:
#     """
#     Wrap user input in chat pattern if necessary.
#     """
#     if isinstance(user_input, str):
#         return [[{"role": "user", "content": user_input}]]
#
#     if isinstance(user_input, list) and all(isinstance(item, str) for item in user_input):
#         return [[{"role": "user", "content": input}] for input in user_input]
#
#     return user_input
#
#
# def decode_if_byte(text: str):
#     return text.decode("utf-8", errors="replace") if isinstance(text, bytes) else text
#
#
# def decode_whitespace(text: str) -> str:
#     # TODO this is necessary because decoded_tokens does not return fully decoded tokens from vllm 0.7.0 anymore - this is an observation, not taken from any website. It is unclear why this behaviour was changed or if there is a way to prevent this
#     return (text.replace("Ċ", "\n")
#             .replace("▁", " ")
#             .replace("Ġ", " "))
#
#
# class VLLMResponseWrapper(ResponseWrapper):
#     def __init__(self, response, n_logprobs=None):
#         self.data = response
#         self.n_logprobs = n_logprobs
#         assert len(self.data) != 0, "Response is empty. Response:" + str(response)
#         self.has_logprobs = all(
#             hasattr(output.outputs[0], 'logprobs') and output.outputs[0].logprobs is not None
#             for output in self.data
#         )
#
#     def get_messages(self) -> List[str]:
#         messages = [output.outputs[0].text for output in self.data]
#         return messages
#
#     def get_token_lengths(self) -> List[List[int]]:
#         token_lengths = [
#             [len(next(iter(logprob_info.values())).decoded_token) for logprob_info in output.outputs[0].logprobs]
#             for output in self.data
#         ]
#         return token_lengths
#
#     def get_tokens(self) -> List[List[str]]:
#         if not self.has_logprobs:
#             raise ValueError("Logprobs are not available in the vLLM response.")
#         tokens = [
#             [decode_whitespace(decode_if_byte(next(iter(logprob_info.values())).decoded_token)) for logprob_info in
#              output.outputs[0].logprobs]
#             for output in self.data
#         ]
#         return tokens
#
#     def get_token_ids(self) -> List[List[int]]:
#         token_ids = [output.outputs[0].token_ids for output in self.data]
#         return token_ids
#
#     def get_logprobs(self) -> List[List[float]]:
#         if not self.has_logprobs:
#             raise ValueError("Logprobs are not available in the vLLM response.")
#         logprobs = [
#             [next(iter(logprob_info.values())).logprob for logprob_info in output.outputs[0].logprobs]
#             for output in self.data
#         ]
#         return logprobs
#
#     def get_tokens_alternatives(self) -> List[List[List[Tuple[str, float]]]]:
#         if not self.has_logprobs:
#             raise ValueError("Logprobs are not available in the vLLM response.")
#         token_alternatives = [
#
#             [
#                 [
#                     [decode_whitespace(decode_if_byte(token.decoded_token)), token.logprob]
#                     for token in logprob_info.values()
#                 ][:self.n_logprobs]
#                 for logprob_info in output.outputs[0].logprobs
#             ]
#             for output in self.data
#         ]
#         return token_alternatives
#
#     def get_greedy_log_probs(self) -> List[List[List[float]]]:
#         if not self.has_logprobs:
#             raise ValueError("Logprobs are not available in the vLLM response.")
#         greedy_logprobs = [
#             [[token.logprob for token in logprob_info.values()][:self.n_logprobs]
#              for logprob_info in output.outputs[0].logprobs]
#             for output in self.data
#         ]
#         return greedy_logprobs
#
#     def get_finish_reasons(self) -> List[str]:
#         finish_reasons = [output.outputs[0].finish_reason for output in self.data]
#         return finish_reasons
#
#
# class RemoteVLLMModelOld:
#     def __init__(self, endpoint: str, use_chat_template: bool = True):
#         self.endpoint = endpoint.rstrip("/")
#         self.use_chat_template = use_chat_template
#
#     async def query(self, prompt, generation_params):
#         prompts = normalize_chat_input(prompt) if self.use_chat_template else prompt
#         sampling_kwargs = generation_params.to_dict()
#
#         data = pickle.dumps((prompts, sampling_kwargs))
#         url = f"{self.endpoint}/chat" if self.use_chat_template else f"{self.endpoint}/generate"
#
#         async with aiohttp.ClientSession() as session:
#             async with session.post(url, data=data) as resp:
#                 raw = await resp.read()
#                 if resp.status != 200:
#                     raise RuntimeError(f"Request failed with status {resp.status}: {raw.decode()}")
#                 try:
#                     outputs = pickle.loads(raw)
#                 except:
#                     print("Failed to decode response")
#                     raise
#         return VLLMResponseWrapper(outputs, n_logprobs=getattr(generation_params, "logprobs", None))
