import os
from typing import List
import asyncio
from . import Model, GenerationParameters

try:
    from openai import AsyncOpenAI
except ImportError as e:
    raise ImportError("To use this functionality, you need to install the 'openai' module") from e

import logging

logging.getLogger("httpx").setLevel(logging.WARNING)


def decode_whitespace(str):
    return str.replace("Ġ", " ").replace("Ċ", " ").replace("▁", " ")


class OpenAIAPIResponseWrapper:
    def __init__(self, n_logprobs=None):
        self.responses = []
        self.has_logprobs = None
        self.n_logprobs = n_logprobs

    def check_logprobs(self) -> bool:
        if self.has_logprobs is None:
            self.has_logprobs = all(
                hasattr(data.choices[0], 'logprobs') and bool(data.choices[0].logprobs) for data in self.responses)
        return self.has_logprobs

    def append_response(self, response):
        self.responses.append(response)
        self.has_logprobs = None

    def get_messages(self) -> List[str]:
        return [choice.message.content for response in self.responses for choice in response.choices]

    def get_token_lengths(self) -> List[List[int]]:
        assert self.check_logprobs(), f"No logprobs present in the responses, please check the generation parameters"
        return [[len(c.token) for c in choice.logprobs.content] for response in self.responses for choice in
                response.choices]

    def get_tokens(self) -> List[List[str]]:
        assert self.check_logprobs(), f"No logprobs present in the responses, please check the generation parameters"
        return [[decode_whitespace(c.token) for c in choice.logprobs.content] for response in self.responses for choice
                in
                response.choices]

    def get_logprobs(self) -> list[list[float]]:
        assert self.check_logprobs(), f"No logprobs present in the responses, please check the generation parameters"
        return [[c.logprob for c in choice.logprobs.content] for response in self.responses for choice in
                response.choices]

    def get_top_logprobs(self) -> list[list[dict]]:
        assert self.check_logprobs(), f"No logprobs present in the responses, please check the generation parameters"
        return [
            [{decode_whitespace(t.token): t.logprob for t in content.top_logprobs} for content in
             choice.logprobs.content] \
            for response in self.responses for choice in response.choices
        ]

    def get_tokens_alternatives(self) -> list[list[list]]:
        assert self.check_logprobs(), f"No logprobs present in the responses, please check the generation parameters"
        return [
            [[[decode_whitespace(t.token), t.logprob] for t in content.top_logprobs[:self.n_logprobs]] for content in
             choice.logprobs.content] \
            for response in self.responses for choice in response.choices
        ]

    def get_greedy_log_probs(self):
        assert self.check_logprobs(), f"No logprobs present in the responses, please check the generation parameters"
        return [
            [[alternative.logprob for alternative in token.top_logprobs[:self.n_logprobs]] for token in
             choice.logprobs.content] \
            for response in self.responses for choice in response.choices
        ]

    def get_finish_reasons(self) -> list[list[float]]:
        return [choice.finish_reason for response in self.responses for choice in response.choices]


class OpenAIAPIModel(Model):
    def __init__(
            self,
            model_id: str,
            openai_api_key: str = None,
            openai_endpoint: str = None,
            limit_logprobs: bool = False,
            use_chat: bool = True,
    ):
        """
        use_chat: True => call chat.completions.create with `messages=...`
                  False => call completions.create with `prompt=...`
        """
        self.model_id = model_id
        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", None)
        base_url = openai_endpoint or os.environ.get("OPENAI_BASE_URL", None)
        assert api_key is not None, "No API key provided"
        self.openai_api = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.limit_logprobs = limit_logprobs
        self.use_chat = use_chat

        # pick the correct underlying create function and kw name once
        if self.use_chat:
            self._create = self.openai_api.chat.completions.create
            self._payload_kw = "messages"
        else:
            self._create = self.openai_api.completions.create
            self._payload_kw = "prompt"

    async def query(self, prompt, generation_params: GenerationParameters) -> OpenAIAPIResponseWrapper:
        wrapper = OpenAIAPIResponseWrapper(
            n_logprobs=generation_params.logprobs
            if self.limit_logprobs and hasattr(generation_params, "logprobs")
            else None
        )

        params = generation_params.to_dict()
        if "logprobs" in params:
            if params["logprobs"] != 0:
                params["top_logprobs"] = min(params["logprobs"], 20)
            params["logprobs"] = bool(params["logprobs"])

        def messages_to_text(messages_list):
            return "\n".join(m.get("content", "") for m in messages_list)

        # normalize into payloads appropriate for the chosen API:
        # - chat -> payloads is a list where each element is a messages list (List[Dict])
        # - completions -> payloads is a list where each element is a prompt string
        payloads = []

        if isinstance(prompt, str):
            payloads = ([{"role": "user", "content": prompt}] if self.use_chat else prompt)
            # wrap into list-of-payloads
            payloads = [payloads]

        elif isinstance(prompt, list):
            if not prompt:
                raise ValueError("Empty prompt list is not supported")

            # list[str]
            if all(isinstance(item, str) for item in prompt):
                if self.use_chat:
                    # IMPORTANT: each item must be a messages *list* (not a dict)
                    payloads = [[{"role": "user", "content": s}] for s in prompt]
                else:
                    payloads = list(prompt)

            # list[dict] -> treat as a single messages list (chat) or convert to one prompt (completions)
            elif all(isinstance(item, dict) for item in prompt):
                if self.use_chat:
                    payloads = [prompt]  # single messages list
                else:
                    payloads = [messages_to_text(prompt)]

            # list[list[dict]] -> multiple messages payloads
            elif all(isinstance(item, list) and item and all(isinstance(m, dict) for m in item) for item in prompt):
                if self.use_chat:
                    payloads = list(prompt)  # already list of messages lists
                else:
                    payloads = [messages_to_text(item) for item in prompt]
            else:
                raise ValueError(
                    "Invalid prompt shape. Accepts: str, list[str], list[dict], or list[list[dict]]."
                )
        else:
            raise ValueError("Unsupported prompt type")

        # Build and run tasks
        base_params = {**params, "model": self.model_id}
        tasks = []
        for p in payloads:
            if self.use_chat:
                call_kwargs = {**base_params, "messages": p}
            else:
                call_kwargs = {**base_params, "prompt": p}
            tasks.append(self._create(**call_kwargs))

        responses = await asyncio.gather(*tasks)
        for r in responses:
            wrapper.append_response(r)
        return wrapper

# TODO deprecated
# class OpenAIAPIModel(Model):
#     def __init__(
#             self,
#             model_id: str,
#             openai_api_key: str = None,
#             openai_endpoint: str = None,
#             limit_logprobs: bool = False
#     ):
#         """
#         Parameters:
#             openai_api_key (Optional[str]): OpenAI API key if the blackbox model comes from OpenAI. Default: None.
#             model_id (Optional[str]): Unique model path. Openai model name, if `openai_api_key` is specified,
#                 huggingface path, if `hf_api_token` is specified. Default: None.
#         """
#         self.model_id = model_id
#         api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", None)
#         base_url = openai_endpoint or os.environ.get("OPENAI_BASE_URL", None)
#         assert api_key is not None, f"No API key provided: {api_key}"
#         self.openai_api = AsyncOpenAI(api_key=api_key, base_url=base_url)
#         self.limit_logprobs = limit_logprobs
#
#     async def query(self, prompt, generation_params: GenerationParameters) -> OpenAIAPIResponseWrapper:
#         wrapper = OpenAIAPIResponseWrapper(
#             n_logprobs=generation_params.logprobs if self.limit_logprobs and hasattr(generation_params,
#                                                                                      'logprobs') else None)
#         params = generation_params.to_dict()
#         if 'logprobs' in params:  # logprobs is numeric in GenerationParameters
#             if params['logprobs'] != 0:
#                 params['top_logprobs'] = min(params['logprobs'], 5)
#             params['logprobs'] = bool(params['logprobs'])
#
#         if isinstance(prompt, str) or (
#                 isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt)):  # single message
#             # If prompt is a string, create a single message with "user" role
#             if isinstance(prompt, str):
#                 prompt = [{"role": "user", "content": prompt}]
#             # open ai clients already use retry logic
#             response = await self.openai_api.chat.completions.create(
#                 model=self.model_id,
#                 messages=prompt,
#                 **params
#             )
#             wrapper.append_response(response)
#         else:  # multiple messages
#             tasks = [
#                 self.openai_api.chat.completions.create(
#                     model=self.model_id,
#                     messages=item,
#                     **params
#                 )
#                 for item in prompt
#             ]
#             responses = await asyncio.gather(*tasks)
#             for response in responses:
#                 wrapper.append_response(response)
#
#         return wrapper
