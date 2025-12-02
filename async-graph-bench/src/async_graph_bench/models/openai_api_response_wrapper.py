from typing import List, Tuple

from .abstract_classes import ResponseWrapper

try:
    from openai import AsyncOpenAI
except ImportError as e:
    raise ImportError("To use this functionality, you need to install the 'openai' module") from e

import logging

logging.getLogger("httpx").setLevel(logging.WARNING)


def decode_whitespace(str: str) -> str:
    return str.replace("Ġ", " ").replace("Ċ", " ").replace("▁", " ")


class OpenAIAPIResponseWrapper(ResponseWrapper):
    def get_token_ids(self) -> List[List[int]]:
        token_ids = [
            c.token_ids
            for r in self.responses
            for c in r.choices
        ]
        assert all(t is not None for t in
                   token_ids), f"No token ids present in the responses, either due to configuration of generation parameters or model/inference api/provider restrictions."
        return token_ids

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
        assert self.check_logprobs(), f"No logprobs present in the responses, please check the generation parameters - to get tokens from openai api endpoints, logprobs must be specified"
        return [[decode_whitespace(c.token) for c in choice.logprobs.content] for response in self.responses for choice
                in response.choices]

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

    def get_assistant_messages(self) -> List[str]:
        return self.get_messages()

    def get_reasoning_messages(self) -> List[List[str]]:
        """Return reasoning segments (if available)."""
        return [
            [c.message.reasoning_content] if getattr(c.message, "reasoning_content", None) else []
            for r in self.responses
            for c in r.choices
        ]

    def get_reasoning_tokens(self) -> List[List[List[str]]]:
        """Reasoning models may expose reasoning token streams; stub for compatibility."""
        raise ValueError("Reasoning tokens are not provided by OpenAI API endpoints.")

    def get_assistant_tokens(self) -> List[List[str]]:
        return self.get_tokens()  # same as main tokens for now

    def get_reasoning_token_ids(self) -> List[List[List[int]]]:
        raise ValueError("Reasoning token ids are not provided by OpenAI API endpoints.")

    def get_assistant_token_ids(self) -> List[List[int]]:
        return self.get_token_ids()

    def get_reasoning_logprobs(self) -> List[List[List[float]]]:
        raise ValueError("Reasoning logprobs are not provided by OpenAI API endpoints.")

    def get_assistant_logprobs(self) -> List[List[float]]:
        return self.get_logprobs()

    def get_reasoning_tokens_alternatives(self) -> List[List[List[List[Tuple[str, float]]]]]:
        raise ValueError("Reasoning token alternatives are not provided by OpenAI API endpoints.")

    def get_assistant_tokens_alternatives(self) -> List[List[List[Tuple[str, float]]]]:
        return self.get_tokens_alternatives()

    def get_finish_reasons(self) -> list[list[float]]:
        return [choice.finish_reason for response in self.responses for choice in response.choices]
