from async_graph_bench.models.reasoning_parsers import parse_deepseek_reasoning, parse_mistral_reasoning, \
    parse_gpt_oss_reasoning
from typing import Literal, Optional

try:
    import torch
except ImportError as e:
    raise ImportError("To use this functionality, you need to install the 'torch' module") from e

from typing import List, Tuple
from . import ResponseWrapper


def decode_if_byte(text: str):
    return text.decode("utf-8", errors="replace") if isinstance(text, bytes) else text


def decode_whitespace(text: str) -> str:
    # TODO this is necessary because decoded_tokens does not return fully decoded tokens from vllm 0.7.0 anymore - this is an observation, not taken from any website. It is unclear why this behaviour was changed or if there is a way to prevent this
    return (text.replace("Ċ", "\n")
            .replace("▁", " ")
            .replace("Ġ", " "))


def find_subarray_backward(arr, sub):
    n, m = len(arr), len(sub)
    for i in range(n - m, -1, -1):  # start from the end
        if arr[i:i + m] == sub:
            return i
    return -1


def _get_post_reasoning_index(tokens, end_of_thinking_pattern):
    index = find_subarray_backward(tokens, end_of_thinking_pattern)
    if index == -1:
        return 0
    return index + len(end_of_thinking_pattern)


def find_index(lst, item):
    try:
        return lst.index(item)
    except ValueError:
        return -1


ReasoningParserMode = Literal[None, "gpt-oss", "deepseek", "mistral"]


class VLLMResponseWrapper(ResponseWrapper):

    def __init__(self, response, n_logprobs=None,
                 reasoning_parser_mode: ReasoningParserMode = None):
        self.data = response
        self.n_logprobs = n_logprobs
        self.reasoning_parser_mode = reasoning_parser_mode
        assert len(self.data) != 0, "Response is empty. Response:" + str(response)
        self.has_logprobs = all(
            hasattr(output.outputs[0], 'logprobs') and output.outputs[0].logprobs is not None
            for output in self.data
        )
        if self.reasoning_parser_mode is not None and not self.has_logprobs:
            raise AssertionError(
                "If a reasoning parser is specified, the response must have logprobs present in order to parse the response!")
        self.reasoning_indices = self._parse_reasoning() if self.reasoning_parser_mode is not None else None

    def _get_post_reasoning_indices(self, end_of_thinking_pattern):
        if isinstance(end_of_thinking_pattern, list):
            if all(isinstance(e, str) for e in end_of_thinking_pattern):
                return [_get_post_reasoning_index(tokens, end_of_thinking_pattern) for tokens in self.get_tokens()]
            elif all(isinstance(e, int) for e in end_of_thinking_pattern):
                return [_get_post_reasoning_index(tokens, end_of_thinking_pattern) for tokens in self.get_token_ids()]
        raise TypeError("end_of_thinking_pattern must be a list of tokens")

    def _parse_reasoning(self):
        if not self.has_logprobs:  # Note: In VLLM the decoded tokens are only available when logprobs is set
            raise ValueError("Logprobs are not available in the vLLM response.")
        if self.reasoning_parser_mode == "gpt-oss" or self.reasoning_parser_mode == "deepseek":
            tokens_per_output = self.get_tokens()
            parser = (parse_gpt_oss_reasoning if self.reasoning_parser_mode == "gpt-oss" else parse_deepseek_reasoning)
            reasoning_indices = [parser(tokens) for tokens in tokens_per_output]
        elif self.reasoning_parser_mode == "mistral":
            token_ids_per_output = self.get_token_ids()
            reasoning_indices = [parse_mistral_reasoning(token_ids) for token_ids in token_ids_per_output]
        else:
            raise ValueError("Unknown reasoning parser mode")

        #for idx, reasoning_index in enumerate(reasoning_indices):
        #    if len(reasoning_index["reasoning"]) == 0:
        #        raise AssertionError("Reasoning mode specified, but no reasoning found in assistant response! " + str(self.get_tokens()[idx]) + " Token lengths= " + str([len(t) for t in self.get_tokens()]))

        return reasoning_indices

    def _get_reasoning(self, properties_in_list):
        return [[tokens[start: end] for (start, end, _) in indices["reasoning"]] for tokens, indices in
                zip(properties_in_list, self.reasoning_indices)]

    def _get_assistant(self, properties_in_list):
        return [tokens[indices["message"][0]: indices["message"][1]] for tokens, indices in
                zip(properties_in_list, self.reasoning_indices)]

    def get_messages(self):
        return [output.outputs[0].text for output in self.data]

    def get_assistant_messages(self) -> List[str]:
        if self.reasoning_indices is None:
            return [output.outputs[0].text for output in self.data]

        return ["".join(tokens) for tokens in self.get_assistant_tokens()]

    def get_reasoning_messages(self) -> List[str]:
        if self.reasoning_indices is None:
            return [[] for output in self.data]

        return [
            ["".join(reasoning_tokens) for reasoning_tokens in reasoning_tokens_list]
            for reasoning_tokens_list
            in self.get_reasoning_tokens()
        ]

    def get_tokens(self) -> List[List[str]]:
        if hasattr(self, "tokens"):
            return self.tokens
        if not self.has_logprobs:  # Note: In VLLM the decoded tokens are only available when logprobs is set
            raise ValueError("Logprobs are not available in the vLLM response.")
        self.tokens = [
            [
                decode_whitespace(decode_if_byte(next(iter(logprob_info.values())).decoded_token))
                for logprob_info
                in output.outputs[0].logprobs
            ] for i, output in enumerate(self.data)
        ]
        return self.tokens

    def get_reasoning_tokens(self):
        if self.reasoning_indices is None:
            return [[] for output in self.data]
        return self._get_reasoning(self.get_tokens())

    def get_assistant_tokens(self):
        if self.reasoning_indices is None:
            return self.get_tokens()
        return self._get_assistant(self.get_tokens())

    def get_token_ids(self) -> List[List[int]]:
        return [output.outputs[0].token_ids for output in self.data]

    def get_reasoning_token_ids(self):
        if self.reasoning_indices is None:
            return [[] for output in self.data]
        return self._get_reasoning(self.get_token_ids())

    def get_assistant_token_ids(self):
        if self.reasoning_indices is None:
            return self.get_token_ids()
        return self._get_assistant(self.get_token_ids())

    def get_logprobs(self) -> List[List[float]]:
        if hasattr(self, "logprobs"):
            return self.logprobs
        if not self.has_logprobs:
            raise ValueError("Logprobs are not available in the vLLM response.")
        self.logprobs = [
            [next(iter(logprob_info.values())).logprob for logprob_info in output.outputs[0].logprobs]
            for output in self.data
        ]
        return self.logprobs

    def get_reasoning_logprobs(self):
        if self.reasoning_indices is None:
            return [[] for output in self.data]
        return self._get_reasoning(self.get_logprobs())

    def get_assistant_logprobs(self):
        if self.reasoning_indices is None:
            return self.get_logprobs()
        return self._get_assistant(self.get_logprobs())

    def get_tokens_alternatives(self) -> List[List[List[Tuple[str, float]]]]:
        if hasattr(self, "token_alternatives"):
            return self.token_alternatives
        if not self.has_logprobs:
            raise ValueError("Logprobs are not available in the vLLM response.")
        self.token_alternatives = [
            [
                [
                    [decode_whitespace(decode_if_byte(token.decoded_token)), token.logprob]
                    for token in logprob_info.values()
                ][:self.n_logprobs]
                for logprob_info in output.outputs[0].logprobs
            ]
            for output in self.data
        ]
        return self.token_alternatives

    def get_reasoning_tokens_alternatives(self):
        if self.reasoning_indices is None:
            return [[] for output in self.data]
        return self._get_reasoning(self.get_tokens_alternatives())

    def get_assistant_tokens_alternatives(self):
        if self.reasoning_indices is None:
            return self.get_tokens_alternatives()
        return self._get_assistant(self.get_tokens_alternatives())

    def get_finish_reasons(self) -> List[str]:
        finish_reasons = [output.outputs[0].finish_reason for output in self.data]
        return finish_reasons
