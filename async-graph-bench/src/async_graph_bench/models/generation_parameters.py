from typing import Any, Dict, List, Optional, Union


# TODO might want to adjust to feature every parameter from https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py


class GenerationParameters:
    """
    Parameters to control text generation in language models.

    Attributes:
        n (int): Number of output sequences to return for the given prompt. Default is 1.
        best_of (Optional[int]): Number of output sequences generated from the prompt. From these, the top `n` sequences are returned. Must be greater than or equal to `n`. Default is None.
        temperature (float): Controls the randomness of sampling. Lower values make the model more deterministic, while higher values make it more random. Zero means greedy sampling. Default is 1.0.
        top_p (float): Controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1 to consider all tokens. Default is 1.0.
        top_k (int): Controls the number of top tokens to consider. Set to -1 to consider all tokens. Default is -1.
        min_p (float): Minimum probability for a token to be considered, relative to the probability of the most likely token. Must be in [0, 1]. Set to 0 to disable this. Default is 0.0.
        presence_penalty (float): Penalizes new tokens based on their presence in the generated text so far. Positive values encourage new topics, while negative values encourage repetition. Default is 0.0.
        frequency_penalty (float): Penalizes new tokens based on their frequency in the generated text so far. Positive values encourage new tokens, while negative values encourage repetition. Default is 0.0.
        repetition_penalty (float): Penalizes new tokens based on their presence in the prompt and generated text. Values > 1 encourage new tokens, while values < 1 encourage repetition. Default is 1.0.
        seed (Optional[int]): Random seed for generation. Default is None.
        stop (Union[str, List[str], None]): String or list of strings that stop the generation when encountered. The returned output will not contain the stop strings. Default is None.
        stop_token_ids (Optional[List[int]]): List of token IDs that stop the generation when encountered. The returned output will contain the stop tokens unless they are special tokens. Default is None.
        bad_words (Optional[List[str]]): List of words that are not allowed to be generated. Only the last token of a corresponding token sequence is restricted when the next generated token can complete the sequence. Default is None.
        ignore_eos (bool): If True, the model ignores the end-of-sequence token and continues generating tokens. Default is False.
        max_tokens (Optional[int]): Maximum number of tokens to generate per output sequence. Default is None.
        min_tokens (int): Minimum number of tokens to generate per output sequence before EOS or stop tokens can be generated. Default is 0.
        logprobs (Optional[int]): Number of log probabilities to return per output token. If set, the result includes the log probabilities of the specified number of most likely tokens, as well as the chosen tokens. Default is None.
        prompt_logprobs (Optional[int]): Number of log probabilities to return per prompt token. Default is None.
        detokenize (bool): If True, detokenizes the output. Default is True.
        skip_special_tokens (bool): If True, skips special tokens in the output. Default is True.
        spaces_between_special_tokens (bool): If True, adds spaces between special tokens in the output. Default is True.
        include_stop_str_in_output (bool): If True, includes the stop strings in the output text. Default is False.
        truncate_prompt_tokens (Optional[int]): If set, uses only the last `k` tokens from the prompt (left truncation). Default is None.
        logit_bias (Optional[dict]): Dictionary mapping token IDs to bias scores, modifying the likelihood of specific tokens being generated. Default is None.
        allowed_token_ids (Optional[List[int]]): List of token IDs that are allowed to be generated. If set, only these tokens will be considered during generation. Default is None.
    """

    def __init__(
            self,
            n: Optional[int] = None,
            best_of: Optional[int] = None,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            top_k: Optional[int] = None,
            min_p: Optional[float] = None,
            presence_penalty: Optional[float] = None,
            frequency_penalty: Optional[float] = None,
            repetition_penalty: Optional[float] = None,
            seed: Optional[int] = None,
            stop: Optional[Union[str, List[str]]] = None,
            stop_token_ids: Optional[List[int]] = None,
            bad_words: Optional[List[str]] = None,
            ignore_eos: Optional[bool] = None,
            max_tokens: Optional[int] = None,
            min_tokens: Optional[int] = None,
            logprobs: Optional[int] = None,
            prompt_logprobs: Optional[int] = None,
            detokenize: Optional[bool] = None,
            skip_special_tokens: Optional[bool] = None,
            spaces_between_special_tokens: Optional[bool] = None,
            include_stop_str_in_output: Optional[bool] = None,
            truncate_prompt_tokens: Optional[int] = None,
            logit_bias: Optional[Dict[int, float]] = None,
            allowed_token_ids: Optional[List[int]] = None,
            response_format: Optional[Dict] = None
    ):
        """
        Initialize generation parameters with optional user-defined values.

        Args:
            All parameters are optional. Only those provided will be stored.
        """
        # Store only user-defined parameters
        self.parameters = {
            key: value
            for key, value in locals().items()
            if key != "self" and value is not None
        }

    def to_dict(self) -> dict:
        """
        Retrieve the internal parameters' dictionary.

        Returns:
            dict: Dictionary of user-defined parameters.
        """
        return self.parameters.copy()

    def adapt_for_model(self, mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        Adapts the parameter names to match the naming conventions of a specific model API.

        Args:
            mapping (dict): A dictionary mapping internal parameter names to the model's parameter names.

        Returns:
            dict: A dictionary of parameters mapped to the model's parameter names, excluding any where mapping[key] is None.
        """
        return {
            mapped_key: value
            for key, value in self.parameters.items()
            if (mapped_key := mapping.get(key)) is not None
        }
