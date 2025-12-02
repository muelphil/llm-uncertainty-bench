from abc import ABC, abstractmethod
from typing import Any, List, Tuple

from .generation_parameters import GenerationParameters


class ResponseWrapper(ABC):
    """
    Abstract base class for wrapping language model responses.

    Provides a standard interface to access tokens, log probabilities,
    reasoning information, and assistant messages from a model response.

    Subclasses should implement these methods according to the specific
    response format returned by the language model.
    """

    def __init__(self, response: Any):
        """
        Initialize the ResponseWrapper with the raw response.

        Args:
            response (Any): Raw response data from the language model.
        """
        self.data = response

    @abstractmethod
    def get_messages(self) -> List[str]:
        """
        Retrieve the primary messages (text outputs) from the response.

        Returns:
            List[str]: List of messages from each response.
        """
        pass

    @abstractmethod
    def get_assistant_messages(self) -> List[str]:
        """
        Retrieve only the assistant messages from the response,
        if reasoning parsing is enabled. Otherwise returns all messages.

        Returns:
            List[str]: Assistant messages.
        """
        pass

    @abstractmethod
    def get_reasoning_messages(self) -> List[List[str]]:
        """
        Retrieve the reasoning messages for each response, if available.

        Returns:
            List[List[str]]: List of lists of reasoning segments per response.
        """
        pass

    @abstractmethod
    def get_tokens(self) -> List[List[str]]:
        """
        Retrieve the tokens from the response, if available.

        Returns:
            List[List[str]]: Tokenized messages.
        """
        pass

    @abstractmethod
    def get_reasoning_tokens(self) -> List[List[List[str]]]:
        """
        Retrieve reasoning tokens for each response, if reasoning parsing is enabled.

        Returns:
            List[List[List[str]]]: Nested list of reasoning tokens per response.
        """
        pass

    @abstractmethod
    def get_assistant_tokens(self) -> List[List[str]]:
        """
        Retrieve the assistant tokens for each response.

        Returns:
            List[List[str]]: Tokenized assistant messages.
        """
        pass

    @abstractmethod
    def get_token_ids(self) -> List[List[int]]:
        """
        Retrieve token IDs for each response.

        Returns:
            List[List[int]]: Token IDs.
        """
        pass

    @abstractmethod
    def get_reasoning_token_ids(self) -> List[List[List[int]]]:
        """
        Retrieve reasoning token IDs for each response, if available.

        Returns:
            List[List[List[int]]]: Nested list of reasoning token IDs.
        """
        pass

    @abstractmethod
    def get_assistant_token_ids(self) -> List[List[int]]:
        """
        Retrieve token IDs corresponding to the assistant messages.

        Returns:
            List[List[int]]: Token IDs of assistant messages.
        """
        pass

    @abstractmethod
    def get_logprobs(self) -> List[List[float]]:
        """
        Retrieve log probabilities of each token in the response.

        Returns:
            List[List[float]]: Log probabilities per token.
        """
        pass

    @abstractmethod
    def get_reasoning_logprobs(self) -> List[List[List[float]]]:
        """
        Retrieve log probabilities corresponding to reasoning tokens.

        Returns:
            List[List[List[float]]]: Nested list of log probabilities per reasoning token.
        """
        pass

    @abstractmethod
    def get_assistant_logprobs(self) -> List[List[float]]:
        """
        Retrieve log probabilities corresponding to assistant tokens.

        Returns:
            List[List[float]]: Log probabilities of assistant tokens.
        """
        pass

    @abstractmethod
    def get_tokens_alternatives(self) -> List[List[List[Tuple[str, float]]]]:
        """
        Retrieve alternative tokens and their log probabilities for each token.

        Returns:
            List[List[List[Tuple[str, float]]]]: Nested list of alternative tokens with logprobs.
        """
        pass

    @abstractmethod
    def get_reasoning_tokens_alternatives(self) -> List[List[List[List[Tuple[str, float]]]]]:
        """
        Retrieve alternative tokens for reasoning segments.

        Returns:
            List[List[List[List[Tuple[str, float]]]]]: Nested alternatives for reasoning tokens.
        """
        pass

    @abstractmethod
    def get_assistant_tokens_alternatives(self) -> List[List[List[Tuple[str, float]]]]:
        """
        Retrieve alternative tokens for assistant messages.

        Returns:
            List[List[List[Tuple[str, float]]]]: Nested alternatives for assistant tokens.
        """
        pass

    @abstractmethod
    def get_finish_reasons(self) -> List[str]:
        """
        Retrieve the reasons for finishing each generation.

        Returns:
            List[str]: Finish reasons for each response.
        """
        pass


class Model(ABC):
    """
    Abstract model class. Used as base class for both White-box models and Black-box models.
    """

    @abstractmethod
    async def query(self, prompt: List[str] | str | List[dict] | List[List[dict]],
                    generation_params: GenerationParameters) -> ResponseWrapper:
        """
        Abstract method. Generates a list of model answers using input texts batch.

        Parameters:
            prompt:
            generation_params: TODO
        Return:
            List[str]: corresponding model generations. Have the same length as `input_texts`.
        """
        raise Exception("Not implemented")
