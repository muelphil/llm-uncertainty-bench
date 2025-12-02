from typing import Dict, List

from async_graph_bench import Model
from async_graph_bench.models import GenerationParameters


class QueryModel:
    """
    Node that queries a language model with a set of input texts and returns the model responses along with token counts.

    This node is intended for benchmarking LLM performance on large prompts. Each input is wrapped in a
    standard instruction asking the model to provide a detailed response.
    """

    description = "Queries a language model with large prompts and returns generated responses and token counts."
    requires = ["input_texts"]
    provides = ["responses", "token_lengths"]

    def __init__(self, max_tokens: int):
        """
        Initialize the QueryModel node.

        Args:
            max_tokens (int): Maximum number of tokens to generate per prompt.
        """
        self.max_tokens = max_tokens
        self.generation_params = GenerationParameters(
            max_tokens=self.max_tokens,
            logprobs=1,
            temperature=1.0
        )

    async def __call__(self, item_stats: Dict[str, List], model: Model) -> Dict[str, List]:
        """
        Generate responses from the LLM for the given input texts.

        Args:
            item_stats (Dict[str, list]): Dictionary containing input texts under the key 'input_texts'.
            model (Model): The language model resource provided by async-graph-bench.

        Returns:
            Dict[str, List]: Dictionary with:
                - 'responses': List of text responses from the model.
                - 'token_lengths': List of token counts for each response.
        """
        input_texts = item_stats["input_texts"]

        # Wrap each prompt in a standard instruction for detailed response
        messages = [
            [
                {
                    "role": "user",
                    "content": (
                            "Please provide your best answer to the following question. "
                            "I do know it requires a lengthy response, but try to get as much into it as possible.\n"
                            + prompt
                    )
                }
            ]
            for prompt in input_texts
        ]

        # Query the model
        response_wrapper = await model.query(messages, generation_params=self.generation_params)
        responses = response_wrapper.get_messages()
        token_lengths = [len(t) for t in response_wrapper.get_tokens()]

        return {"responses": responses, "token_lengths": token_lengths}
