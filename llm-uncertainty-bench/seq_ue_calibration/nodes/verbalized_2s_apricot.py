from typing import Dict

import numpy as np
from async_graph_bench import Model, GenerationParameters

from .apricot_mc_calc import build_apricot_pre, build_apricot_post


class Verbalized2SApricot:
    """
    Asks model to output its confidence in a provided follow-up prompt and
    extracts the confidence estimate from the model's answer using a provided regex.
    Only usabe for instruct-finetuned models with chat template support.
    Adapted from the original implementation in the paper https://arxiv.org/abs/2305.14975
    """

    requires = ["questions", "options", "selected_options", "assistant_texts", "reasoning_texts"]

    def __init__(
            self,
            confidence_prompt: str,
            max_new_tokens,
            dependency_name="verbalized_2s_answer",
            response_format=None  # {"type": "regex", "regex": "Probability: (1\\.0|0\\.\\d+)$"}
    ):
        self.confidence_prompt = confidence_prompt
        self.max_new_tokens = max_new_tokens
        self.dependency_name = dependency_name
        self.provides = [dependency_name, "verb_reasoning_token_count", "verb_token_count",
                         "verb_assistant_token_count", "verb_token_alts"]
        self.generation_params = GenerationParameters(max_tokens=self.max_new_tokens, logprobs=20, temperature=0.0,
                                                      response_format=response_format)

    async def __call__(self, stats: Dict[str, np.ndarray], model: Model):
        pre_prompts = [
            build_apricot_pre(question, options, selected_option)
            for question, options, selected_option
            in zip(stats["questions"], stats["options"], stats["selected_options"])
        ]
        guesses = [
            (f"Thought Process:\n{reasoning_text}\n\nFinal Answer:\n" if len(reasoning_text) else "") + text
            for reasoning_text, text
            in zip(stats["reasoning_texts"], stats["assistant_texts"])
        ]
        post_prompts = [
            build_apricot_post(options, selected_option)
            for options, selected_option
            in zip(stats["options"], stats["selected_options"])
        ]

        chats = []
        for pre_prompt, guess, post_prompt in zip(pre_prompts, guesses, post_prompts):
            messages = [
                {"role": "user", "content": pre_prompt},
                {"role": "assistant", "content": guess},
                {"role": "user", "content": self.confidence_prompt},
            ]
            chats.append(messages)

        # make Post call
        out = await model.query(chats, generation_params=self.generation_params)
        answers = out.get_assistant_tokens_alternatives()

        return {
            "verb_assistant_token_count": [len(t) for t in out.get_assistant_tokens()],
            "verb_reasoning_token_count": [len(t) for t in out.get_reasoning_tokens()],
            "verb_reasoning_message": out.get_reasoning_messages(),
            "verb_assistant_message": out.get_assistant_messages(),
            self.dependency_name: answers
        }


class VerbalizedArithmetic:
    """
    Asks model to output its confidence in a provided follow-up prompt and
    extracts the confidence estimate from the model's answer using a provided regex.
    Only usabe for instruct-finetuned models with chat template support.
    Adapted from the original implementation in the paper https://arxiv.org/abs/2305.14975
    """
    requires = ["questions", "assistant_texts", "reasoning_texts"]

    def __init__(
            self,
            confidence_prompt: str,
            max_new_tokens: int,
            dependency_name="verbalized_2s_answer",
            response_format=None  # {"type": "regex", "regex": "Probability: (1\\.0|0\\.\\d+)$"}
    ):
        self.dependency_name = dependency_name
        self.provides = [dependency_name, "verb_reasoning_token_count", "verb_token_count",
                         "verb_assistant_token_count", "verb_token_alts"]
        self.confidence_prompt = confidence_prompt
        self.max_new_tokens = max_new_tokens
        self.response_format = response_format
        self.generation_params = GenerationParameters(max_tokens=self.max_new_tokens, logprobs=20, temperature=0,
                                                      response_format=response_format)

    async def __call__(self, stats: Dict[str, np.ndarray], model: Model):
        prompts = [
            question + "\nLet's think step by step."
            for question
            in stats["questions"]
        ]
        guesses = [
            (f"Thought Process:\n{reasoning_text}\n\nFinal Answer:\n" if len(reasoning_text) else "") + text
            for reasoning_text, text
            in zip(stats["reasoning_texts"], stats["assistant_texts"])
        ]
        chats = []
        for prompt, guess in zip(prompts, guesses):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": guess},
                {"role": "user", "content": self.confidence_prompt},
            ]
            chats.append(messages)

        # make Post call
        out = await model.query(chats, generation_params=self.generation_params)
        answers = out.get_assistant_tokens_alternatives()

        return {
            "verb_assistant_token_count": [len(t) for t in out.get_assistant_tokens()],
            "verb_reasoning_token_count": [len(t) for t in out.get_reasoning_tokens()],
            "verb_reasoning_message": out.get_reasoning_messages(),
            "verb_assistant_message": out.get_assistant_messages(),
            self.dependency_name: answers
        }