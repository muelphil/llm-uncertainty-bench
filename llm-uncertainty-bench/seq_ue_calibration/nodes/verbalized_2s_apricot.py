from typing import Dict

import numpy as np
from async_graph_bench import Model, GenerationParameters

from .apricot_mc_calc import build_apricot_pre, build_apricot_post
from .format_assistant_message import format_assistant_message


class Verbalized2SApricot:
    """
    Asks model to output its confidence in a provided follow-up prompt and
    extracts the confidence estimate from the model's answer using a provided regex.
    Only usabe for instruct-finetuned models with chat template support.
    Adapted from the original implementation in the paper https://arxiv.org/abs/2305.14975
    """

    dependencies = ["questions", "options", "selected_options", "assistant_texts", "reasoning_texts"]

    def __init__(
            self,
            confidence_prompt: str,
            max_new_tokens: int = 15,
            system_prompt: str = None,
            dependency_name="verbalized_2s_answer"
    ):
        self.system_prompt = system_prompt
        self.confidence_prompt = confidence_prompt
        self.max_new_tokens = max_new_tokens
        self.dependency_name = dependency_name
        self.stats = [dependency_name]

    async def __call__(self, stats: Dict[str, np.ndarray], model: Model):

        pre_prompts = [
            build_apricot_pre(question, options, selected_option)
            for question, options, selected_option
            in zip(stats["questions"], stats["options"], stats["selected_options"])
        ]
        guesses = [
            (f"Though Process:\n{reasoning_text}\n\nFinal Answer:\n" if len(reasoning_text) else "") + text
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
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.extend([
                {"role": "user", "content": pre_prompt},
                {"role": "assistant", "content": guess},
                # {"role": "user", "content": post_prompt},
                # {"role": "assistant", "content": conclusion},
                {"role": "user", "content": self.confidence_prompt},
            ])
            chats.append(messages)

        # make Post call
        generation_params = GenerationParameters(max_tokens=self.max_new_tokens, logprobs=20, temperature=0.0)
        out = await model.query(chats, generation_params=generation_params)
        answers = out.get_assistant_tokens_alternatives()

        return {
            self.dependency_name: answers
        }


class VerbalizedArithmetic:
    """
    Asks model to output its confidence in a provided follow-up prompt and
    extracts the confidence estimate from the model's answer using a provided regex.
    Only usabe for instruct-finetuned models with chat template support.
    Adapted from the original implementation in the paper https://arxiv.org/abs/2305.14975
    """
    dependencies = ["questions", "assistant_texts", "reasoning_texts"]

    def __init__(
            self,
            confidence_prompt: str,
            max_new_tokens: int = 15,
            system_prompt: str = None,
            dependency_name="verbalized_2s_answer"
    ):
        self.system_prompt = system_prompt
        self.dependency_name = dependency_name
        self.stats = [dependency_name]
        self.confidence_prompt = confidence_prompt
        self.max_new_tokens = max_new_tokens

    async def __call__(self, stats: Dict[str, np.ndarray], model: Model):
        prompts = [
            question + "\nLet's think step by step."
            for question
            in stats["questions"]
        ]
        guesses = [
            (f"Though Process:\n{reasoning_text}\n\nFinal Answer:\n" if len(reasoning_text) else "") + text
            for reasoning_text, text
            in zip(stats["reasoning_texts"], stats["assistant_texts"])
        ]
        chats = []
        for prompt, guess in zip(prompts, guesses):
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.extend([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": guess},
                {"role": "user", "content": self.confidence_prompt},
            ])
            chats.append(messages)

        # make Post call
        generation_params = GenerationParameters(max_tokens=self.max_new_tokens, logprobs=20, temperature=0)
        out = await model.query(chats, generation_params=generation_params)
        answers = out.get_assistant_tokens_alternatives()
        return {self.dependency_name: answers}
