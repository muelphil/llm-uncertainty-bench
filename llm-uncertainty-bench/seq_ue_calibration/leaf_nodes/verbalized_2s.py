import numpy as np
import re
from typing import Dict

from async_graph_bench import GenerationParameters, Model


class Verbalized2S:
    """
    Asks model to output it's confidence in a provided follow-up prompt and
    extracts the confidence estimate from the model's answer using a provided regex.
    Only usabe for instruct-finetuned models with chat template support.
    Adapted from the original implementation in the paper https://arxiv.org/abs/2305.14975
    """
    dependencies = ["input_texts", "greedy_texts"]

    def __init__(
            self,
            confidence_prompt: str,
            confidence_regex: str = "",
            max_new_tokens: int = 10,
            name_postfix="",
            system_prompt: str = None,
    ):
        self.max_new_tokens = max_new_tokens
        self.confidence_prompt = confidence_prompt
        self.confidence_regex = confidence_regex
        self.postfix = name_postfix
        self.system_prompt = system_prompt

    def __str__(self):
        return f"Verbalized2S{self.postfix}"

    async def __call__(self, stats: Dict[str, np.ndarray], model: Model) -> np.ndarray:
        chats = []
        prompts = stats["input_texts"]
        guesses = stats["greedy_texts"]
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
        out = await model.query(chats, generation_params=GenerationParameters(temperature=0.0, max_tokens=100))  # TODO
        answers = out.get_messages()

        ues = []
        conf_re = re.compile(self.confidence_regex)
        for answer in answers:
            match = re.search(conf_re, answer)

            try:
                ue = 1 - float(match.groups()[0])
            except AttributeError:
                ue = np.nan

            ues.append(ue)

        return np.array(ues)


class Verbalized2SUEExtractor:
    """
    Asks model to output it's confidence in a provided follow-up prompt and
    extracts the confidence estimate from the model's answer using a provided regex.
    Only usabe for instruct-finetuned models with chat template support.
    Adapted from the original implementation in the paper https://arxiv.org/abs/2305.14975
    """
    dependencies = ["verbalized_2s_answer"]

    def __init__(
            self,
            # confidence_prompt: str,
            confidence_regex: str = "(0(?:\.\d+)?|1(?:\.0+)?)",
            # max_new_tokens: int = 10,
            name_postfix="",
            # system_prompt: str = None,
    ):
        self.confidence_regex = re.compile(confidence_regex)
        self.postfix = name_postfix

    def __str__(self):
        return f"Verbalized2S{self.postfix}"

    async def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        verbalized_2s = stats["verbalized_2s_answer"]
        answers = [
            "".join([alternatives[0][0] for alternatives in alternatives_per_token])
            for alternatives_per_token
            in verbalized_2s
        ]

        certainties = []
        for answer in answers:
            match = re.search(self.confidence_regex, answer)
            try:
                certainty = float(match.groups()[0])
            except AttributeError:
                certainty = None

            certainties.append(certainty)

        return np.array(certainties)
