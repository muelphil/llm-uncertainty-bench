import numpy as np
import re
from typing import Dict

from async_graph_bench import GenerationParameters, Model

class Verbalized2SUEExtractor:
    """
    Asks model to output it's confidence in a provided follow-up prompt and
    extracts the confidence estimate from the model's answer using a provided regex.
    Only usabe for instruct-finetuned models with chat template support.
    Adapted from the original implementation in the paper https://arxiv.org/abs/2305.14975
    """
    requires = ["verbalized_2s_answer"]

    def __init__(
            self,
            confidence_regex: str = "(0(?:\.\d+)?|1(?:\.0+)?)",
            name_postfix="",
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
