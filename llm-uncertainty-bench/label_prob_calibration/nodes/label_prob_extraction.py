import math
import numpy as np
from typing import Dict
import re

WHITESPACE_TOKENS_REGEX = re.compile(r"[Ġ▁Ċ▂▃\s]")


def decode(text: str):
    return text.decode("utf-8", errors="ignore") if isinstance(text, bytes) else text


def remove_whitespace(text: str) -> str:
    return WHITESPACE_TOKENS_REGEX.sub("", text)


def get_token_probabilities(alternatives, tokens):
    cleaned_alternatives = [[remove_whitespace(decode(token)), math.exp(log_prob)] for token, log_prob in alternatives]
    result = []
    for t in tokens:
        probabilities_for_token = [probability for token, probability in cleaned_alternatives if token == t]
        result.append(max(probabilities_for_token) if len(probabilities_for_token) != 0 else 0)

    return result


class LabelProbExtractor:
    dependencies = ["greedy_tokens_decoded_alternatives"]

    def __call__(self, stats):
        alternatives_per_choice = stats["greedy_tokens_decoded_alternatives"]

        confidence_per_choice = []
        for alternatives in alternatives_per_choice:
            if len(alternatives) == 0:
                confidence_per_choice.append([0, 0, 0, 0])
            else:
                first_token = alternatives[0]
                confidence_per_choice.append(get_token_probabilities(first_token, ["A", "B", "C", "D"]))

        return {
            "confidence_per_choice": confidence_per_choice
        }
