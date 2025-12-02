import math
import re
from typing import Dict

import numpy as np

WHITESPACE_TOKENS_REGEX = re.compile(r"[Ġ▁Ċ▂▃\s]")

def extract_text_and_numbers(str):
    return "".join(re.findall(r"[a-zA-Z0-9]", str))

def decode(text: str):
    return text.decode("utf-8", errors="ignore") if isinstance(text, bytes) else text


def remove_whitespace(text: str) -> str:
    return WHITESPACE_TOKENS_REGEX.sub("", text)


def get_token_probabilities(alternatives, tokens, anywhere_okay=False):
    cleaned_alternatives = [[remove_whitespace(decode(token)), math.exp(log_prob)] for token, log_prob in alternatives]
    result = []
    for t in tokens:
        probabilities_for_token = [probability for token, probability in cleaned_alternatives if
                                   (extract_text_and_numbers(token) == t != -1 if anywhere_okay else (token == t))]
        result.append(max(probabilities_for_token) if len(probabilities_for_token) != 0 else 0)

    return result


class ConclusionProbabilityExtractor:
    provides = ["estimations", "token_probs"]

    def __init__(self, dependency_name, valid_answers=None, alternative_equal_or_anywhere_okay_valid_answer="equal",
                 normalize=True):
        self.requires = [dependency_name]
        self.dependency_name = dependency_name
        self.valid_answers = ["Yes", "No"] if valid_answers is None else valid_answers
        self.normalize = normalize

        self.alternative_equal_or_anywhere_okay_valid_answer = alternative_equal_or_anywhere_okay_valid_answer

    async def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        conclusions_alternatives = stats[self.dependency_name]

        certainties = []
        token_probs_list = []
        for alternatives in conclusions_alternatives:
            if len(alternatives) == 0:
                certainties.append(None)
                token_probs_list.append(None)
                continue

            first_alternative = alternatives[0]
            token_probs = get_token_probabilities(first_alternative, self.valid_answers,
                                                  self.alternative_equal_or_anywhere_okay_valid_answer == "anywhere")
            token_probs_list.append(token_probs)
            if sum(token_probs) == 0:
                certainties.append(None)
                continue

            if self.normalize:
                token_probs = np.array(token_probs) / sum(token_probs)

            certainties.append(token_probs[0])

        return {
            "estimations": np.array(certainties),
            "token_probs": token_probs_list
        }
