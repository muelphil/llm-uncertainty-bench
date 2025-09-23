import numpy as np
import re
from typing import Dict

import re
import math

WHITESPACE_TOKENS_REGEX = re.compile(r"[Ġ▁Ċ▂▃\s]")


def decode(text: str):
    return text.decode("utf-8", errors="ignore") if isinstance(text, bytes) else text


def remove_whitespace(text: str) -> str:
    return WHITESPACE_TOKENS_REGEX.sub("", text)


def get_token_probabilities(alternatives, tokens, startswith_okay=False):
    cleaned_alternatives = [[remove_whitespace(decode(token)), math.exp(log_prob)] for token, log_prob in alternatives]
    result = []
    for t in tokens:
        probabilities_for_token = [probability for token, probability in cleaned_alternatives if
                                   (token.startswith(t) if startswith_okay else (token == t))]
        result.append(max(probabilities_for_token) if len(probabilities_for_token) != 0 else 0)

    return result


class ConclusionProbabilityExtractor:
    def __init__(self, dependency_name, valid_answers=None, alternative_equal_or_startswith_valid_answer="equal",
                 normalize=True):
        self.dependencies = [dependency_name]
        self.dependency_name = dependency_name
        self.valid_answers = ["Yes", "No"] if valid_answers is None else valid_answers
        self.normalize = normalize

        self.alternative_equal_or_startswith_valid_answer = alternative_equal_or_startswith_valid_answer

    async def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        conclusions_alternatives = stats[self.dependency_name]

        certainties = []
        for alternatives in conclusions_alternatives:
            if len(alternatives) == 0:
                certainties.append(None)
                continue

            first_alternative = alternatives[0]
            token_probs = get_token_probabilities(first_alternative, self.valid_answers,
                                                  self.alternative_equal_or_startswith_valid_answer == "startswith")
            if sum(token_probs) == 0:
                certainties.append(None)
                continue

            if self.normalize:
                token_probs = np.array(token_probs) / sum(token_probs)
            certainties.append(token_probs[0])

        return np.array(certainties)
