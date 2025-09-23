import numpy as np

from typing import Dict


class ClaimConditionedProbability:
    dependencies = [
        "assistant_tokens_decoded_alternatives",
        "assistant_tokens_alternatives_nli",
    ]

    def __str__(self):
        return "CCP"

    def _reduce(self, logprobs: list[float]):
        return np.exp(np.sum(logprobs))

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        alternatives = stats["assistant_tokens_decoded_alternatives"]
        alternatives_nli = stats["assistant_tokens_alternatives_nli"]

        prob_nli = []

        for sample_alternatives, sample_alternatives_nli in zip(
                alternatives,
                alternatives_nli,
        ):
            sample_mnlis = []
            for word_alternatives, word_alternatives_nli in zip(
                    sample_alternatives,
                    sample_alternatives_nli
            ):
                entail_logprobs, entail_words = [], []
                contra_logprobs, contra_words = [], []
                for (word_alt, logprob), nli_outcome in zip(word_alternatives, [1] + word_alternatives_nli):
                    if nli_outcome == 1:
                        entail_logprobs.append(logprob)
                        entail_words.append(word_alt)
                    elif nli_outcome == -1:
                        contra_logprobs.append(logprob)
                        contra_words.append(word_alt)
                entail_logprob = np.logaddexp.reduce(entail_logprobs)
                total_logprob = np.logaddexp.reduce(entail_logprobs + contra_logprobs)
                sample_mnlis.append(entail_logprob - total_logprob)
            prob_nli.append(self._reduce(sample_mnlis))
        return np.array(prob_nli)
