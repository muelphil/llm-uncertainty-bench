import math
import re
from collections import defaultdict

import numpy as np

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


class AnsweredCorrectlyMC:
    dependencies = ["sampled_conclusion_tokens_decoded_alternatives", "selected_options", "correct_answer",
                    "sampled_assistant_tokens_decoded", "sampled_reasoning_tokens_decoded"]
    stats = ["is_correct", "yes_no_probabilities", "correct_answer_idx", "answer_token_len", "reasoning_token_len",
             "model_guessed_its_correct", "frequency_of_answer"]
    spread = True

    def __call__(self, deps):
        result = defaultdict(list)
        for conclusion_token_alt_samples, correct_ans, evaluated_ans, tokens_samples, reasoning_tokens_samples in zip(
                deps["sampled_conclusion_tokens_decoded_alternatives"],
                deps["correct_answer"],
                deps["selected_options"],
                deps["sampled_assistant_tokens_decoded"],
                deps["sampled_reasoning_tokens_decoded"]
        ):
            yes_no_probabilities = [get_token_probabilities(conclusion_token_alts[0], ["Yes", "No"]) if len(
                conclusion_token_alts) > 0 else None for conclusion_token_alts in conclusion_token_alt_samples]
            model_guessed_its_correct = [np.argmax(yn_prob) == 0 if yn_prob is not None else None for yn_prob in
                                         yes_no_probabilities]
            is_correct = [(correct_ans == evaluated_ans) == guessed_correct if guessed_correct is not None else False
                          for guessed_correct in model_guessed_its_correct]

            frequency_of_answer = [
                (model_guessed_its_correct.count(guessed_correct) / len(model_guessed_its_correct)) if guessed_correct is not None else 0.0 for
                guessed_correct in model_guessed_its_correct
            ]

            result["is_correct"].append(is_correct)
            result["model_guessed_its_correct"].append(model_guessed_its_correct)
            result["yes_no_probabilities"].append(yes_no_probabilities)
            result["correct_answer_idx"].append(correct_ans)
            result["frequency_of_answer"].append(frequency_of_answer)

            result["answer_token_len"].append([len(tokens) for tokens in tokens_samples])
            result["reasoning_token_len"].append([len(tokens) for tokens in reasoning_tokens_samples])
        return result
