# Extract the first match and parse it into a number
import re
from collections import defaultdict

pattern = r"\d+([,\.]\d+)*"


def extract_number(s):
    match = re.search(pattern, s)
    if match:
        num_str = match.group(0).replace(",", "")  # Remove commas if present
        try:
            return float(num_str) if "." in num_str else int(num_str)
        except ValueError:
            pass
    return None  # Return None if no valid number is found


class AnsweredCorrectlyArithmeticSimple:
    requires = ["sampled_conclusion_texts", "correct_answer", "sampled_assistant_tokens_decoded",
                "sampled_reasoning_tokens_decoded", "finish_reasons", "finish_reasons_post"]
    provides = ["is_correct", "extracted_number", "correct_answer", "answer_token_len", "reasoning_token_len",
                "frequency_of_answer", "finish_reasons", "finish_reasons_post"]

    spread = True

    def __call__(self, deps):
        # for assistant_tokens_decoded in deps["sampled_assistant_tokens_decoded"]:
        result = defaultdict(list)
        for correct_answer, conclusion_text_samples, tokens_samples, reasoning_tokens_samples in zip(
                deps["correct_answer"],
                deps["sampled_conclusion_texts"],
                deps["sampled_assistant_tokens_decoded"],
                deps["sampled_reasoning_tokens_decoded"]
        ):
            extracted_numbers = [extract_number(conclusion_text) for conclusion_text in conclusion_text_samples]
            frequency_of_answer = [(extracted_numbers.count(extracted_number) / len(
                extracted_numbers)) if extracted_number is not None else 0.0
                                   for extracted_number in extracted_numbers]
            is_correct = [correct_answer == extracted for extracted in extracted_numbers]

            result["extracted_number"].append(extracted_numbers)
            result["answer_token_len"].append([len(tokens) for tokens in tokens_samples])
            result["reasoning_token_len"].append([len(tokens) for tokens in reasoning_tokens_samples])
            result["correct_answer"].append(correct_answer)
            result["frequency_of_answer"].append(frequency_of_answer)
            result["is_correct"].append(is_correct)
        result["finish_reasons"] = deps["finish_reasons"]
        result["finish_reasons_post"] = deps["finish_reasons_post"]
        return result
