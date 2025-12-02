import math
import re
from typing import Dict

import numpy as np
from async_graph_bench import GenerationParameters, Model


def decode(text: str):
    return text.decode("utf-8", errors="ignore") if isinstance(text, bytes) else text


WHITESPACE_TOKENS_REGEX = re.compile(r"[Ġ▁Ċ▂▃\s]")


def remove_whitespace(text: str) -> str:
    return WHITESPACE_TOKENS_REGEX.sub("", text)


def get_token_probabilities(alternatives, tokens):
    cleaned_alternatives = [[remove_whitespace(decode(token)), math.exp(log_prob)] for token, log_prob in alternatives]
    result = []
    for t in tokens:
        probabilities_for_token = [probability for token, probability in cleaned_alternatives if token == t]
        result.append(max(probabilities_for_token) if len(probabilities_for_token) != 0 else 0)

    return result


def ith_uppercase(i):
    return chr(65 + i)


# from prompting import scot_prompt

def flatten(nested):
    if len(nested) == 1:
        return nested[0]
    return [item for sublist in nested for item in sublist]


def build_apricot_pre(question, choices, current_choice_index):
    task = question + '\n' + '\n'.join([f"choice {ith_uppercase(i)}: {choice}" for i, choice in enumerate(choices)])
    return "Below you will see a question and answer choices.\n" + task + "\nLet's evaluate choice " + ith_uppercase(
        current_choice_index) + ": " + choices[current_choice_index] + " step by step."


def build_apricot_post(choices, current_choice_index):
    return f"In conclusion, does your last response imply that choice {ith_uppercase(current_choice_index)}: {choices[current_choice_index]} is correct? Do not engage in any further reasoning about the question, respond only based on your previous response. Answer only with Yes or No. Respond with Yes, if your previous response implies choice {ith_uppercase(current_choice_index)} is correct, or No if your response implies choice {ith_uppercase(current_choice_index)} is incorrect."


class MCQAAPriCoTResponseGenerator:
    """
    Calculates generation texts for Open AI model that provides `logprobs` and `top_logprobs`.
    """
    requires = ["questions", "options", "selected_options"]
    provides = [
        "assistant_texts",
        "assistant_tokens_decoded",
        "assistant_tokens_decoded_alternatives",
        "reasoning_texts",
        "reasoning_tokens_decoded",
        "reasoning_tokens_decoded_alternatives",
        "conclusion_texts",
        "conclusion_log_probs",
        "conclusion_tokens_decoded_alternatives",
        "conclusion_reasoning_texts",
        "conclusion_reasoning_log_probs",
        "conclusion_reasoning_tokens_decoded_alternatives",
        "finish_reasons"
    ]

    def __init__(self, max_tokens, is_reasoning=False, max_tokens_post_prompt=4096, system_prompt=None):
        self.generation_params_pre = GenerationParameters(max_tokens=max_tokens, logprobs=5, temperature=1.0)
        # TODO braucht es für generation_params_post logprobs?
        self.generation_params_post = GenerationParameters(max_tokens=1 if not is_reasoning else max_tokens_post_prompt,
                                                           logprobs=5, temperature=0.0)
        self.system_prompt = system_prompt

    async def __call__(
            self,
            item_stats: Dict[str, list],
            model: Model
    ) -> Dict[str, list]:
        """
        Calculates generation texts for Blackbox model on the input batch.

        Parameters:
            item_stats (Dict[str, np.ndarray]): input statistics, holding the "input_texts"
            model (Model): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with List[List[float]] generation texts at 'greedy_texts' key.
        """
        questions = item_stats["questions"]
        options = item_stats["options"]
        selected_options = item_stats["selected_options"]

        # prepare messages
        messages = [[
            *([{"role": "system", "content": self.system_prompt}] if self.system_prompt else []),
            {"role": "user", "content": build_apricot_pre(q, o, s)}
        ] for q, o, s in zip(questions, options, selected_options)]

        # make Pre call
        pre_result = await model.query(messages, generation_params=self.generation_params_pre)

        pre_assistant_messages = pre_result.get_assistant_messages()
        pre_reasoning_messages = ["\n".join(thoughts) for thoughts in pre_result.get_reasoning_messages()]

        return {
            "assistant_texts": pre_assistant_messages,
            "assistant_tokens_decoded": pre_result.get_assistant_tokens(),
            "assistant_tokens_decoded_alternatives": pre_result.get_assistant_tokens_alternatives(),

            "reasoning_texts": pre_reasoning_messages,
            "reasoning_tokens_decoded": [flatten(t) for t in pre_result.get_reasoning_tokens()],
            "reasoning_tokens_decoded_alternatives": [flatten(alts) for alts in
                                                      pre_result.get_reasoning_tokens_alternatives()],
            "finish_reasons": pre_result.get_finish_reasons()
        }


class MCQAAPriCoTResponseGeneratorPost:
    """
    Calculates generation texts for Open AI model that provides `logprobs` and `top_logprobs`.
    """
    requires = [
        "questions",
        "options",
        "selected_options",
        "assistant_texts",
        "reasoning_texts",
        "finish_reasons"
    ]
    provides = [
        "conclusion_texts",
        "conclusion_log_probs",
        "conclusion_tokens_decoded_alternatives",
        "conclusion_reasoning_texts",
        "conclusion_reasoning_log_probs",
        "conclusion_reasoning_tokens_decoded_alternatives",
        "finish_reasons_post",
        "yes_no_probabilities",
        "model_labeled_option_correct"
    ]

    def __init__(self, is_reasoning=False, max_tokens_post_prompt=4096, system_prompt=None):
        self.generation_params_post = GenerationParameters(max_tokens=1 if not is_reasoning else max_tokens_post_prompt,
                                                           logprobs=10, temperature=0.0,
                                                           response_format={"type": "regex", "regex": "(Yes|No)$"})
        self.system_prompt = system_prompt

    async def __call__(
            self,
            item_stats: Dict[str, list],
            model: Model
    ) -> Dict[str, list]:
        """
        Calculates generation texts for Blackbox model on the input batch.

        Parameters:
            item_stats (Dict[str, np.ndarray]): input statistics, holding the "input_texts"
            model (Model): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with List[List[float]] generation texts at 'greedy_texts' key.
        """
        questions = item_stats["questions"]
        options = item_stats["options"]
        selected_options = item_stats["selected_options"]
        pre_assistant_texts = item_stats["assistant_texts"]
        pre_reasoning_texts = item_stats["reasoning_texts"]
        finish_reasons = item_stats["finish_reasons"]

        # prepare messages
        messages = [[
            *([{"role": "system", "content": self.system_prompt}] if self.system_prompt else []),
            {"role": "user", "content": build_apricot_pre(q, o, s)}
        ] for q, o, s in zip(questions, options, selected_options)]

        # prepare post messages
        for i, (assistant_message, reasoning_message, finish_reason) in enumerate(
                zip(pre_assistant_texts, pre_reasoning_texts, finish_reasons)):
            assistant_response = {
                "role": "assistant",
                "content": assistant_message if finish_reason.startswith(
                    "stop") else f"<think>{reasoning_message}</think>\n\n{assistant_message}"
            }
            messages[i].append(assistant_response)
            messages[i].append(
                {"role": "user", "content": build_apricot_post(options[i], selected_options[i])})

        # make Post call
        post_result = await model.query(messages, generation_params=self.generation_params_post)
        finish_reasons_post = post_result.get_finish_reasons()
        conclusion_alternatives_per_response = post_result.get_assistant_tokens_alternatives()

        yes_no_probabilities = []
        model_labeled_option_correct = []
        for conclusion_token_alts, finish_reason in zip(conclusion_alternatives_per_response, finish_reasons_post):
            if finish_reason.startswith("stop") and len(conclusion_token_alts):
                yn_prob = get_token_probabilities(conclusion_token_alts[0], ["Yes", "No"])
                yes_no_probabilities.append(yn_prob)
                if sum(yn_prob) == 0:
                    model_labeled_option_correct.append(None)
                else:
                    model_labeled_option_correct.append(np.argmax(yn_prob) == 0 if sum(yn_prob) != 0 else None)
            else:
                yes_no_probabilities.append(None)
                model_labeled_option_correct.append(None)

        return {
            "conclusion_texts": post_result.get_assistant_messages(),
            "conclusion_log_probs": post_result.get_assistant_logprobs(),
            "conclusion_tokens_decoded_alternatives": conclusion_alternatives_per_response,

            "conclusion_reasoning_texts": ["\n".join(thoughts) for thoughts in post_result.get_reasoning_messages()],
            "conclusion_reasoning_log_probs": [flatten(t) for t in post_result.get_reasoning_logprobs()],
            "conclusion_reasoning_tokens_decoded_alternatives": [flatten(alts) for alts in
                                                                 post_result.get_reasoning_tokens_alternatives()],
            "yes_no_probabilities": yes_no_probabilities,
            "model_labeled_option_correct": model_labeled_option_correct,
            "finish_reasons_post": post_result.get_finish_reasons()
        }
