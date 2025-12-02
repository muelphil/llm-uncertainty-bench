from typing import Dict, List

from async_graph_bench import GenerationParameters, Model

from .format_assistant_message import format_assistant_message


def ith_uppercase(i):
    return chr(65 + i)


def flatten(nested):
    if len(nested) == 1:
        return nested[0]
    return [item for sublist in nested for item in sublist]


def get_prompt_post_question_remarks(unit):
    return (f"\nExpress your final answer as a floating-point value in decimal form" +
            (f" in the unit {unit}" if len(unit.strip()) != 0 else "") +
            ", without using fractions, functions like \\sqrt, or constants like \\pi.")


class ArithmeticResponseGeneratorPre:
    requires = ["questions", "unit"]
    provides = [
        "assistant_texts",
        "assistant_tokens_decoded",
        "assistant_tokens_decoded_alternatives",
        "reasoning_texts",
        "reasoning_tokens_decoded",
        "reasoning_tokens_decoded_alternatives",
        "finish_reasons"
    ]

    def __init__(self, max_tokens, is_reasoning=False, system_prompt=None):
        self.max_tokens = max_tokens
        self.generation_params_pre = GenerationParameters(max_tokens=self.max_tokens, logprobs=5, temperature=1.0)
        self.generation_params_post = GenerationParameters(max_tokens=50 if not is_reasoning else 500, logprobs=5,
                                                           temperature=0)
        self.system_prompt = system_prompt

    async def __call__(
            self,
            item_stats: Dict[str, list],
            model: Model
    ) -> Dict[str, List]:
        questions = item_stats["questions"]
        unit = item_stats["unit"]

        # prepare messages
        messages = [[
            *([{"role": "system", "content": self.system_prompt}] if self.system_prompt else []),
            {"role": "user",
             "content": (questions[i] +
                         # only bring up unit remarks for SciBench
                         (get_prompt_post_question_remarks(unit[i]) if unit[i] is not None else "") +
                         "\nLet's think step by step.")}
        ] for i in range(len(questions))]

        # make Pre call
        pre_result = await model.query(messages, generation_params=self.generation_params_pre)

        # prepare post messages
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
            "finish_reasons": pre_result.get_finish_reasons(),
        }


class ArithmeticResponseGeneratorPost:
    requires = ["questions", "unit", "assistant_texts", "reasoning_texts"]
    provides = [
        "conclusion_texts",
        "conclusion_log_probs",
        "conclusion_tokens_decoded_alternatives",
        "conclusion_reasoning_texts",
        "conclusion_reasoning_log_probs",
        "conclusion_reasoning_tokens_decoded_alternatives",
        "finish_reasons_post"
    ]

    def __init__(self, is_reasoning=False):
        self.generation_params_post = GenerationParameters(max_tokens=256 if not is_reasoning else 2048, logprobs=5,
                                                           temperature=0)

    async def __call__(
            self,
            item_stats: Dict[str, list],
            model: Model
    ) -> Dict[str, List]:
        questions = item_stats["questions"]
        unit = item_stats["unit"]

        # prepare messages
        messages = [[
            {"role": "user",
             "content": (questions[i] +
                         (get_prompt_post_question_remarks(unit[i]) if unit[
                                                                           i] is not None else "") +  # only bring up unit remarks for SciBench
                         "\nLet's think step by step.")}
        ] for i in range(len(questions))]

        # prepare post messages
        pre_assistant_messages = item_stats["assistant_texts"]
        pre_reasoning_messages = item_stats["reasoning_texts"]
        for i, (assistant_message, reasoning_message) in enumerate(
                zip(pre_assistant_messages, pre_reasoning_messages)):
            messages[i].append(format_assistant_message(assistant_message, reasoning_message))

            post_prompt = "Extract and return the final numerical result you obtained! If there is a unit, include it. Do not add words, commentary or explanations - only the raw floating point result and unit. Do **not** engage in any further reasoning about the question, the answer, rounding or anything else. Output **only the raw number and unit**. If you did not reach any numerical solution, output <NONE> instead."
            messages[i].append({"role": "user", "content": post_prompt})

        # make Post call
        post_result = await model.query(messages, generation_params=self.generation_params_post)

        return {
            "conclusion_texts": post_result.get_assistant_messages(),
            "conclusion_log_probs": post_result.get_assistant_logprobs(),
            "conclusion_tokens_decoded_alternatives": post_result.get_assistant_tokens_alternatives(),

            "conclusion_reasoning_texts": ["\n".join(thoughts) for thoughts in post_result.get_reasoning_messages()],
            "conclusion_reasoning_log_probs": [flatten(t) for t in post_result.get_reasoning_logprobs()],
            "conclusion_reasoning_tokens_decoded_alternatives": [flatten(alts) for alts in
                                                                 post_result.get_reasoning_tokens_alternatives()],

            "finish_reasons_post": post_result.get_finish_reasons(),
        }
