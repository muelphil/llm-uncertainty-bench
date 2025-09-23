from typing import Dict, List, Iterable

from async_graph_bench import GenerationParameters, Model

from .format_assistant_message import format_assistant_message


def ith_uppercase(i):
    return chr(65 + i)


def flatten(nested):
    if len(nested) == 1:
        return nested[0]
    return [item for sublist in nested for item in sublist]


def get_prompt_post_question_remarks(unit):
    if unit is None:
        return ""
    if len(unit.strip()) == 0:
        return "\nExpress your final answer as a floating-point value in decimal form, without using fractions, functions like \sqrt, or constants like \pi."
    return f"\nExpress your final answer as a floating-point value in decimal form in the unit {unit}, without using fractions, functions like \sqrt, or constants like \pi."


class ArithmeticResponseGenerator:
    dependencies = ["questions", "unit"]
    stats = [
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

    def __init__(self, max_tokens, is_reasoning=False):
        self.max_tokens = max_tokens
        self.generation_params_pre = GenerationParameters(max_tokens=self.max_tokens, logprobs=5, temperature=1.0)
        self.generation_params_post = GenerationParameters(max_tokens=50 if not is_reasoning else 500, logprobs=5,
                                                           temperature=0)

    async def __call__(
            self,
            dependencies: Dict[str, list],
            model: Model
    ) -> Dict[str, List]:
        questions = dependencies["questions"]
        unit = dependencies["unit"]

        # prepare messages
        messages = [[
            {"role": "user",
             "content": questions[i] + (get_prompt_post_question_remarks(unit[i])) + "\nLet's think step by step."}
        ] for i in range(len(questions))]

        # make Pre call
        pre_result = await model.query(messages, generation_params=self.generation_params_pre)

        # prepare post messages
        pre_assistant_messages = pre_result.get_assistant_messages()
        pre_reasoning_messages = ["\n".join(thoughts) for thoughts in pre_result.get_reasoning_messages()]
        for i, (assistant_message, reasoning_message) in enumerate(
                zip(pre_assistant_messages, pre_reasoning_messages)):
            messages[i].append(format_assistant_message(assistant_message, reasoning_message))

            post_prompt = "Output only your final numerical solution for the task! If there is a unit, include it. Do not add words or explanations - only the raw floating point result and unit. If you did not reach any numerical solution, answer <NONE>."
            messages[i].append({"role": "user", "content": post_prompt})

        # make Post call
        post_result = await model.query(messages, generation_params=self.generation_params_post)

        return {
            "assistant_texts": pre_assistant_messages,
            "assistant_tokens_decoded": pre_result.get_assistant_tokens(),
            "assistant_tokens_decoded_alternatives": pre_result.get_assistant_tokens_alternatives(),

            "reasoning_texts": pre_reasoning_messages,
            "reasoning_tokens_decoded": [flatten(t) for t in pre_result.get_reasoning_tokens()],
            "reasoning_tokens_decoded_alternatives": [flatten(alts) for alts in
                                                      pre_result.get_reasoning_tokens_alternatives()],

            "conclusion_texts": post_result.get_assistant_messages(),
            "conclusion_log_probs": post_result.get_assistant_logprobs(),
            "conclusion_tokens_decoded_alternatives": post_result.get_assistant_tokens_alternatives(),

            "conclusion_reasoning_texts": ["\n".join(thoughts) for thoughts in post_result.get_reasoning_messages()],
            "conclusion_reasoning_log_probs": [flatten(t) for t in post_result.get_reasoning_logprobs()],
            "conclusion_reasoning_tokens_decoded_alternatives": [flatten(alts) for alts in
                                                                 post_result.get_reasoning_tokens_alternatives()],

            "finish_reasons": pre_result.get_finish_reasons(),
        }
