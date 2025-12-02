from typing import Dict

import numpy as np
from async_graph_bench import Model, GenerationParameters


def ith_uppercase(i):
    return chr(65 + i)


original_ptrue_prompt = """Question: {question}
Proposed Answer: {proposed_answer}
Is the proposed answer:
(A) True
(B) False
The proposed answer is:"""


def build_p_true_prompt(question, choices, current_choice_index, evaluation, model_guessed_its_correct):
    if model_guessed_its_correct is not None:
        correctness_remark = f"\nYou evaluated choice {ith_uppercase(current_choice_index)}: {choices[current_choice_index]} to be {'correct' if model_guessed_its_correct else 'incorrect'}.\n"
    else:
        correctness_remark = ""
    possible_answers = '\n'.join([f"choice {ith_uppercase(i)}: {choice}" for i, choice in enumerate(choices)])
    return f"""Question: {question}
Possible Answers: {possible_answers}
Evaluation of choice {ith_uppercase(current_choice_index)}: {choices[current_choice_index]}:
{evaluation}
{correctness_remark}
Is the proposed evaluation:
(A) True
(B) False
The proposed evaluation is:"""


class OriginalPTrueApricot:
    """
    Asks model to output its confidence in a provided follow-up prompt and
    extracts the confidence estimate from the model's answer using a provided regex.
    Only usabe for instruct-finetuned models with chat template support.
    Adapted from the original implementation in the paper https://arxiv.org/abs/2305.14975
    """

    requires = ["questions", "options", "selected_options", "assistant_texts", "reasoning_texts", "model_labeled_option_correct"]
    provides = ["p_true_original_alternatives", "ptrue_reasoning_token_count", "ptrue_token_count",
                "ptrue_assistant_token_count", "p_true_reasoning_messages", "p_true_assistant_messages"]

    def __init__(self, max_new_tokens: int = 8):
        self.max_new_tokens = max_new_tokens
        self.params = GenerationParameters(max_tokens=self.max_new_tokens,
                                           logprobs=20, response_format={"type": "regex",
                                                                         "regex": "(\\(A\\)?|\\(B\\)?|A|B|A\\)|B\\))"})

    async def __call__(self, stats: Dict[str, np.ndarray], model: Model) -> np.ndarray:
        chats = []
        for question, choices, selected_choice, evaluation, reasoning_text, model_guessed_its_correct \
                in zip(stats["questions"], stats["options"], stats["selected_options"], stats["assistant_texts"],
                       stats["reasoning_texts"], stats["model_labeled_option_correct"]):
            p_true_prompt = build_p_true_prompt(question, choices, selected_choice, (
                f"Thought Process:\n{reasoning_text}\n\nFinal Answer:\n" if len(reasoning_text) else "") + evaluation,
                                                model_guessed_its_correct)
            messages = [
                {"role": "user", "content": p_true_prompt},
                # {"role": "assistant", "content": "("}
            ]
            chats.append(messages)

        out = await model.query(prompt=chats, generation_params=self.params)
        answers = out.get_assistant_tokens_alternatives()

        return {
            "p_true_original_alternatives": answers,
            "p_true_reasoning_messages": out.get_reasoning_messages(),
            "p_true_assistant_messages": out.get_assistant_messages(),

            "ptrue_token_count": [len(t) for t in out.get_reasoning_tokens()],
            "ptrue_reasoning_token_count": [len(t) for t in out.get_reasoning_tokens()],
            "ptrue_assistant_token_count": [len(t) for t in out.get_reasoning_tokens()]
        }


class OriginalPTrueArithmetic:
    """
    Asks model to output its confidence in a provided follow-up prompt and
    extracts the confidence estimate from the model's answer using a provided regex.
    Only usabe for instruct-finetuned models with chat template support.
    Adapted from the original implementation in the paper https://arxiv.org/abs/2305.14975
    """
    requires = ["questions", "assistant_texts", "reasoning_texts"]
    provides = ["p_true_original_alternatives", "ptrue_reasoning_token_count", "ptrue_token_count",
                "ptrue_assistant_token_count", "p_true_reasoning_messages", "p_true_assistant_messages"]

    def __init__(self, max_new_tokens: int = 8):
        self.max_new_tokens = max_new_tokens
        self.params = GenerationParameters(max_tokens=self.max_new_tokens, logprobs=20,
                                           response_format={"type": "regex",
                                                            "regex": "(\\(A\\)?|\\(B\\)?|A|B|A\\)|B\\))"})

    async def __call__(self, stats: Dict[str, np.ndarray], model: Model) -> np.ndarray:
        chats = []
        for question, answer, reasoning_text in zip(stats["questions"], stats["assistant_texts"],
                                                    stats["reasoning_texts"]):
            prompt = original_ptrue_prompt.format(
                question=question,
                proposed_answer=(f"Thought Process:\n{reasoning_text}\n\nFinal Answer:\n" if len(
                    reasoning_text) else "") + answer)
            messages = [
                {"role": "user", "content": prompt},
                # {"role": "assistant", "content": "("}
            ]
            chats.append(messages)

        out = await model.query(chats, generation_params=self.params)
        answers = out.get_assistant_tokens_alternatives()

        return {
            "p_true_original_alternatives": answers,
            "p_true_reasoning_messages": out.get_reasoning_messages(),
            "p_true_assistant_messages": out.get_assistant_messages(),

            "ptrue_token_count": [len(t) for t in out.get_reasoning_tokens()],
            "ptrue_reasoning_token_count": [len(t) for t in out.get_reasoning_tokens()],
            "ptrue_assistant_token_count": [len(t) for t in out.get_reasoning_tokens()]
        }
