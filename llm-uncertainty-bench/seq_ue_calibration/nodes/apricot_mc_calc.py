from typing import List, Dict, Tuple

from async_graph_bench import GenerationParameters, Model
from .format_assistant_message import format_assistant_message


# from prompting import scot_prompt

def flatten(nested):
    if len(nested) == 1:
        return nested[0]
    return [item for sublist in nested for item in sublist]


def ith_uppercase(i):
    return chr(65 + i)


def build_apricot_pre(question, choices, current_choice_index):
    task = question + '\n' + '\n'.join([f"choice {ith_uppercase(i)}: {choice}" for i, choice in enumerate(choices)])
    return "Below you will see a question and answer choices.\n" + task + "\nLet's evaluate choice " + ith_uppercase(
        current_choice_index) + ": " + choices[current_choice_index] + " step by step."


def build_apricot_post(choices, current_choice_index):
    return f"In conclusion, do you believe choice {ith_uppercase(current_choice_index)}: {choices[current_choice_index]} is most correct? Answer only with Yes or No"


class MCQAAPriCoTResponseGenerator:
    """
    Calculates generation texts for Open AI model that provides `logprobs` and `top_logprobs`.
    """
    dependencies = ["questions", "options", "selected_options"]
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
        # TODO braucht es für generation_params_post logprobs?
        self.generation_params_post = GenerationParameters(max_tokens=1 if not is_reasoning else 500, logprobs=5,
                                                           temperature=0.0)

    async def __call__(
            self,
            dependencies: Dict[str, list],
            model: Model
    ) -> Dict[str, list]:
        """
        Calculates generation texts for Blackbox model on the input batch.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, holding the "input_texts"
            model (Model): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with List[List[float]] generation texts at 'greedy_texts' key.
        """
        questions = dependencies["questions"]
        options = dependencies["options"]
        selected_options = dependencies["selected_options"]

        # prepare messages
        messages = [[
            {"role": "user", "content": build_apricot_pre(questions[i], options[i], selected_options[i])}
        ] for i in range(len(questions))]

        # make Pre call
        pre_result = await model.query(messages, generation_params=self.generation_params_pre)

        # prepare post messages
        pre_assistant_messages = pre_result.get_assistant_messages()
        pre_reasoning_messages = ["\n".join(thoughts) for thoughts in pre_result.get_reasoning_messages()]
        for i, (assistant_message, reasoning_message) in enumerate(zip(pre_assistant_messages, pre_reasoning_messages)):
            messages[i].append(format_assistant_message(assistant_message, reasoning_message))

            messages[i].append(
                {"role": "user", "content": build_apricot_post(options[i], selected_options[i])})

        # make Post call
        post_result = await model.query(messages, generation_params=self.generation_params_post)

        # if not all(len(m) > 0 for m in pre_assistant_messages): # this only happened in 1/200 cases - that is fine!
        #     for i, m in enumerate(pre_assistant_messages):
        #         if len(m) == 0:
        #             print(
        #                 f"Assistant message was empty for the following full message: '{pre_result.get_messages()[i]}'")
        #     raise AssertionError("Not all messages were generated, empty messages in " + str(
        #         sum(len(m) == 0 for m in pre_assistant_messages)) + f"of {len(pre_assistant_messages)} cases")
        # assert all(len(m) > 0 for m in
        #            post_result.get_assistant_tokens_alternatives()), "Not all token alternatives for conclusion are available, missing in " + str(
        #     sum(len(m) == 0 for m in
        #         post_result.get_assistant_tokens_alternatives())) + f" of {len(post_result.get_assistant_tokens_alternatives())} cases"

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

        # assert len({len(v) for v in result.values()}) == 1, "Not all properties have the same length! " + str(
        #     {key: len(val) for key, val in result.items()})
        #
        # return result
