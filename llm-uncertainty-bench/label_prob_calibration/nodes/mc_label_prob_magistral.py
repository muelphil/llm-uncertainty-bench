from typing import Dict

from async_graph_bench import Model, GenerationParameters


def get_chosen_answer(tokens):
    if tokens and len(tokens[0].strip()) == 1:
        char = tokens[0].strip()
        # Check if the character is an uppercase ASCII letter (between 'A' and 'Z')
        if 65 <= ord(char) <= 90:  # ASCII codes for 'A' (65) to 'Z' (90)
            return ord(char) - 65  # Return the index (0 for 'A', 1 for 'B', ..., 25 for 'Z')
    return None


def adjust_thinking(message):
    index = message.find("[/THINK]")
    if index != -1:
        message = message[:index]
    index = max(message.rfind("."), message.rfind("!"))
    if index != -1:
        message = message[:index - 1]
    message += ". I should now respond with the single label A,B,C or D associated with the answer I consider most correct.[/THINK]"
    return message


class MultipleChoiceLabelProbGeneratorMagistral:
    requires = ["questions", "options"]
    provides = [
        "greedy_tokens_decoded",
        "greedy_tokens_decoded_alternatives",
        "chosen_option",
        "finish_reason",
        "messages",
        "amount_answer_tokens"
    ]

    def __init__(self, get_prompt, system_prompt, max_tokens=4096, regex="[ABCD]$"):
        self.max_tokens = max_tokens
        self.generation_params_think = GenerationParameters(max_tokens=self.max_tokens - 10, logprobs=0,
                                                            temperature=0.0,
                                                            response_format={"type": "regex", "regex": "^$"})
        self.generation_params_answer = GenerationParameters(max_tokens=5, logprobs=20, temperature=0.0,
                                                             response_format={"type": "regex", "regex": regex})
        self.get_prompt = get_prompt
        self.system_prompt = system_prompt

    async def __call__(self, item_stats: Dict[str, list], model: Model) -> Dict[str, list]:
        # "chat_template_kwargs": {"enable_thinking": false} # this may be able to disable the thinking/reasoning of reasoning models

        questions = item_stats["questions"]
        options = item_stats["options"]
        messages = [
            self.get_prompt(question=questions[i], choices=options[i])
            for i in range(len(questions))
        ]
        messages = [[
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ] for prompt in messages]

        response_wrapper_think = await model.query(messages, generation_params=self.generation_params_think)
        think_messages = response_wrapper_think.get_messages()
        think_messages = [adjust_thinking(tm) for tm in think_messages]
        for m, tm in zip(messages, think_messages):
            m.append({"role": "assistant", "content": tm})
        response_wrapper_answer = await model.query(messages, generation_params=self.generation_params_answer)

        tokens = response_wrapper_answer.get_tokens()
        token_lengths = [len(t) for t in response_wrapper_think.get_tokens()]
        token_alternatives = response_wrapper_answer.get_assistant_tokens_alternatives()
        answer_messages = response_wrapper_answer.get_messages()
        for tas in token_alternatives:
            assert len(tas) >= 0, "Response must have at least one token"

        return {
            "messages": [tm + m for tm, m in zip(think_messages, answer_messages)],
            "greedy_tokens_decoded": tokens,
            "greedy_tokens_decoded_alternatives": token_alternatives,
            "finish_reason": response_wrapper_answer.get_finish_reasons(),
            "chosen_option": [get_chosen_answer(t) for t in tokens],
            "amount_answer_tokens": token_lengths
        }
