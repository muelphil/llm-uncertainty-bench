from async_graph_bench import Model, GenerationParameters
from typing import List, Tuple, Dict


def get_chosen_answer(tokens):
    if tokens and len(tokens[0].strip()) == 1:
        char = tokens[0].strip()
        # Check if the character is an uppercase ASCII letter (between 'A' and 'Z')
        if 65 <= ord(char) <= 90:  # ASCII codes for 'A' (65) to 'Z' (90)
            return ord(char) - 65  # Return the index (0 for 'A', 1 for 'B', ..., 25 for 'Z')
    return None


class MultipleChoiceLabelProbGenerator:
    dependencies = ["questions", "options"]
    stats = [
        "greedy_tokens_decoded",
        "greedy_tokens_decoded_alternatives",
        "chosen_option",
        "finish_reason",
        "messages",
        "token_lengths"
    ]

    def __init__(self, get_prompt, model_type="base", end_of_reasoning_pattern=None, system_prompt=None):
        self.model_type = model_type
        self.max_tokens = 4096 if model_type == "reasoning" else 25
        self.generation_params = GenerationParameters(max_tokens=self.max_tokens, logprobs=20, temperature=0.0)
        self.use_chat_template = model_type != "base"
        self.end_of_reasoning_pattern = end_of_reasoning_pattern
        self.get_prompt = get_prompt
        self.system_prompt = system_prompt

    async def __call__(self, dependencies: Dict[str, list], model: Model) -> Dict[str, list]:
        # "chat_template_kwargs": {"enable_thinking": false} # this may be able to disable the thinking/reasoning of reasoning models

        questions = dependencies["questions"]
        options = dependencies["options"]
        messages = [
            self.get_prompt(question=questions[i], choices=options[i])
            for i in range(len(questions))
        ]
        if self.use_chat_template:
            if self.system_prompt:
                messages = [[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}] for
                            prompt in messages]
            else:
                messages = [[{"role": "user", "content": prompt}] for prompt in messages]

        response_wrapper = await model.query(messages, generation_params=self.generation_params)
        tokens = response_wrapper.get_tokens()
        token_lengths = [len(t) for t in tokens]
        token_alternatives = response_wrapper.get_tokens_alternatives()
        if self.end_of_reasoning_pattern:
            post_reasoning_indices = response_wrapper._get_post_reasoning_indices(self.end_of_reasoning_pattern)
            tokens = [t[idx:] for idx, t in zip(post_reasoning_indices, tokens)]
            token_alternatives = [alt[idx:] for idx, alt in zip(post_reasoning_indices, token_alternatives)]
            # token_alternatives = response_wrapper.get_reasoning_tokens_alternatives()

        assert len(token_alternatives) >= 0, "Response must have at least one token"

        return {
            "messages": response_wrapper.get_messages(),
            "greedy_tokens_decoded": tokens,
            "greedy_tokens_decoded_alternatives": token_alternatives,
            "finish_reason": response_wrapper.get_finish_reasons(),
            "chosen_option": [get_chosen_answer(t) for t in tokens],
            "token_lengths": token_lengths
        }
