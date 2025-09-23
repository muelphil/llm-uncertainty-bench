from typing import Dict

from async_graph_bench.models import Model, GenerationParameters


class QueryModel():
    dependencies = ["input_texts"]
    stats = ["greedy_texts"]

    def __init__(self, max_tokens):
        self.max_tokens = max_tokens
        self.generation_params = GenerationParameters(max_tokens=self.max_tokens, logprobs=0, temperature=1.0)

    async def __call__(self, dependencies: Dict[str, list], model: Model) -> Dict[str, list]:
        input_texts = dependencies["input_texts"]
        messages = [[{"role": "user",
                      "content": "Please provide your best answer to the following question. I do know it requires a lengthy response, but try to get as much into it as possible. " + prompt}]
                    for prompt in input_texts]

        response_wrapper = await model.query(messages, generation_params=self.generation_params)
        responses = response_wrapper.get_messages()

        return {"greedy_texts": responses}
