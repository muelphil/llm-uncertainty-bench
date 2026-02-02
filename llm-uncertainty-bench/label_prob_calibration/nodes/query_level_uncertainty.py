import string
from collections import defaultdict
from typing import Dict, Any

from ql_uncertainty import QLUncertainty


def get_uppercase_letter(i: int) -> str:
    assert 0 <= i < 26, "i does not correspond to an uppercase letter, must be 0 <= i < 26"
    return string.ascii_uppercase[i]


def join_choices(choices):
    return '\n'.join(f"{get_uppercase_letter(idx)}) {choice}" for idx, choice in enumerate(choices))


example = \
    """Below you will see a multiple choice question with multiple answers options. Your task is to select the answer choice you believe is correct by responding with the corresponding label: A, B, C, or D.
    Let p = (1, 2, 5, 4)(2, 3) in S_5 . Find the index of <p> in S_5.
    A) 8
    B) 2
    C) 24
    D) 120"""

template = \
    """Below you will see a multiple choice question with multiple answers options. Your task is to select the answer choice you believe is correct by responding with the corresponding label: A, B, C, or D.
    {question}
    {choices}"""


def generate_query(question, options):
    return template.format(question=question, choices=join_choices(options))


class QueryLevelUncertainty:
    requires = ["questions", "options"]
    provides = ['max_prob', 'pd_entropy', 'mink_entropy', 'attn_entropy', 'ppl', 'internal_confidence',
                'internal_confidence_examples']

    def __init__(self, target_tokens):
        # Case: single flat list of numbers → wrap each number
        assert target_tokens is not None, "target_tokens cannot be None"
        if all(isinstance(x, (int, float)) for x in target_tokens):
            self.target_tokens = [[x] for x in target_tokens]
        else:
            self.target_tokens = target_tokens

    async def __call__(self, item_stats: Dict[str, list], model: Any, tokenizer: Any) -> Dict[str, list]:
        questions = item_stats["questions"]
        options = item_stats["options"]
        queries = [generate_query(q, o) for q, o in zip(questions, options)]

        results = defaultdict(list)
        for method in ['max_prob', 'pd_entropy', 'mink_entropy', 'attn_entropy', 'ppl', 'internal_confidence']:
            # It is necessary to have token ids for Yes (7566) and No (2360). Replace target tokens when using different LLMs
            ql_uncertainty = QLUncertainty(model, tokenizer, method=method, target_tokens=self.target_tokens)
            for query in queries:
                score = ql_uncertainty.estimate(query)
                results[method].append(score)

        for query in queries:
            # internal confidence with in-context learning
            ql_uncertainty = QLUncertainty(model, tokenizer, method='internal_confidence',
                                           target_tokens=self.target_tokens)
            examples = [{'query': 'the capital of China is Beijing', 'answer': 'Yes'},
                        {'query': 'the capital of Spain is London', 'answer': 'No'}]
            score = ql_uncertainty.estimate(query, examples)
            results['internal_confidence_examples'].append(score)

        return results
