from typing import List

from datasets import load_dataset

from async_graph_bench import DataSource


class MMLUDataSetProvider(DataSource):
    provides = ["questions", "options"]

    def __init__(self, limit_items=None):
        # Load the entire MMLU dataset (all subjects)
        mmlu = load_dataset("cais/mmlu", "all", revision="c30699e8356da336a370243923dbaf21066bb9fe")["test"].to_pandas()
        if limit_items is not None:
            mmlu = mmlu.iloc[:limit_items]
        mmlu['subject_index'] = mmlu.groupby('subject').cumcount()
        mmlu.rename(columns={'answer': 'correct_answer'}, inplace=True)
        mmlu.set_index(["subject", "subject_index"], inplace=True)
        self.df = mmlu

    def __len__(self):
        return len(self.df)

    async def iter_items(self):
        items = self.df
        for idx, row in items.iterrows():
            yield {
                "id": row.name,  # idx,
                "questions": row['question'],
                "options": row['choices']
            }

    def iter_ids(self):
        for idx, row in self.df.iterrows():
            yield row.name


mmlu_shots = [
    {
        "question": "What is the capital of France?",
        "answer_options": [
            "Berlin",
            "Madrid",
            "Paris",
            "Rome"
        ],
        "correct_answer_index": 2  # C (zero-based index)
    },
    {
        "question": "Which element has the chemical symbol 'O'?",
        "answer_options": [
            "Oxygen",
            "Gold",
            "Silver",
            "Hydrogen"
        ],
        "correct_answer_index": 0  # A
    },
    {
        "question": "Who wrote 'Hamlet'?",
        "answer_options": [
            "Charles Dickens",
            "William Shakespeare",
            "Mark Twain",
            "Jane Austen"
        ],
        "correct_answer_index": 1  # B
    }
]
