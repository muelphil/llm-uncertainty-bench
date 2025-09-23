import hashlib
import random
from typing import List

import pandas as pd
from datasets import load_dataset

import string

from async_graph_bench import DataSource


def get_chosen_answer(tokens):
    if tokens and len(tokens[0].strip()) == 1:
        char = tokens[0].strip()
        # Check if the character is an uppercase ASCII letter (between 'A' and 'Z')
        if 65 <= ord(char) <= 90:  # ASCII codes for 'A' (65) to 'Z' (90)
            return ord(char) - 65  # Return the index (0 for 'A', 1 for 'B', ..., 25 for 'Z')
    return None


class GSM8KMCDataSetProvider(DataSource):
    stats = ["questions", "options"]

    def __init__(self, limit_items=None):
        # Load both ARC-Easy and ARC-Challenge datasets

        df = load_dataset("guipenedo/gsm8k-mc", "default", revision="fca1f0a07ad5b0c21fd67fcaec8532a37d9063e2")[
            "train"].to_pandas()
        if limit_items is not None:
            df = df.iloc[:limit_items]

        def transform_row(row):
            # Shuffle the choices and get the correct answer index
            # Return the transformed row as a dictionary
            return pd.Series({
                'id': row.name,
                'question': row['Question'],
                'options': [row['A'], row['B'], row['C'], row['D']],
                'correct_answer': get_chosen_answer(row['Answer']),
            })

        # Apply the transformation function to each row
        self.df = df.apply(transform_row, axis=1)

    def __len__(self):
        return len(self.df)

    async def iter_items(self):
        for idx, row in self.df.iterrows():
            yield {
                "id": row.name,  # (subject, subject_index)
                "questions": row["question"],
                "options": row["options"]
            }

    def iter_keys(self):
        for idx, row in self.df.iterrows():
            yield row.name


def get_uppercase_letter(i: int) -> str:
    return string.ascii_uppercase[i] if 0 <= i < 26 else None


def join_choices(choices):
    return '\n'.join(get_uppercase_letter(idx) + ') ' + choices[idx] for idx in range(len(choices)))


gsm8k_shots = [
    {
        "question": "There are some oranges in a basket. Ana spends 3 minutes peeling an orange and Jane spends 4 minutes doing the same. If Ana and Jane start picking oranges from this basket to peel at the same time, how many more oranges will Ana have peeled than Jane after an hour?",
        "answer_options": [
            "-60",
            "-1",
            "5",
            "2.5"
        ],
        "correct_answer_index": 2  # C (zero-based index)
    },
    {
        "question": "Mark's car breaks down and he needs to get a new radiator. The cost for a new radiator is $400 but he goes to get it at a junk shop and gets it for 80% off. He then hires a mechanic to install it and it takes 3 hours at $50 an hour. How much did he pay?",
        "answer_options": [
            "3",
            "230",
            "49.91",
            "150"
        ],
        "correct_answer_index": 1  # B
    },
    {
        "question": "John invited 20 people to a birthday party. Each guest will eat 2 hot dogs. He already has 4 hot dogs left over from a previous party. If a pack of hot dogs contains 6 hot dogs and costs $2, how much does he need to spend on hot dogs?",
        "answer_options": [
            "4",
            "14",
            "48",
            "12"
        ],
        "correct_answer_index": 3  # D
    }
]
