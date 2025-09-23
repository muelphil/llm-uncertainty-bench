import hashlib
import random
from typing import List

import pandas as pd
from datasets import load_dataset

import string

from async_graph_bench import DataSource


def shuffle_with_hash_seed(string, array):
    """
    Shuffle an array deterministically based on a hash of a given string.

    This function takes a string and an array as inputs, computes a deterministic hash
    of the string using SHA-256, and uses this hash as the seed for the `random.shuffle`
    function. This ensures that the shuffling is consistent across different machines
    and runtimes, as long as the inputs remain the same.

    Args:
        string (str): The input string used to seed the shuffling process.
        array (list): The list to be shuffled.

    Returns:
        list: A new list containing the elements of the input array in shuffled order.

    Notes:
        - The function is deterministic. Given the same string and array inputs,
          it will produce the same output on different machines and in different
          Python runtimes.
        - The original input array remains unmodified; a copy of the array is shuffled
          and returned.
        - The SHA-256 hash function is used to ensure a stable and reproducible hash
          value.

    Example:
        >>> shuffle_with_hash_seed("example", [1, 2, 3, 4])
        [3, 1, 4, 2]  # Output will always be the same for "example" and [1, 2, 3, 4].
    """
    hash_value = int(hashlib.sha256(string.encode()).hexdigest(), 16)
    random.seed(hash_value)
    shuffled = array[:]
    random.shuffle(shuffled)
    return shuffled


class GPQADataSetprovider(DataSource):
    stats = ["questions", "options"]

    def __init__(self, limit_items=None):
        # Load both ARC-Easy and ARC-Challenge datasets

        df = load_dataset("Idavidrein/gpqa", "gpqa_main", revision="90b8e5be2b1d3d2dbfe016cdab47981150600c4a")[
            "train"].to_pandas()
        if limit_items is not None:
            df = df.iloc[:limit_items]

        def transform_row(row):
            # Shuffle the choices and get the correct answer index
            choices_list = shuffle_with_hash_seed(
                row['Question'],
                [row['Correct Answer'], row['Incorrect Answer 1'], row['Incorrect Answer 2'], row['Incorrect Answer 3']]
            )
            correct_answer_index = choices_list.index(row['Correct Answer'])

            # Return the transformed row as a dictionary
            return pd.Series({
                'id': row.name,
                'question': row['Question'],
                'options': choices_list,
                'correct_answer': correct_answer_index
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


# For the 3-shot prompting, questions from gpqa_extended were used that were not in gpqa_main
# from datasets import load_dataset
# df_main = load_dataset("Idavidrein/gpqa", "gpqa_main", revision="90b8e5be2b1d3d2dbfe016cdab47981150600c4a")["train"].to_pandas()
# df_extended = load_dataset("Idavidrein/gpqa", "gpqa_extended", revision="90b8e5be2b1d3d2dbfe016cdab47981150600c4a")["train"].to_pandas()
# questions_in_main = set(df_main["Question"].to_list())
# extended_questions = df_extended[df_extended.apply(lambda e: e["Question"] not in questions_in_main, axis=1)]
# extended_questions


gpqa_shots = [
    {
        "question": "(3R,4S)-1,2,3,4-tetramethylcyclobut-1-ene and dimethylacetylene dicarboxylate are heated together in a sealed tube. What is the molecular symmetry group of the product formed by this reaction?",
        "answer_options": [
            "C2v",
            "Cs",
            "C2",
            "C1"
        ],
        "correct_answer_index": 2  # C (zero-based index)
    },
    {
        "question": "What would be the ionization energy of Uranium with atomic number 92, if all the electrons were bosons? Assume that an electron feels the nuclear charge shielded by the charge of half the other electrons in the shell.",
        "answer_options": [
            "22.7 keV",
            "28.8 keV",
            "39.2 keV",
            "35.1 keV"
        ],
        "correct_answer_index": 1  # B
    },
    {
        "question": "Identify the compound C10H13NO2 using the given data.\n1H-NMR: δ 1.14 (d, 6H), 3.6 (bs, 1H) 4.17 (sept, 1H), 6.92 (d, 2H), 7.78 (d, 2H), 11.67 (s, 1H)\nIR: Medium intensity band at 3330 cm-1, strong broad band centered at 3000 cm-1, and a strong band at 1765 cm-1.",
        "answer_options": [
            "3-(ethylamino)benzoic acid",
            "Isopropyl 4-aminobenzoate",
            "4-Isopropoxybenzamide",
            "4-(isopropylamino)benzoic acid"
        ],
        "correct_answer_index": 3  # D
    }
]
