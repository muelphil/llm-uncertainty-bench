import hashlib
import random
import string

import pandas as pd
from datasets import load_dataset

from .multiple_choice import MultipleChoiceDataSource


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

class GPQADataSetProvider(MultipleChoiceDataSource):
    def __init__(self, limit_items=None):
        df = load_dataset("Idavidrein/gpqa", "gpqa_main", revision="90b8e5be2b1d3d2dbfe016cdab47981150600c4a")[
            "train"].to_pandas()

        if limit_items is not None:
            df = df.iloc[:limit_items]

        # Transform the dataset to match the expected format
        def transform_row(row):
            choices_list = shuffle_with_hash_seed(
                row['Question'],
                [row['Correct Answer'], row['Incorrect Answer 1'], row['Incorrect Answer 2'], row['Incorrect Answer 3']]
            )
            return pd.Series({
                'id': row.name,
                'question': row['Question'],
                'choices': choices_list,
                'correct_answer': choices_list.index(row['Correct Answer']),
            })

        transformed_df = df.apply(transform_row, axis=1)
        super().__init__(transformed_df, choice_length=4)


def get_uppercase_letter(i: int) -> str:
    return string.ascii_uppercase[i] if 0 <= i < 26 else None


def join_choices(choices):
    return '\n'.join(get_uppercase_letter(idx) + ') ' + choices[idx] for idx in range(len(choices)))
