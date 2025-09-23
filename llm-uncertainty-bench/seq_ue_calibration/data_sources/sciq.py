import hashlib
import random

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


class SciQDataSetProvider(MultipleChoiceDataSource):
    def __init__(self, limit_items=None):
        original_df = load_dataset("allenai/sciq", revision="2c94ad3e1aafab77146f384e23536f97a4849815")[
            "train"].to_pandas()
        if limit_items is not None:
            original_df = original_df.iloc[:limit_items]

        def transform_row(row):
            # Shuffle the choices and get the correct answer index
            choices_list = shuffle_with_hash_seed(
                row['question'],
                [row['correct_answer'], row['distractor1'], row['distractor2'], row['distractor3']]
            )
            correct_answer_index = choices_list.index(row['correct_answer'])

            # Return the transformed row as a dictionary
            return pd.Series({
                'id': row.name,
                'question': row['question'],
                'choices': choices_list,
                'correct_answer': correct_answer_index
            })

        # Apply the transformation function to each row
        unified_df = original_df.apply(transform_row, axis=1)

        super().__init__(unified_df, choice_length=4)
