from abc import abstractmethod
from typing import List

from datasets import load_dataset, concatenate_datasets
from async_graph_bench import DataSource
from .split_dataframe import split_dataframe


class MultipleChoiceDataSource(DataSource):
    provides = ["questions", "options", "selected_options", "correct_answer"]

    def __init__(self, df, choice_length=4, subset: tuple = None, limit_items=None):
        """
        Initializes the MultipleChoiceDataSource

        Parameters
        ----------
        limit_items : int, optional
            If provided, limits the number of items from the combined dataset.
        subset : tuple of (int, int), optional
            A tuple specifying how to partition the dataset into equal (or nearly equal)
            subsets and which subset to select. The tuple should be of the form (n, i)
            where:
              - n (int): The total number of subsets to divide the dataset into.
              - i (int): The 0-indexed subset to select.

            The dataset is split by computing indices using the formula:
                start = floor(total_items * i / n)
                end   = floor(total_items * (i + 1) / n)
            such that slicing the dataset with [start:end] yields the i-th subset.
            For example, if there are 1001 items and subset is (4, 3), then:
                start = floor(1001 * 3 / 4) = 750
                end   = floor(1001 * 4 / 4) = 1001
            meaning the last subset contains items with indices from 750 to 1000 (inclusive).
            This ensures that all items are included in one of the subsets, even when
            total_items is not perfectly divisible by n.
        """

        self.choice_length = choice_length

        if limit_items is not None:
            df = df.iloc[:limit_items]

        # If a subset tuple is provided, partition the dataset accordingly.
        if subset is not None:
            num_subsets, subset_index = subset
            df = split_dataframe(df, num_subsets, subset_index)

        self.df = df

    def __len__(self):
        return len(self.df) * self.choice_length

    def iter_ids(self):
        items = self.df
        for idx, row in items.iterrows():
            choices = row['choices']
            for current_option in range(len(choices)):
                yield (idx, current_option)

    def iter_items(self):
        items = self.df
        for idx, row in items.iterrows():
            choices = row['choices']
            for current_option in range(len(choices)):
                yield {
                    "id": (idx, current_option),
                    "questions": row['question'],
                    "options": choices,
                    "selected_options": current_option,
                    "correct_answer": row['correct_answer'],
                }
