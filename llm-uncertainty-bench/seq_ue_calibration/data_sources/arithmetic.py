from abc import abstractmethod
from typing import List

from async_graph_bench import DataSource
from datasets import load_dataset
from .split_dataframe import split_dataframe


class ArithmeticBaseDatasetProvider(DataSource):
    stats = ["questions", "correct_answer", "unit"]

    def __init__(self, df, subset: tuple = None, limit_items=None):  # TODO remove iterations

        if limit_items is not None:
            df = df.iloc[:limit_items]

        # If a subset tuple is provided, partition the dataset accordingly.
        if subset is not None:
            num_subsets, subset_index = subset
            df = split_dataframe(df, num_subsets, subset_index)

        self.df = df

    def __len__(self):
        return len(self.df)

    def iter_items(self):
        counter = 0
        items = self.df
        has_unit_column = 'unit' in items.columns
        for idx, row in items.iterrows():
            yield {
                "id": row.name,  # (row["id"] if has_id_column else idx)
                "questions": row["question"],
                "correct_answer": row["correct_answer"],
                "unit": row["unit"] if has_unit_column else None,
            }
            counter += 1

    def iter_keys(self):
        for idx, row in self.df.iterrows():
            yield row.name
