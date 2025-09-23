from datasets import load_dataset

from .arithmetic import ArithmeticBaseDatasetProvider
import pandas as pd


class SVampDatasetProvider(ArithmeticBaseDatasetProvider):
    def __init__(self, limit_items=None, subset=None):
        df = load_dataset(
            "ChilleD/SVAMP",
            revision="5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef"
        )["train"].to_pandas()
        unified_df = df.apply(
            lambda row: pd.Series({
                'id': row['ID'],
                'question': row['question_concat'],
                'correct_answer': int(row['Answer'])  # all SVAMP problem solutions are integers
            }),
            axis=1
        )

        super().__init__(unified_df, limit_items=limit_items, subset=subset)
