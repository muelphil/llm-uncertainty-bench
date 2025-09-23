from datasets import load_dataset

from .arithmetic import ArithmeticBaseDatasetProvider
import pandas as pd


class SciBenchDatasetProvider(ArithmeticBaseDatasetProvider):
    def __init__(self, limit_items=None, subset=None):
        df = load_dataset(
            "xw27/scibench",
            revision="93931252bc1b71d495e67390235940643d926958"
        )["train"].to_pandas()
        unified_df = df.apply(
            lambda row: pd.Series({
                'id': row['problemid'],
                'question': row['problem_text'],
                'correct_answer': float(row['answer_number'].replace(",", "").replace("−", "-")),
                'unit': row['unit'],
            }),
            axis=1
        )

        super().__init__(unified_df, limit_items=limit_items, subset=subset)