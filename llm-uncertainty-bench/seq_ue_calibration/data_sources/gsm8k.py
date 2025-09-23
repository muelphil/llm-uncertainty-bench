import pandas as pd
import re
from datasets import load_dataset

from .arithmetic import ArithmeticBaseDatasetProvider


def extract_correct_result(str):
    match = re.search(r'####\s([^\s]*)$', str)
    return float(match.group(1).replace(',', ''))  # Remove commas for numbers like 1,080


class GSM8KDatasetProvider(ArithmeticBaseDatasetProvider):
    def __init__(self, limit_items=None, subset=None):
        df = load_dataset(
            "openai/gsm8k",
            "main",
            revision="e53f048856ff4f594e959d75785d2c2d37b678ee"
        )["train"].to_pandas()
        unified_df = df.apply(
            lambda row: pd.Series({
                'id': row.name,
                'question': row['question'],
                'correct_answer': extract_correct_result(row['answer']),
            }),
            axis=1
        )
        super().__init__(unified_df, limit_items=limit_items, subset=subset)
