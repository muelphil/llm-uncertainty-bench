from typing import Literal

import pandas as pd
from datasets import load_dataset

from .multiple_choice import MultipleChoiceDataSource


class ArcReasoningDataSetProvider(MultipleChoiceDataSource):
    def __init__(self, dataset_name: Literal["ARC-Easy", "ARC-Challenge"] = "ARC-Easy", limit_items=None):
        original_df = load_dataset(
            "allenai/ai2_arc",
            dataset_name,
            revision="210d026faf9955653af8916fad021475a3f00453"
        )["train"].to_pandas()

        if limit_items is not None:
            original_df = original_df.iloc[:limit_items]

        unified_df = original_df.apply(
            lambda row: pd.Series({
                'id': row['id'],
                'question': row['question'],
                'choices': row['choices']['text'],
                'correct_answer': list(row['choices']['label']).index(row['answerKey'])
            }),
            axis=1
        )

        super().__init__(unified_df, choice_length=4)
