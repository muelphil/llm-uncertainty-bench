from typing import List

import pandas as pd
from datasets import load_dataset

from async_graph_bench import DataSource


# Removed rows due to length of options not 4
# [('easy', 289), ('easy', 690), ('easy', 753), ('easy', 1094), ('easy', 1138),
# ('easy', 1530), ('easy', 1591), ('easy', 1732), ('easy', 1941), ('easy', 2051),
# ('challenge', 461), ('challenge', 626)]

class ArcReasoningDataSetProvider(DataSource):
    stats = ["questions", "options"]

    def __init__(self, limit_items=None):
        # Load both ARC-Easy and ARC-Challenge datasets
        datasets = []
        for category in ["ARC-Easy", "ARC-Challenge"]:
            df = load_dataset("allenai/ai2_arc", category, revision="210d026faf9955653af8916fad021475a3f00453")["train"].to_pandas()
            if limit_items is not None:
                df = df.iloc[:limit_items]

            df = df.assign(
                subject=category[4:].lower(),
                subject_index=df.index  # Directly use index as subject_index
            )

            df["options"] = df["choices"].apply(lambda x: x["text"])
            df["correct_answer"] = df.apply(
                lambda row: list(row["choices"]["label"]).index(row["answerKey"]), axis=1
            )

            df = df[df["options"].map(len) == 4]  # len 4: 3358, len 3: 7, len 5: 5
            df = df[["question", "options", "correct_answer", "subject", "subject_index"]]
            datasets.append(df)

        # Merge both datasets
        self.df = pd.concat(datasets)
        self.df.set_index(["subject", "subject_index"], inplace=True)

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


arc_reasoning_shots = [
    {
        "question": "How are the particles in a block of iron affected when the block is melted?",
        "answer_options": [
            "The particles gain mass.",
            "The particles contain less energy.",
            "The particles move more rapidly.",
            "The particles increase in volume."
        ],
        "correct_answer_index": 2  # C (zero-based index)
    },
    {
        "question": "Biological evolution can occur through all of these except",
        "answer_options": [
            "competition.",
            "fossilization.",
            "variation.",
            "adaptation."
        ],
        "correct_answer_index": 1  # B
    },
    {
        "question": "Which of these objects will most likely float in water?",
        "answer_options": [
            "glass marble",
            "steel ball",
            "hard rubber ball",
            "table tennis ball"
        ],
        "correct_answer_index": 3  # D
    }
]
