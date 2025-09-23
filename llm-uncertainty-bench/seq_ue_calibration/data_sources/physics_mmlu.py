from datasets import load_dataset, concatenate_datasets

from .multiple_choice import MultipleChoiceDataSource


class MMLUDataSetProvider(MultipleChoiceDataSource):
    def __init__(self, limit_items=None, subset=None):
        subsets = ["college_physics", "conceptual_physics", "high_school_physics"]
        datasets = [load_dataset("cais/mmlu", subset, revision="c30699e8356da336a370243923dbaf21066bb9fe") for subset in
                    subsets]
        original_df = concatenate_datasets([dataset["test"] for dataset in datasets]).to_pandas()

        unified_df = original_df.rename(columns={'answer': 'correct_answer'})
        unified_df['id'] = original_df.index
        super().__init__(unified_df, choice_length=4, subset=subset, limit_items=limit_items)
