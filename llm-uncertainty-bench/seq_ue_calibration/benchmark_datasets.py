from data_sources import MMLUDataSetProvider, ArcReasoningDataSetProvider, GSM8KDatasetProvider, SciQDataSetProvider, \
    SVampDatasetProvider, SciBenchDatasetProvider, GPQADataSetProvider

subsample_size = 1000

DATASETS = [
    {
        "id": "MMLU",
        "data_source": lambda: MMLUDataSetProvider(limit_items=subsample_size),
        "is_multiple_choice": True
    },
    {
        "id": "ARC_Easy",
        "data_source": lambda: ArcReasoningDataSetProvider(dataset_name="ARC-Easy", limit_items=subsample_size),
        "is_multiple_choice": True
    },
    {
        "id": "ARC_Challenge",
        "data_source": lambda: ArcReasoningDataSetProvider(dataset_name="ARC-Challenge", limit_items=subsample_size),
        "is_multiple_choice": True
    },
    {
        "id": "SciQ",
        "data_source": lambda: SciQDataSetProvider(limit_items=subsample_size),
        "is_multiple_choice": True
    },
    {
        "id": "GPQA",
        "data_source": lambda: GPQADataSetProvider(limit_items=subsample_size),
        "is_multiple_choice": True
    },
    {
        "id": "GSM8K",
        "data_source": lambda: GSM8KDatasetProvider(limit_items=subsample_size),
        "is_multiple_choice": False
    },
    {
        "id": "SVamp",
        "data_source": lambda: SVampDatasetProvider(limit_items=subsample_size),
        "is_multiple_choice": False
    },
    {
        "id": "SciBench",
        "data_source": lambda: SciBenchDatasetProvider(limit_items=subsample_size),
        "is_multiple_choice": False
    },
]
