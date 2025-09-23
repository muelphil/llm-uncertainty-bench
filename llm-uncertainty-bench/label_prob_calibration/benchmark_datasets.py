from data_sources import arc_reasoning_shots, ArcReasoningDataSetProvider, MMLUDataSetProvider, mmlu_shots, \
    GSM8KMCDataSetProvider, gsm8k_shots, GPQADataSetprovider, gpqa_shots

datasets = [
    {
        "id": "MMLU",
        "data_source": MMLUDataSetProvider,
        "shots": mmlu_shots,
        "n": 14042
    },
    {
        "id": "ArcReasoning",
        "data_source": ArcReasoningDataSetProvider,
        "shots": arc_reasoning_shots,
        "n": 3358
    },
    {
        "id": "GSM8KMC",
        "data_source": GSM8KMCDataSetProvider,
        "shots": gsm8k_shots,
        "n": 7468
    },
    {
        "id": "GPQA",
        "data_source": GPQADataSetprovider,
        "shots": gpqa_shots,
        "n": 448
    },
]
