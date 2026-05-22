"""
Dataset registry for Experiment 1 (label probability calibration).

Each entry in ``datasets`` is a dict with:
    id            -- unique string identifier used as the display name and
                     as the subdirectory name under every prompt data path,
                     *except* when ``data_path_id`` is set.
    data_source   -- zero-argument callable returning a dataset provider
                     with a ``.df`` attribute (a pandas DataFrame).
    data_path_id  -- (optional) the subdirectory name to use when loading
                     benchmark results from disk.  Defaults to ``id``.
                     Set to ``"MMLU"`` for MMLU_Physics so result files are
                     read from the MMLU directory.
    shots         -- few-shot examples for the dataset prompt.
    n             -- approximate total number of items in the dataset.

MMLU_Physics is not included in ``datasets``; build it at runtime from the
already-loaded MMLU DataFrame via :func:`build_mmlu_physics_dataset`.

Example usage::

    from datasets_exp1 import datasets, build_mmlu_physics_dataset
    mmlu_df = datasets[0]["data_source"]().df
    mmlu_physics = build_mmlu_physics_dataset(mmlu_df)
    all_datasets = datasets[:1] + [mmlu_physics] + datasets[1:]
"""

from label_prob_calibration.data_sources import (
    ArcReasoningDataSetProvider,
    arc_reasoning_shots,
    MMLUDataSetProvider,
    mmlu_shots,
    GSM8KMCDataSetProvider,
    gsm8k_shots,
    GPQADataSetprovider,
    gpqa_shots,
)

MMLU_PHYSICS_SUBSETS = ["college_physics", "conceptual_physics", "high_school_physics"]

datasets = [
    {
        "id": "MMLU",
        "data_source": MMLUDataSetProvider,
        "shots": mmlu_shots,
        "n": 14042,
    },
    {
        "id": "ArcReasoning",
        "data_source": ArcReasoningDataSetProvider,
        "shots": arc_reasoning_shots,
        "n": 3358,
    },
    {
        "id": "GSM8KMC",
        "data_source": GSM8KMCDataSetProvider,
        "shots": gsm8k_shots,
        "n": 7468,
    },
    {
        "id": "GPQA",
        "data_source": GPQADataSetprovider,
        "shots": gpqa_shots,
        "n": 448,
    },
]


def build_mmlu_physics_dataset(mmlu_df):
    """Build the MMLU_Physics dataset dict from an already-loaded MMLU DataFrame.

    Filters the full MMLU DataFrame to the three physics subject areas
    (college, conceptual, and high-school physics) and returns a dataset dict
    compatible with the Experiment 1 dataset registry.

    ``data_path_id`` is set to ``"MMLU"`` so result files are looked up under
    the MMLU benchmark directory rather than a non-existent MMLU_Physics one.

    Args:
        mmlu_df: The pandas DataFrame returned by ``MMLUDataSetProvider().df``.

    Returns:
        dict: Dataset registry entry for MMLU_Physics.
    """
    df_physics = mmlu_df[mmlu_df.index.get_level_values(0).isin(MMLU_PHYSICS_SUBSETS)]
    return {
        "id": "MMLU_Physics",
        "data_path_id": "MMLU",
        "df": df_physics,
        "shots": mmlu_shots,
        "n": len(df_physics),
    }
