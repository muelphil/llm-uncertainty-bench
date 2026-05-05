"""
Dataset registry for Experiment 2 (sequence-level UE calibration).

Exports
-------
mc_datasets_dict : dict
    Ordered dict of multiple-choice datasets, keyed by ``id``.
arithmetic_datasets_dict : dict
    Ordered dict of arithmetic datasets, keyed by ``id``.
datasets_combined : dict
    Merged dict of all datasets (MC first, then arithmetic), keyed by ``id``.
    Each entry has the fields below plus ``id``, ``label``, and
    ``is_arithmetic`` added automatically.
mc_datasets : list
    List view of ``mc_datasets_dict.values()``.
arithmetic_datasets : list
    List view of ``arithmetic_datasets_dict.values()``.
datasets : list
    List view of ``datasets_combined.values()``.

Each entry dict contains:
    id               -- unique string identifier (subdirectory under data_base_path)
    label            -- human-readable display name (underscores replaced by spaces)
    data_source      -- zero-argument callable returning a dataset provider
                        with a ``.df`` attribute (a pandas DataFrame)
    is_arithmetic    -- ``True`` for arithmetic datasets, ``False`` for MC
"""

from data_sources import (
    MMLUDataSetProvider,
    ArcReasoningDataSetProvider,
    SciQDataSetProvider,
    GPQADataSetProvider,
    GSM8KDatasetProvider,
    SVampDatasetProvider,
    SciBenchDatasetProvider,
)

_subsample_size = 250

mc_datasets_dict = {
    "MMLU": {
        "data_source": lambda: MMLUDataSetProvider(limit_items=_subsample_size),
    },
    "ARC_Easy": {
        "data_source": lambda: ArcReasoningDataSetProvider(dataset_name="ARC-Easy", limit_items=_subsample_size),
    },
    "ARC_Challenge": {
        "data_source": lambda: ArcReasoningDataSetProvider(dataset_name="ARC-Challenge", limit_items=_subsample_size),
    },
    "SciQ": {
        "data_source": lambda: SciQDataSetProvider(limit_items=_subsample_size),
    },
    "GPQA": {
        "data_source": lambda: GPQADataSetProvider(limit_items=_subsample_size),
    },
}

arithmetic_datasets_dict = {
    "GSM8K": {
        "data_source": lambda: GSM8KDatasetProvider(limit_items=_subsample_size),
    },
    "SVamp": {
        "data_source": lambda: SVampDatasetProvider(limit_items=_subsample_size),
    },
    "SciBench": {
        "data_source": lambda: SciBenchDatasetProvider(limit_items=_subsample_size),
    },
}

datasets_combined = {**mc_datasets_dict, **arithmetic_datasets_dict}

for _id, _ds in datasets_combined.items():
    _ds["id"] = _id
    _ds["label"] = _id.replace("_", " ")
    _ds["is_arithmetic"] = _id in arithmetic_datasets_dict

mc_datasets = list(mc_datasets_dict.values())
arithmetic_datasets = list(arithmetic_datasets_dict.values())
datasets = list(datasets_combined.values())
