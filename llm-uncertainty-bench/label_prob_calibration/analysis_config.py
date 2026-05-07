"""Shared configuration for label_prob_calibration analysis notebooks.

Both the data-preparation notebook (analysis_prepare_data) and the
visualization notebook (analysis_plot_data) import from here so that
models, datasets, prompt designs, and file paths are defined in exactly
one place.
"""

import os
import re
from pathlib import Path

from models import MODELS
from datasets_exp1 import datasets, build_mmlu_physics_dataset

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# Attach convenience fields to every model dict.
for model in MODELS:
    model["id"] = model.get("basename", os.path.basename(model["name"]))
    if "basename" not in model:
        model["basename"] = os.path.basename(model["name"])
    if "shortname" not in model:
        model["shortname"] = re.sub(r"(\-\d+|\-v\d.\d)$", "", model["id"])

#: Experiment-1 model subset: all MODELS except the last three.
models = MODELS[:-3]

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

#: Full MMLU DataFrame is loaded once here and reused for MMLU_Physics.
_mmlu_dataset = next(d for d in datasets if d["id"] == "MMLU")
_mmlu_df = _mmlu_dataset["data_source"]().df
_mmlu_dataset["df"] = _mmlu_df

_mmlu_physics_dataset = build_mmlu_physics_dataset(_mmlu_df)

#: Ordered list of all datasets used in the experiment.
all_datasets = [_mmlu_dataset, _mmlu_physics_dataset] + [
    d for d in datasets if d["id"] != "MMLU"
]
for _ds in all_datasets:
    if "df" not in _ds:
        _ds["df"] = _ds["data_source"]().df

#: Lookup from dataset id to dataset dict.
datasets_by_id = {d["id"]: d for d in all_datasets}

# ---------------------------------------------------------------------------
# Prompt designs
# ---------------------------------------------------------------------------

DATA_BASE_PATH = Path(".\\data_struct_dec")

#: Set to [1, 2, 3, 4] to include all four structural-decoding prompt designs.
PROMPT_INDICES = [1]

prompt_designs = [
    {
        "path": DATA_BASE_PATH / f"data_prompt_{i}",
        "label": f"Prompt {i}",
        "idx": i,
        "basename": f"data_prompt_{i}",
    }
    for i in PROMPT_INDICES
]

prompt_basenames = [p["basename"] for p in prompt_designs]

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

RESOURCES_DIR = Path(".\\resources_struct_decoding")
FIGURES_DIR = RESOURCES_DIR / "figures"
TABLES_DIR = RESOURCES_DIR / "tables"

#: Pickled output of the data-preparation notebook; consumed by the visualization notebook.
PREPARED_DATA_PATH = RESOURCES_DIR / "prepared_data.pkl"

# ---------------------------------------------------------------------------
# Performance flags
# ---------------------------------------------------------------------------

#: When True, ``rp.prepare_rel_diagram`` is computed for all four
#: (normalize × chosen_only) calibration variants during the loading loop.
#: Set to False during development to skip the two ``chosen_only=False``
#: variants and roughly halve the relplot bootstrap-CI computation cost.
#: The ``chosen_only=True`` variants (``"norm_chosen"``, ``"raw_chosen"``) are
#: always computed regardless of this flag.
COMPUTE_RELPLOT_FOR_ALL_VARIANTS: bool = True

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def safe_format(val, fmt="4f"):
    """Format *val* as a fixed-point float string, falling back to ``str`` on failure.

    Args:
        val: Value to format.
        fmt: Format specifier suffix, e.g. ``"4f"`` produces ``":.4f"``.

    Returns:
        str: Formatted string.
    """
    try:
        return f"{val:.{fmt}}"
    except (TypeError, ValueError):
        return str(val)
