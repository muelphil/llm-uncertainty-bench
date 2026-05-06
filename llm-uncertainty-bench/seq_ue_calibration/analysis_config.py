"""Shared configuration for seq_ue_calibration analysis notebooks.

Both the data-preparation notebook (analysis_prepare_data) and the
visualization notebook (analysis_plot_data) import from here so that
models, UQ methods, and file paths are defined in exactly one place.
"""

from pathlib import Path
from models import MODELS

# ---------------------------------------------------------------------------
# Models & UQ methods
# ---------------------------------------------------------------------------

models = [
    m for m in MODELS
    if m["type"] in ["instruct", "reasoning"] and "Magistral" not in m["name"]
]

uq_methods = [
    {"label": "Verbalized Uncertainty", "id": "verbalized",          "type": "certainty", "n_bins": 15},
    {"label": "P(True)",                "id": "p_true",              "type": "certainty", "n_bins": 15},
    {"label": "Frequency of Answer",    "id": "frequency_of_answer", "type": "certainty", "n_bins": 11},
    {"label": "CCP",                    "id": "ccp",                 "type": "certainty", "n_bins": 15},
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_BASE_PATH = Path(".\\data")
RESOURCES_DIR  = Path(".\\resources")
FIGURES_DIR    = RESOURCES_DIR / "figures"
TABLES_DIR     = RESOURCES_DIR / "tables"

#: Pickled output of the data-preparation notebook; consumed by the visualization notebook.
PREPARED_DATA_PATH = RESOURCES_DIR / "prepared_data.pkl"
