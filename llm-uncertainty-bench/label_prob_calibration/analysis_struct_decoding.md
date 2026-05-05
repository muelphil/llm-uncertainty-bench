---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Experiment 1 – Label Probability Calibration

This notebook analyses **token-level uncertainty** in multiple-choice QA: how well
the label-choice probabilities assigned by a model reflect its actual likelihood of
being correct. The pipeline is:

1. Load raw benchmark results (one CSV per prompt × dataset × model) into `raw_data`.
2. Align predictions with ground-truth answers and compute calibration metrics into `cal_data`.
3. Visualise calibration grids, the normalisation effect, and label-choice bias.

# Imports

```python
%load_ext
autoreload
%autoreload
2

import os
import pickle
import re
import statistics
from collections import defaultdict
from itertools import product

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from async_graph_bench import CSVDataStore
from matplotlib.collections import PatchCollection
from matplotlib.colors import to_rgb
from sklearn.metrics import roc_auc_score
```

```python
import sys
from pathlib import Path

project_root = Path().resolve().parent
sys.path.insert(0, str(project_root / "calibration_visualization"))
sys.path.insert(0, str(project_root / "shared"))
sys.path.insert(0, str(Path().resolve()))
```

```python
from calculate_calibration_data import calculate_calibration_data
from ece import calculate_ece
from normalized_entropy import calculate_normalized_entropy
from plot_calibration_curve import plot_calibration_curve

from config import CALIBRATION_PLOT_COLORS, apply_matplotlib_defaults
from datasets_exp1 import datasets, build_mmlu_physics_dataset
from generate_grid_plot import generate_grid_plot
from models import MODELS
from plot_empty import plot_empty
```

# Configuration

Apply global plotting defaults and create the output directory tree.

```python
apply_matplotlib_defaults()

resources_dir = Path(".\\resources_struct_decoding")
figures_dir = resources_dir / "figures"
tables_dir = resources_dir / "tables"

for d in [resources_dir, figures_dir, tables_dir, figures_dir / "full_plots"]:
    os.makedirs(d, exist_ok=True)


def safe_format(val, fmt="4f"):
    """Format *val* as a float string; fall back to str on failure."""
    try:
        return f"{val:.{fmt}}"
    except (TypeError, ValueError):
        return str(val)
```

# Model Selection

Attach `id`, `basename`, and `shortname` convenience fields to every model dict, then
restrict to the subset used in this experiment (all models except the last three).

```python
for model in MODELS:
    model["id"] = model.get("basename", os.path.basename(model["name"]))
    if "basename" not in model:
        model["basename"] = os.path.basename(model["name"])
    if "shortname" not in model:
        model["shortname"] = re.sub(r"(\-\d+|\-v\d.\d)$", "", model["id"])

models = MODELS[:-3]
```

# Dataset Setup

Load the MMLU DataFrame once, then derive the physics-only subset from it.
All other datasets are loaded via their `data_source` callable. A `df` field is
stored in each dataset dict so the same DataFrame is reused throughout the notebook
without re-reading disk.

```python
mmlu_dataset = next(d for d in datasets if d["id"] == "MMLU")
mmlu_df = mmlu_dataset["data_source"]().df
mmlu_dataset["df"] = mmlu_df

mmlu_physics_dataset = build_mmlu_physics_dataset(mmlu_df)
all_datasets = [mmlu_dataset, mmlu_physics_dataset] + [d for d in datasets if d["id"] != "MMLU"]

for dataset in all_datasets:
    if "df" not in dataset:
        dataset["df"] = dataset["data_source"]().df

datasets_by_id = {d["id"]: d for d in all_datasets}
[d["id"] for d in all_datasets]
```

# Prompt Design

Define the prompt variants included in this analysis. Set `prompt_indices` to
`[1, 2, 3, 4]` to run all four structural-decoding prompt designs.

```python
base_path = Path(".\\data_struct_dec")

prompt_indices = [1]  # extend to [1, 2, 3, 4] to include all prompt variants
prompt_designs = [
    {
        "path": base_path / f"data_prompt_{i}",
        "label": f"Prompt {i}",
        "idx": i,
        "basename": f"data_prompt_{i}",
    }
    for i in prompt_indices
]
prompt_basenames = [p["basename"] for p in prompt_designs]
```

# Prompt Example

Render a sample prompt to illustrate the structured-decoding input format.

```python
from data_sources import get_prompt_3
from IPython.display import display, HTML

dataset_ex = datasets_by_id["MMLU"]
item = dataset_ex["df"].iloc[10]
prompt_text = get_prompt_3(item["question"], item["choices"], dataset_ex["shots"])
display(HTML(f"<pre>{prompt_text}</pre>"))
```

# Data Loading

`load_label_prob_df` reads the raw benchmark output (one CSV per cell) from disk.
All CSVs are loaded here into the nested dict `raw_data` so that no subsequent
section ever re-reads disk.

`raw_data[prompt_basename][dataset_id][model_id]` → `pd.DataFrame | None`

```python
def load_label_prob_df(prompt, dataset, model):
    """Load the raw label-probability CSV for one (prompt, dataset, model) triplet.

    Uses ``dataset['data_path_id']`` (if present) as the on-disk subdirectory name
    so MMLU_Physics results are read from the MMLU benchmark directory rather than
    a non-existent MMLU_Physics one.

    Args:
        prompt:  Prompt design dict with a ``path`` key.
        dataset: Dataset dict from the experiment-1 registry.
        model:   Model dict with an ``id`` key.

    Returns:
        pd.DataFrame or None: Raw model output, or None if the file is missing.
    """
    disk_id = dataset.get("data_path_id", dataset["id"])
    result_path = prompt["path"] / disk_id / model["id"]
    try:
        return CSVDataStore(result_path, "LabelProbExtractor").to_dataframe()
    except Exception as e:
        print(f"  [load] missing: {result_path} – {e}")
        return None
```

```python
raw_data = {}
for prompt in prompt_designs:
    pname = prompt["basename"]
    raw_data[pname] = {}
    for dataset in all_datasets:
        ds_id = dataset["id"]
        raw_data[pname][ds_id] = {}
        for model in models:
            m_id = model["id"]
            print(f"Loading {pname}/{ds_id}/{m_id}")
            raw_data[pname][ds_id][m_id] = load_label_prob_df(prompt, dataset, model)
```

```python
# Optional: persist raw_data to disk for fast reloads
with open("./cached_raw_data.pkl", "wb") as f:
    pickle.dump(raw_data, f)
```

# Data Transformation

Compute calibration metrics from the loaded DataFrames.

`align_predictions_with_ground_truth` aligns each CSV row with the ground-truth
answer and collects the model's chosen confidence value.
`compute_calibration_metrics` orchestrates alignment, calibration-curve binning,
and label-probability summary statistics into a single metrics dict.

`cal_data[norm_key][prompt_basename][dataset_id][model_id][label_key]` → metrics dict

```python
def align_predictions_with_ground_truth(ground_truth_df, label_prob_df,
                                        chosen_only=True, normalize=True):
    """Align model label-probability rows with ground-truth answers.

    Iterates over ``label_prob_df``, skipping rows whose ``id`` has no match in
    ``ground_truth_df``.  Per row, optionally L1-normalises the probability vector
    and collects either the single argmax confidence (``chosen_only=True``) or all
    per-label confidences (``chosen_only=False``).

    Args:
        ground_truth_df: Ground-truth DataFrame indexed by item id, with a
                         ``correct_answer`` column (integer label index).
        label_prob_df:   Model output DataFrame with ``id`` and
                         ``confidence_per_choice`` columns.
        chosen_only:     If True, record only the argmax choice per question.
        normalize:       If True, L1-normalise each per-choice probability vector.

    Returns:
        tuple:
            correct (list[bool]):    Whether each recorded prediction is correct.
            certainties (list[float]): Corresponding confidence values.
            accuracy (float):        Fraction of all rows where argmax == correct label.
            invalid_count (int):     Rows where all probabilities are zero
                                     (only counted when ``chosen_only`` is True).
    """
    correct, certainties = [], []
    n_correct = 0
    invalid_count = 0

    for _, row in label_prob_df.iterrows():
        if row["id"] not in ground_truth_df.index:
            continue
        mc_task = ground_truth_df.loc[row["id"]]
        probs = np.array(row["confidence_per_choice"], dtype=float)
        if normalize and probs.sum() > 0:
            probs /= probs.sum()
        chosen = int(np.argmax(probs)) if probs.sum() > 0 else None

        if chosen is not None and chosen == mc_task.correct_answer:
            n_correct += 1

        if chosen_only:
            if chosen is None:
                invalid_count += 1
            else:
                correct.append(chosen == mc_task.correct_answer)
                certainties.append(probs[chosen])
        else:
            for i, p in enumerate(probs):
                correct.append(i == mc_task.correct_answer)
                certainties.append(p)

    accuracy = n_correct / len(label_prob_df) if len(label_prob_df) > 0 else 0.0
    return correct, certainties, accuracy, invalid_count
```

```python
def compute_calibration_metrics(ground_truth_df, label_prob_df,
                                chosen_only=True, normalize=True, n_bins=15):
    """Compute the full calibration metrics dict for one (prompt, dataset, model) cell.

    Combines alignment, calibration-curve binning, and label-probability summary
    statistics into a single dict consumed by the visualisation layer.

    Args:
        ground_truth_df: Ground-truth DataFrame (see :func:`align_predictions_with_ground_truth`).
        label_prob_df:   Raw model output DataFrame.
        chosen_only:     Use only the argmax label per question.
        normalize:       L1-normalise confidence vectors before computing metrics.
        n_bins:          Number of calibration histogram bins.

    Returns:
        dict: Keys include ``bin_confidences``, ``bucket_accuracies``,
              ``bucket_counts``, ``ece``, ``auroc``, ``accuracy``,
              ``invalid_answers``, ``total_items``, ``normalized_entropy``,
              and label-probability summary statistics.
    """
    correct, certainties, accuracy, invalid_count = align_predictions_with_ground_truth(
        ground_truth_df, label_prob_df, chosen_only=chosen_only, normalize=normalize
    )
    bin_confidences, bucket_accuracies, bucket_counts = calculate_calibration_data(
        correct, certainties, n_bins
    )
    ece = calculate_ece(bin_confidences, bucket_accuracies, bucket_counts)
    try:
        auroc = roc_auc_score(correct, certainties)
    except Exception:
        auroc = "undefined"

    # Per-question sum-of-label-probabilities statistics (clipped to [0, 1])
    sum_series = (
        label_prob_df["confidence_per_choice"]
        .apply(sum)
        .apply(lambda v: max(0.0, min(1.0, v)))
    )
    metrics = {
        "is_normalized": normalize,
        "bin_confidences": bin_confidences,
        "bucket_accuracies": bucket_accuracies,
        "bucket_counts": bucket_counts,
        "ece": ece,
        "auroc": auroc,
        "accuracy": accuracy,
        "label_prob_sum_mean": sum_series.mean(),
        "label_prob_sum_median": sum_series.quantile(0.5),
        "label_prob_sum_std": sum_series.std(),
        "label_prob_sum_iqr": sum_series.quantile(0.75) - sum_series.quantile(0.25),
        "normalized_entropy": calculate_normalized_entropy(bucket_counts),
        "total_items": len(label_prob_df),
        "invalid_answers": invalid_count,
    }
    if certainties:
        metrics["label_prob_median"] = np.median(certainties)
        metrics["label_prob_iqr"] = (
                np.percentile(certainties, 75) - np.percentile(certainties, 25)
        )
        metrics["average_certainty"] = np.mean(certainties)
    else:
        metrics["label_prob_median"] = "undefined"
        metrics["label_prob_iqr"] = "undefined"
        metrics["average_certainty"] = "undefined"
    return metrics
```

Compute calibration metrics for all combinations of normalisation mode, prompt,
dataset, model, and label selection. The loop reads exclusively from `raw_data`
and `all_datasets["df"]`; no additional file reads occur here.

```python
cal_data = {}

for norm_key, normalize in [("normalized", True), ("non-normalized", False)]:
    cal_data[norm_key] = {pname: {ds["id"]: {} for ds in all_datasets}
                          for pname in prompt_basenames}
    for prompt, dataset, model in product(prompt_designs, all_datasets, models):
        pname, ds_id, m_id = prompt["basename"], dataset["id"], model["id"]
        df = raw_data[pname][ds_id][m_id]
        if df is None:
            cal_data[norm_key][pname][ds_id][m_id] = {"chosen_labels": None, "all_labels": None}
            continue
        cal_data[norm_key][pname][ds_id][m_id] = {}
        for label_key, chosen_only in [("chosen_labels", True), ("all_labels", False)]:
            print(f"Computing {norm_key}/{pname}/{ds_id}/{m_id}/{label_key}")
            try:
                cal_data[norm_key][pname][ds_id][m_id][label_key] = compute_calibration_metrics(
                    dataset["df"], df, chosen_only=chosen_only, normalize=normalize
                )
            except Exception as e:
                print(f"  ERROR: {e}")
                cal_data[norm_key][pname][ds_id][m_id][label_key] = None
```

```python
# Optional: persist cal_data to disk for fast reloads
with open("./cached_data_struct.pkl", "wb") as f:
    pickle.dump(cal_data, f)
```

# Data Visualization

## Calibration Subplot Helpers

`make_calibration_subplot_fn` returns a closure matching the
`plot_subplot_fn(ax, data, model_type)` signature expected by `generate_grid_plot`.
`render_metrics_table` draws a two-column summary-statistics table on a given Axes.

```python
def make_calibration_subplot_fn(ece_in_plot=False):
    """Return a grid-cell rendering function for calibration curves.

    Args:
        ece_in_plot: If True, annotate the curve with the ECE value.

    Returns:
        Callable matching ``plot_subplot_fn(ax, data, model_type)``.
    """

    def render_calibration_subplot(ax, data, model_type):
        plot_calibration_curve(
            data["bin_confidences"], data["bucket_accuracies"], data["bucket_counts"],
            ax=ax, colormap=CALIBRATION_PLOT_COLORS[model_type],
            fontsize=14, tick_fontsize=10,
            xlabel="Confidence Bins", ylabel="Accuracy in Bin",
            ece=data["ece"] if ece_in_plot else None,
        )

    return render_calibration_subplot


def render_metrics_table(ax, data):
    """Render a per-subplot summary-statistics table on *ax*.

    Displays ECE, AUROC, normalised entropy, invalid-answer count, accuracy,
    and label-probability statistics in a two-column matplotlib table.

    Args:
        ax:   The matplotlib Axes to draw on.
        data: Calibration metrics dict (output of :func:`compute_calibration_metrics`).
    """
    table_data = [
        ["ECE", safe_format(data["ece"])],
        ["AUROC", safe_format(data["auroc"])],
        ["Norm. Entropy of Bucket Counts", safe_format(data["normalized_entropy"])],
        ["Invalid Answer Count", f"{data['invalid_answers']}/{data['total_items']}"],
        ["Accuracy", safe_format(data["accuracy"])],
        ["Median of Label Probabilities", safe_format(data["label_prob_median"])],
        ["IQR of Label Probabilities", safe_format(data["label_prob_iqr"])],
        ["Median of Sum of Label Probabilities", safe_format(data["label_prob_sum_median"])],
        ["IQR of Sum of Label Probabilities", safe_format(data["label_prob_sum_iqr"])],
    ]
    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (_, col), cell in table.get_celld().items():
        if col == 0:
            cell.PAD = 0.02
            cell.get_text().set_ha("right")
            cell.set_width(0.65)
        elif col == 1:
            cell.PAD = 0.04
            cell.get_text().set_ha("left")
            cell.set_width(0.35)
    ax.axis("off")
```

## Full Calibration Grid

`build_calibration_grid` reads pre-computed `cal_data` and assembles a
dataset × model calibration grid for one (prompt, normalisation, label) combination.
`save_all_calibration_grids` iterates over all combinations, saves each grid to SVG,
and closes the figure to free memory.

```python
def build_calibration_grid(prompt, datasets, models, chosen_only, normalize,
                           with_table, skip_annotation=False, with_title=True,
                           ece_in_plot=False, **kwargs):
    """Build a dataset × model calibration grid for one configuration.

    Reads pre-computed metrics from the module-level ``cal_data`` dict.

    Args:
        prompt:           Prompt design dict.
        datasets:         List of dataset dicts (grid rows).
        models:           List of model dicts (grid columns).
        chosen_only:      If True, use ``chosen_labels``; otherwise ``all_labels``.
        normalize:        If True, use the ``normalized`` norm-key.
        with_table:       If True, render a metrics table below each curve.
        skip_annotation:  If True, omit row/column title annotations.
        with_title:       If True, add a descriptive plot title.
        ece_in_plot:      If True, annotate each curve with its ECE value.
        **kwargs:         Forwarded to :func:`generate_grid_plot`.

    Returns:
        The matplotlib.pyplot module after rendering.
    """
    norm_key = "normalized" if normalize else "non-normalized"
    label_key = "chosen_labels" if chosen_only else "all_labels"
    cell_data = [
        [
            (cal_data[norm_key][prompt["basename"]][ds["id"]][m["id"]][label_key],
             m["type"])
            for m in models
        ]
        for ds in datasets
    ]
    plot_title = (
        f"Calibration of "
        f"{'Normalized' if normalize else 'Non-Normalized'} "
        f"Label Probabilities for {prompt['label']} "
        f"{'(Most Probable Label per Question Only)' if chosen_only else '(All Labels)'}"
    ) if with_title else None
    return generate_grid_plot(
        cell_data,
        row_titles=[ds["id"] for ds in datasets],
        col_titles=[m["shortname"] for m in models],
        plot_subplot_fn=make_calibration_subplot_fn(ece_in_plot=ece_in_plot),
        plot_table_fn=render_metrics_table if with_table else None,
        with_table=with_table,
        skip_annotation=skip_annotation,
        plot_title=plot_title,
        **kwargs,
    )
```

```python
def save_all_calibration_grids(prompt_designs, datasets, models):
    """Save calibration grid SVGs for all prompt × normalisation × label combinations.

    Each file is named
    ``cal_plot_prompt{idx}_table{0|1}_chosenonly{0|1}_norm{0|1}_mmlu_physics.svg``
    and written to `figures_dir / `full_plots/``.

    Args:
        prompt_designs: List of prompt design dicts.
        datasets:       List of dataset dicts (grid rows).
        models:         List of model dicts (grid columns).
    """
    for prompt in prompt_designs:
        for chosen_only, with_table, normalize in product([True, False], repeat=3):
            tag = (
                f"cal_plot_prompt{prompt['idx']}"
                f"_table{int(with_table)}"
                f"_chosenonly{int(chosen_only)}"
                f"_norm{int(normalize)}"
            )
            print(f"Generating {prompt['basename']}/{tag}.svg")
            try:
                plot = build_calibration_grid(
                    prompt, datasets, models,
                    chosen_only=chosen_only, normalize=normalize,
                    with_table=with_table, with_title=True,
                    row_col_titles_font_size=26,
                )
                plot.savefig(
                    figures_dir / "full_plots/{tag}_mmlu_physics.svg",
                    bbox_inches="tight",
                )
                plot.close()
            except Exception as e:
                print(f"  Failed: {e}")
```

Generate and save all calibration grid variants.

```python
save_all_calibration_grids(prompt_designs, all_datasets, models)
```

## Paper Header Plot

A focused calibration grid used as the paper header figure, showing three
representative datasets and four key models with ECE annotations.

```python
plot = build_calibration_grid(
    prompt_designs[0],
    datasets=[d for d in all_datasets if d["id"] in ["MMLU", "GSM8KMC", "GPQA"]],
    models=[m for m in models if any(name in m["basename"] for name in
                                     ["Mistral-Small-3", "Magistral-Small-2507-Reasoning-Enabled",
                                      "Llama-3", "Qwen3-30B-A3B"])],
    chosen_only=True, normalize=True,
    with_table=False, with_title=False,
    ece_in_plot=True,
    row_col_titles_font_size=32,
)
plot.savefig(figures_dir / "prompt_1_header_plot.svg", bbox_inches="tight")
plot.show()
```

## Normalization Effect

Side-by-side calibration grids for one model showing the effect of applying
L1-normalisation to the raw token probabilities.

```python
def plot_normalization_comparison(model, datasets, prompt):
    """Plot calibration curves without vs with L1 probability normalisation.

    Rows correspond to the two normalisation modes; columns to datasets.
    The figure is saved to
    `figures_dir / `normalization_effect_prompt{idx}_{model_id}.svg``.

    Args:
        model:    Model dict to visualise.
        datasets: List of dataset dicts (one column per dataset).
        prompt:   Prompt design dict.
    """
    cell_data = [
        [
            (cal_data["normalized" if normalize else "non-normalized"]
             [prompt["basename"]][ds["id"]][model["id"]]["chosen_labels"],
             model["type"])
            for ds in datasets
        ]
        for normalize in [False, True]
    ]
    plot = generate_grid_plot(
        cell_data,
        row_titles=["Without Normalization", "With Normalization"],
        col_titles=[ds["id"] for ds in datasets],
        plot_subplot_fn=make_calibration_subplot_fn(ece_in_plot=False),
        sharex=False, sharey=False, with_table=False,
        show_axis_labels=True, plot_title=None,
        row_col_titles_font_size=28,
    )
    plot.savefig(
        figures_dir / "normalization_effect_prompt{prompt['idx']}_{model['id']}.svg",
        bbox_inches="tight",
    )
    plot.show()
```

```python
plot_normalization_comparison(
    next(m for m in models if "Mistral-Small-3.1-24B-Base-2503" in m["id"]),
    all_datasets,
    prompt_designs[0],
)
```

# Label Bias Analysis

Compute how often each answer-choice position (A/B/C/D) is assigned the highest
probability by the model, aggregated across all datasets. This reveals systematic
positional biases that are independent of correctness.

Only the first prompt design is used here. Label-choice distributions are derived
from `raw_data` so no additional CSV reads are needed.

```python
first_prompt = prompt_designs[0]

# Accumulate normalised probability mass assigned to each label index per model
labels_chosen_count = defaultdict(lambda: defaultdict(float))

for model, dataset in product(models, all_datasets):
    m_id, ds_id = model["id"], dataset["id"]
    df = raw_data[first_prompt["basename"]][ds_id][m_id]
    if df is None:
        continue
    for _, row in df.iterrows():
        estimations = np.array(row["confidence_per_choice"], dtype=float)
        if estimations.sum() != 0:
            estimations /= estimations.sum()
        else:
            estimations[:] = 0.25  # uniform fallback for all-zero rows
        for i, v in enumerate(estimations):
            labels_chosen_count[m_id][i] += v

# Normalise accumulated counts to proportions (must sum to 1 per model)
labels_chosen_proportion = {}
for m_id, counts in labels_chosen_count.items():
    total = sum(counts.values())
    labels_chosen_proportion[m_id] = {label: counts[label] / total for label in range(4)}
    assert abs(1 - sum(labels_chosen_proportion[m_id].values())) < 1e-6, (
        f"label proportions do not sum to 1 for {m_id}"
    )
```

Compute the ground-truth correct-answer distribution across all datasets to serve
as a reference baseline for the label-bias comparison.

```python
actual_correct_label_count = defaultdict(int)
for dataset in all_datasets:
    for label, count in dataset["df"]["correct_answer"].value_counts().items():
        actual_correct_label_count[label] += count

total_gt = sum(actual_correct_label_count.values())
actual_correct_label_proportion = {i: actual_correct_label_count[i] / total_gt for i in range(4)}
print("Ground-truth label distribution:", actual_correct_label_proportion)
```

# Label Probability Sum

Compute the mean and standard deviation of the per-question sum of label
probabilities for each (model, prompt) pair from pre-computed `cal_data`.
The sum is expected to be ≈ 1 for well-calibrated models; instruction-tuned
models often show strong probability mass polarisation.

```python
# plot_data[model_id][prompt_basename] = (mean, std)
model_ids = [m["id"] for m in models]
dataset_id_first = all_datasets[0]["id"]

plot_data = {m_id: {} for m_id in model_ids}
for prompt, model in product(prompt_designs, models):
    pname, m_id = prompt["basename"], model["id"]
    entry = cal_data["normalized"][pname][dataset_id_first][m_id].get("chosen_labels")
    plot_data[m_id][pname] = (
        (entry["label_prob_sum_mean"], entry["label_prob_sum_std"]) if entry else (0, 0)
    )
```

Grouped bar chart showing mean label-probability sum per model and prompt design.
Each model type (base / instruct / reasoning) uses a distinct colour family; lighter
shades distinguish later prompts.

```python
class MulticolorPatch:
    def __init__(self, colors):
        self.colors = colors


class MulticolorPatchHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        width, height = handlebox.width, handlebox.height
        patches = [
            mpatches.Rectangle(
                [width / 2 * i - handlebox.xdescent, -handlebox.ydescent],
                width / 2, height,
                facecolor=c, edgecolor="black",
            )
            for i, c in enumerate(orig_handle.colors)
        ]
        patch_collection = PatchCollection(patches, match_original=True)
        handlebox.add_artist(patch_collection)
        return patch_collection


def lighten_color(color, amount=0.5):
    """Blend a matplotlib colour towards white by the given amount."""
    c = np.array(to_rgb(color))
    return np.clip(c + (1 - c) * amount, 0, 1)


shades = [0.0, 0.2, 0.4, 0.6]
prompt_colors = {
    prompt["basename"]: {
        "instruct": lighten_color("tab:blue", amount=shades[i]),
        "base": lighten_color("tab:orange", amount=shades[i]),
        "reasoning": lighten_color("tab:green", amount=shades[i]),
    }
    for i, prompt in enumerate(prompt_designs)
}
```

```python
n_models = len(model_ids)
n_prompts = len(prompt_designs)
width = 0.2
x = np.arange(n_models)

fig, ax = plt.subplots(figsize=(20, 6))

for j, prompt in enumerate(prompt_designs):
    offset = (j - (n_prompts - 1) / 2) * width
    for i, model in enumerate(models):
        mean, std = plot_data[model["id"]][prompt["basename"]]
        xpos = x[i] + offset
        c = prompt_colors[prompt["basename"]][model["type"]]
        ax.bar(xpos, mean, width, yerr=std, capsize=3, color=c, edgecolor="black",
               label=prompt["basename"] if i == 0 else "")

ax.set_xticks(x)
ax.set_xticklabels(model_ids, rotation=45, ha="right", fontsize=14)
ax.set_ylabel("Label Prob Sum Mean", fontsize=16)
ax.set_yticks(np.arange(0.0, 1.2, 0.2))
ax.set_yticklabels([f"{v:.1f}" for v in np.arange(0.0, 1.2, 0.2)], fontsize=14)
ax.set_ylim(0, 1.2)
ax.set_axisbelow(True)
ax.grid(axis="y", color="lightgrey", linestyle="--")
ax.grid(axis="y", which="minor", color="lightgrey", linestyle="--")
ax.set_yticks(np.arange(0.0, 1.2, 0.1), minor=True)

legend_entries = [
    MulticolorPatch([prompt_colors[p["basename"]]["instruct"],
                     prompt_colors[p["basename"]]["base"]])
    for p in prompt_designs
]
legend_labels = [p["label"] for p in prompt_designs]
fig.legend(
    legend_entries, legend_labels,
    handler_map={MulticolorPatch: MulticolorPatchHandler()},
    loc="upper center", ncol=n_prompts, title="Prompts",
    bbox_to_anchor=(0.5, 1.05), fontsize=12, title_fontsize=14,
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(figures_dir / "label_prob_calibration.svg", bbox_inches="tight")
plt.savefig(figures_dir / "label_prob_calibration.png", bbox_inches="tight")
plt.show()
```

# LaTeX Tables

Count invalid answers (questions where the model assigns zero probability to all
choices) per model and prompt directly from `raw_data`, avoiding any additional
file reads.

```python
invalid_answers_counter = defaultdict(lambda: defaultdict(int))
for model, prompt, dataset in product(models, prompt_designs, all_datasets):
    m_id, pname, ds_id = model["id"], prompt["basename"], dataset["id"]
    df = raw_data[pname][ds_id][m_id]
    if df is not None:
        count = df["confidence_per_choice"].apply(lambda x: sum(x) <= 0).sum()
        invalid_answers_counter[m_id][pname] += count
```

## Label Probability Sum by Model Type and Prompt

Aggregate mean label-probability sum per model *type* (base, instruct, reasoning)
across all models of that type, then write a LaTeX table.

```python
means_by_type = {
    key: {pname: [] for pname in prompt_basenames}
    for key in ["base", "instruct", "reasoning"]
}
for model, pname in product(models, prompt_basenames):
    mean, _ = plot_data[model["id"]][pname]
    means_by_type[model["type"]][pname].append(mean)

table_rows = {"Base Models": {}, "Instruction Tuned Models": {}, "Reasoning Models": {}, "Average": {}}
for pname in prompt_basenames:
    agg = {t: np.mean(means_by_type[t][pname]) if means_by_type[t][pname] else 0
           for t in ["base", "instruct", "reasoning"]}
    agg["avg"] = statistics.fmean(agg.values())
    table_rows["Base Models"][pname] = f"{agg['base']:.4f}"
    table_rows["Instruction Tuned Models"][pname] = f"{agg['instruct']:.4f}"
    table_rows["Reasoning Models"][pname] = f"{agg['reasoning']:.4f}"
    table_rows["Average"][pname] = f"{agg['avg']:.4f}"

latex_table = (
    "\n\\begin{table}[h]\n"
    "    \\centering\n"
    "    \\begin{tabular}{lcccc}\n"
    "        \\toprule\n"
    "        & Prompt 1 & Prompt 2 & Prompt 3 & Prompt 4 \\\\\n"
    "        \\midrule\n"
)
for row_name, values in table_rows.items():
    row_values = " & ".join(values[pname] for pname in prompt_basenames)
    latex_table += f"        {row_name} & {row_values} \\\\\n"
latex_table += (
    "        \\bottomrule\n"
    "    \\end{tabular}\n"
    "    \\caption{Mean over Sum of Label Probabilities per Question for Base and Instruction Tuned Models}\n"
    "    \\label{tab:labelprobsum}\n"
    "\\end{table}\n"
)

with open(tables_dir / "prompt_design_label_prob_sum.tex", "w", encoding="utf-8") as f:
    f.write(latex_table)
print(latex_table)
```

## ECE by Prompt Design

Per-dataset, per-model ECE values across all prompt designs, with mean and
max–min deviation columns.

```python
prop = "ece"
table_data_ece = {}
dataset_stats = {}

for dataset in all_datasets:
    ds_id = dataset["id"]
    table_data_ece[ds_id] = {}
    deviations, means_list = [], []

    for model in models:
        m_id = model["id"]
        prop_list = []
        table_data_ece[ds_id][m_id] = {}

        for pname in prompt_basenames:
            try:
                value = cal_data["normalized"][pname][ds_id][m_id]["chosen_labels"][prop]
            except (KeyError, TypeError):
                value = 0
            table_data_ece[ds_id][m_id][pname] = value
            prop_list.append(value)

        mean_value = np.mean(prop_list)
        deviation = np.std(prop_list)
        table_data_ece[ds_id][m_id]["Mean"] = mean_value
        table_data_ece[ds_id][m_id]["Deviation"] = deviation
        means_list.append(mean_value)
        deviations.append(deviation)

    dataset_stats[ds_id] = {
        "mean_deviation": np.mean(deviations),
        "std_deviation": np.std(deviations),
        "mean_mean": np.mean(means_list),
    }

latex_ece = (
    f"\n\\begin{{table}}[h]\n"
    f"    \\centering\n"
    f"    \\makebox[\\textwidth][c]{{%\n"
    f"    \\begin{{tabular}}{{l|lcccc|cc}}\n"
    f"        \\toprule\n"
    f"        Dataset & Model & Prompt 1 & Prompt 2 & Prompt 3 & Prompt 4 & Mean & max-min \\\\\n"
    f"        \\midrule\n"
)
for ds_id, models_ece in table_data_ece.items():
    first_row = True
    for model_id, prop_values in models_ece.items():
        row_values = " & ".join(f"{prop_values[pname]:.4f}" for pname in prompt_basenames)
        mean_value = f"{prop_values['Mean']:.4f}"
        prop_list = [prop_values[pname] for pname in prompt_basenames]
        max_minus_min = f"{max(prop_list) - min(prop_list):.4f}"
        if first_row:
            latex_ece += (f"        \\multirow{{{len(models_ece)}}}{{*}}{{{ds_id}}} "
                          f"& {model_id} & {row_values} & {mean_value} & {max_minus_min} \\\\\n")
            first_row = False
        else:
            latex_ece += f"        & {model_id} & {row_values} & {mean_value} & {max_minus_min} \\\\\n"
    latex_ece += "        \\midrule\n"

caption_parts = [
    f"{ds_id}: $\\mu_{{dev}}$={stats['mean_deviation']:.4f}, "
    f"$\\sigma_{{dev}}$={stats['std_deviation']:.4f}, "
    f"$\\mu_{{mean}}$={stats['mean_mean']:.4f}"
    for ds_id, stats in dataset_stats.items()
]
latex_ece += (
    "        \\bottomrule\n"
    "    \\end{tabular}\n"
    "    }\n"
    f"    \\caption{{ECE Across Prompts (using Probability of chosen Labels only) "
    f"with Mean and Deviation for each Dataset. Dataset-level statistics: {' '.join(caption_parts)}. }}\n"
    "    \\label{tab:ece}\n"
    "\\end{table}\n"
)

print(latex_ece)
with open(tables_dir / "prompt_design_ece.tex", "w", encoding="utf-8") as f:
    f.write(latex_ece)
```

## Invalid Answers by Model and Prompt

Number of questions per model per prompt where no probability mass was assigned to
any answer choice, aggregated across all datasets.

```python
latex_invalid = (
        "\\begin{table}[h]\n\\centering\n"
        "\\begin{tabular}{l" + "c" * len(prompt_designs) + "}\n\\hline\n"
        + "Model & " + " & ".join(f"Prompt {p['idx']}" for p in prompt_designs) + " \\\\\n\\hline\n"
)
for model in models:
    row_data = [str(invalid_answers_counter[model["id"]].get(p["basename"], 0))
                for p in prompt_designs]
    latex_invalid += f"{model['id']} & " + " & ".join(row_data) + " \\\\\n"
latex_invalid += (
    "\\hline\n\\end{tabular}\n"
    "\\caption{Number of invalid answers given by models for different prompts across all datasets. "
    "Invalid answers do not assign any probability mass to the answer choice labels.}\n"
    "\\label{tab:invalid_answers}\n"
    "\\end{table}"
)

print(latex_invalid)
with open(tables_dir / "invalid_answers.tex", "w", encoding="utf-8") as f:
    f.write(latex_invalid)
```
