---
jupyter:
  jupytext:
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

Analyses **token-level uncertainty** in multiple-choice QA: how well the
label-choice probabilities assigned by a model reflect its actual likelihood of
being correct.

Pipeline:
1. Load raw benchmark results (one CSV per prompt × dataset × model) and compute
   calibration metrics in a single memory-efficient loop.
2. Delete each raw DataFrame immediately after metric computation to minimise peak
   RAM usage.
3. Visualise calibration grids (binned and relplot variants), the normalisation
   effect, label-choice bias, and generate LaTeX tables.

For each `(prompt, dataset, model)` cell, four calibration objects are produced
(normalize × chosen_only):

* `"norm_chosen"` – L1-normalised probabilities, argmax label only
* `"norm_all"`    – L1-normalised probabilities, all 4 labels
* `"raw_chosen"`  – raw token probabilities, argmax label only
* `"raw_all"`     – raw token probabilities, all 4 labels

Each leaf dict also contains a `"relplot_diagram"` key for smooth-ECE reliability
diagrams.

```python
%load_ext autoreload
%autoreload 2
```

## Imports

```python
import os
import re
import statistics
from collections import defaultdict
from itertools import product

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import relplot as rp
from async_graph_bench import CSVDataStore
from matplotlib.collections import PatchCollection
from matplotlib.colors import to_rgb
```

```python
# Config: models, all_datasets, datasets_by_id, prompt_designs, prompt_basenames, paths, safe_format
from analysis_config import (
    models,
    all_datasets,
    datasets_by_id,
    prompt_designs,
    prompt_basenames,
    FIGURES_DIR,
    TABLES_DIR,
    COMPUTE_RELPLOT_FOR_ALL_VARIANTS,
    safe_format,
)

# compute_all_variants: parse each label-prob DataFrame once and return all four
# (normalize × chosen_only) calibration metrics dicts from shared numpy arrays.
# Also returns raw (correct, certainties) arrays per variant for relplot.
from analysis_utils.calibration_compute import compute_all_variants

# Calibration grid builders, subplot helpers, relplot subplot, normalization comparison
from analysis_utils.calibration_plot_helpers import (
    _cal_key,
    make_calibration_subplot_fn,
    plot_relplot_subplot,
    render_metrics_table,
    build_calibration_grid,
    save_all_calibration_grids,
    plot_normalization_comparison,
)

# Global matplotlib style defaults (font family, figure background, etc.)
from util.config import apply_matplotlib_defaults
```

## Setup

```python
apply_matplotlib_defaults()

for d in [FIGURES_DIR, TABLES_DIR, FIGURES_DIR / "full_plots"]:
    os.makedirs(d, exist_ok=True)
```

## Data Loading and Metric Computation

Data loading and calibration computation in one memory-efficient loop.  For every
`(prompt, dataset, model)` triplet:

* Load the raw CSV into a DataFrame.
* Compute calibration metrics for all 4 `normalize × chosen_only` combinations.
  `return_arrays=True` yields the raw `(correct, certainties)` arrays so that the
  relplot diagram can be computed inline before the arrays are discarded.
* Accumulate label-bias probability mass per model (first prompt only).
* `del df` immediately to free memory.

`cal_data[pname][ds_id][m_id]` → dict with keys `"norm_chosen"`, `"norm_all"`,
`"raw_chosen"`, `"raw_all"` (or `None` when the CSV was missing).

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
# cal_data[pname][ds_id][m_id] → dict of 4 calibration dicts (or None if CSV missing).
cal_data = {
    pname: {ds["id"]: {} for ds in all_datasets}
    for pname in prompt_basenames
}

# Label bias: normalised probability mass accumulated per label index per model.
# Only the first prompt design is used (consistent with the original analysis).
labels_chosen_count  = defaultdict(lambda: defaultdict(float))
_label_bias_prompt   = prompt_designs[0]

# Ground-truth correct-answer distribution (reference for label-bias comparison).
actual_correct_label_count = defaultdict(int)
for dataset in all_datasets:
    for label, count in dataset["df"]["correct_answer"].value_counts().items():
        actual_correct_label_count[label] += count

for prompt, dataset, model in product(prompt_designs, all_datasets, models):
    pname, ds_id, m_id = prompt["basename"], dataset["id"], model["id"]
    print(f"Loading  {pname}/{ds_id}/{m_id}")
    df = load_label_prob_df(prompt, dataset, model)

    if df is None:
        cal_data[pname][ds_id][m_id] = None
        continue

    # Parse the DataFrame once; derive all 4 (normalize × chosen_only) variants
    # from shared numpy arrays — ~4× faster than calling compute_calibration_metrics
    # four times separately.
    try:
        metrics_dict, arrays_dict = compute_all_variants(dataset["df"], df, n_bins=15)
    except Exception as e:
        print(f"  ERROR compute_all_variants: {e}")
        cal_data[pname][ds_id][m_id] = None
        del df
        continue

    entry = {}
    for key, metrics in metrics_dict.items():
        chosen_only = key.endswith("_chosen")
        # Compute relplot diagram.  For "all" variants this can be skipped during
        # development by setting COMPUTE_RELPLOT_FOR_ALL_VARIANTS = False in
        # analysis_config.py to roughly halve bootstrap-CI computation cost.
        should_compute_relplot = chosen_only or COMPUTE_RELPLOT_FOR_ALL_VARIANTS
        correct_arr, certainties_arr = arrays_dict[key]
        if should_compute_relplot and len(correct_arr) > 0:
            try:
                metrics["relplot_diagram"] = rp.prepare_rel_diagram(
                    f=certainties_arr.astype(float),
                    y=correct_arr.astype(float),
                    plot_confidence_band=True,
                    report_CE_std=True,
                )
            except Exception as e:
                print(f"  relplot ERROR {key}: {e}")
                metrics["relplot_diagram"] = None
        else:
            metrics["relplot_diagram"] = None
        entry[key] = metrics

    # Accumulate normalised probability mass per label index (first prompt only).
    # Uses vectorised numpy ops to avoid iterrows() overhead.
    if prompt is _label_bias_prompt:
        probs_mat = np.stack(
            df["confidence_per_choice"].apply(lambda x: np.asarray(x, dtype=float)).values
        )  # (N, 4)
        row_sums = probs_mat.sum(axis=1, keepdims=True)
        probs_norm = np.where(row_sums > 0, probs_mat / row_sums, 0.25)
        for i, v in enumerate(probs_norm.sum(axis=0)):
            labels_chosen_count[m_id][i] += float(v)

    cal_data[pname][ds_id][m_id] = entry
    del df  # free memory immediately after computation
```

```python
import pickle

def save_cal_data(cal_data, filename):
    with open(filename, "wb") as f:
        pickle.dump(cal_data, f)

def load_cal_data(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

# save_cal_data(cal_data, "analysis_cache.pkl")
# cal_data = load_cal_data("analysis_cache.pkl")
```

```python
# Normalise accumulated label-bias counts to proportions (must sum to 1 per model).
labels_chosen_proportion = {}
for m_id, counts in labels_chosen_count.items():
    total = sum(counts.values())
    labels_chosen_proportion[m_id] = {label: counts[label] / total for label in range(4)}

total_gt = sum(actual_correct_label_count.values())
actual_correct_label_proportion = {
    i: actual_correct_label_count[i] / total_gt for i in range(4)
}
print("Ground-truth label distribution:", actual_correct_label_proportion)
```

## Full Calibration Grids

Generate and save calibration grid SVGs for all combinations of prompt ×
normalisation × label selection.  Both binned-calibration and relplot variants
are saved.

```python
save_all_calibration_grids(prompt_designs, all_datasets, models, cal_data, FIGURES_DIR)
apply_matplotlib_defaults()
```

## Paper Header Plot

A focused calibration grid used as the paper header figure, showing three
representative datasets and four key models with ECE annotations.

```python
apply_matplotlib_defaults()
plot = build_calibration_grid(
    prompt_designs[0],
    datasets=[d for d in all_datasets if d["id"] in ["MMLU", "GSM8KMC", "GPQA"]],
    models=[m for m in models if any(name in m["basename"] for name in
                                     ["Mistral-Small-3", "Magistral-Small-2507-Reasoning-Enabled",
                                      "Llama-3", "Qwen3-30B-A3B"])],
    cal_data=cal_data,
    chosen_only=True, normalize=True,
    with_table=False, with_title=False,
    ece_in_plot=True,
    row_col_titles_font_size=32,
)
plot.savefig(FIGURES_DIR / "prompt_1_header_plot.svg", bbox_inches="tight")
plot.show()
```

```python
# Relplot (smooth reliability diagram) variant of the paper header figure.
apply_matplotlib_defaults()
rp_plot = build_calibration_grid(
    prompt_designs[0],
    datasets=[d for d in all_datasets if d["id"] in ["MMLU", "GSM8KMC", "GPQA"]],
    models=[m for m in models if any(name in m["basename"] for name in
                                     ["Mistral-Small-3", "Magistral-Small-2507-Reasoning-Enabled",
                                      "Llama-3", "Qwen3-30B-A3B"])],
    cal_data=cal_data,
    chosen_only=True, normalize=True,
    with_table=False, with_title=False,
    subplot_fn=plot_relplot_subplot,
    row_col_titles_font_size=32,
)
rp_plot.savefig(FIGURES_DIR / "prompt_1_header_plot_relplot.svg", bbox_inches="tight")
rp_plot.show()
```

## Normalization Effect

Side-by-side calibration grids for one model showing the effect of applying
L1-normalisation to the raw token probabilities.

```python
apply_matplotlib_defaults()
plot_normalization_comparison(
    next(m for m in models if "Mistral-Small-3.1-24B-Base-2503" in m["id"]),
    all_datasets,
    prompt_designs[0],
    cal_data=cal_data,
    figures_dir=FIGURES_DIR,
)
```

## Label Bias Analysis

Visualise how much probability mass each model assigns to each answer-choice
position (A/B/C/D) versus the ground-truth correct-answer distribution.
The x-axis groups bars by label position (A–D); each model gets one bar per
position, coloured by model type (Blues = instruct, Oranges = base,
Greens = reasoning).  A local dashed reference line marks the ground-truth
proportion for each label.

```python
apply_matplotlib_defaults()

labels = ["A", "B", "C", "D"]
num_labels = len(labels)
model_ids = [m["id"] for m in models]

ground_truth = [actual_correct_label_proportion.get(i, 0.0) for i in range(num_labels)]
model_distributions = {
    m_id: [labels_chosen_proportion.get(m_id, {}).get(i, 0.0) for i in range(num_labels)]
    for m_id in model_ids
}

# Assign colours per model: Blues for instruct, Oranges for base, Greens for reasoning.
categories = {
    "instruct": ("Blues",    [m["id"] for m in models if m["type"] == "instruct"]),
    "base":     ("Oranges",  [m["id"] for m in models if m["type"] == "base"]),
    "reasoning":("Greens",   [m["id"] for m in models if m["type"] == "reasoning"]),
}
model_colors = {}
for cmap_name, ids in categories.values():
    cmap = plt.cm.get_cmap(cmap_name)
    colors = cmap(np.linspace(0.4, 0.8, max(len(ids), 1)))
    model_colors.update({m_id: colors[i] for i, m_id in enumerate(ids)})
ground_truth_color = "lightgrey"

bar_width = 0.04
x = np.arange(num_labels)

fig, ax = plt.subplots(figsize=(20, 6))

# Ground-truth bars (leftmost in each group)
ax.bar(
    x - bar_width * (len(models) / 2),
    ground_truth, bar_width,
    label="Ground Truth", color=ground_truth_color, edgecolor="black",
)
# Per-model bars
for i, m_id in enumerate(model_ids):
    ax.bar(
        x - bar_width * (len(model_ids) / 2) + (i + 1) * bar_width,
        model_distributions[m_id], bar_width,
        label=m_id, color=model_colors.get(m_id, "grey"), edgecolor="black",
    )
# Local dashed reference lines (one per label position)
for j, gt in enumerate(ground_truth):
    ax.hlines(
        y=gt,
        xmin=x[j] - bar_width * (len(model_ids) / 2 + 1),
        xmax=x[j] + bar_width * (len(model_ids) / 2 + 1),
        colors="black", linestyles="dashed", alpha=0.5,
    )

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=14)
ax.tick_params(axis="x", pad=10)
ax.set_yticks(np.arange(0.0, 0.8, 0.1))
ax.set_yticklabels([round(v, 1) for v in np.arange(0.0, 0.8, 0.1)], fontsize=12)
ax.set_ylabel("Proportion", fontsize=14)
ax.set_ylim(0, 0.7)
handles = [plt.Rectangle((0, 0), 1, 1, color=ground_truth_color, edgecolor="black")]
handles += [plt.Rectangle((0, 0), 1, 1, color=model_colors.get(m, "grey"), edgecolor="black")
            for m in model_ids]
ax.legend(handles, ["Ground Truth"] + model_ids,
          loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=14)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "label_bias.svg", bbox_inches="tight")
plt.savefig(FIGURES_DIR / "label_bias_estimations.svg", bbox_inches="tight")
plt.savefig(FIGURES_DIR / "label_bias.png", bbox_inches="tight")
plt.savefig(FIGURES_DIR / "label_bias_estimations.png", bbox_inches="tight")
plt.show()
```

## Label Probability Sum

Compute mean and std of the per-question label-probability sum per (model, prompt)
from pre-computed `cal_data`.  A sum of ≈1 indicates the model assigns all of its
probability mass to the four answer-choice tokens; deviations reveal overconfidence
or diffuse distributions.

```python
# plot_data[model_id][prompt_basename] = (mean, std)
# Uses raw_chosen (un-normalised probability vectors) so the sum reflects the
# actual probability mass the model assigns to the four answer-choice tokens.
model_ids = [m["id"] for m in models]
dataset_id_first = all_datasets[0]["id"]

plot_data = {m_id: {} for m_id in model_ids}
for prompt, model in product(prompt_designs, models):
    pname, m_id = prompt["basename"], model["id"]
    entry = (cal_data[pname][dataset_id_first][m_id] or {}).get("raw_chosen")
    plot_data[m_id][pname] = (
        (entry["label_prob_sum_mean"], entry["label_prob_sum_std"]) if entry else (0, 0)
    )
```

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
    """Blend a matplotlib colour towards white by the given amount (0=original, 1=white)."""
    c = np.array(to_rgb(color))
    return np.clip(c + (1 - c) * amount, 0, 1)

shades = [0.0, 0.2, 0.4, 0.6]
prompt_colors = {
    prompt["basename"]: {
        "instruct": lighten_color("tab:blue",   amount=shades[i]),
        "base":     lighten_color("tab:orange", amount=shades[i]),
        "reasoning": lighten_color("tab:green", amount=shades[i]),
    }
    for i, prompt in enumerate(prompt_designs)
}
```

```python
apply_matplotlib_defaults()
n_models  = len(model_ids)
n_prompts = len(prompt_designs)
bar_width = 0.2
x = np.arange(n_models)

fig, ax = plt.subplots(figsize=(20, 6))

for j, prompt in enumerate(prompt_designs):
    offset = (j - (n_prompts - 1) / 2) * bar_width
    for i, model in enumerate(models):
        mean, std = plot_data[model["id"]][prompt["basename"]]
        xpos = x[i] + offset
        c = prompt_colors[prompt["basename"]][model["type"]]
        ax.bar(xpos, mean, bar_width, yerr=std, capsize=3, color=c, edgecolor="black",
               label=prompt["basename"] if i == 0 else "")

ax.set_xticks(x)
ax.set_xticklabels(model_ids, rotation=45, ha="right", fontsize=14)
ax.set_ylabel("Label Prob Sum Mean", fontsize=16)
# y-axis: auto-scale to data with a small top margin; show minor gridlines every 0.1.
_max_mean = max(
    (plot_data[m_id][p["basename"]][0] for m_id in model_ids for p in prompt_designs),
    default=1.0,
)
_ylim = max(1.2, _max_mean * 1.15)
ax.set_ylim(0, _ylim)
_ytick_step = 0.2
_yticks = np.arange(0.0, _ylim + _ytick_step, _ytick_step)
ax.set_yticks(_yticks)
ax.set_yticklabels([f"{v:.1f}" for v in _yticks], fontsize=14)
ax.set_yticks(np.arange(0.0, _ylim + 0.1, 0.1), minor=True)
ax.set_axisbelow(True)
ax.grid(axis="y", color="lightgrey", linestyle="--")
ax.grid(axis="y", which="minor", color="lightgrey", linestyle="--", alpha=0.4)

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
plt.savefig(FIGURES_DIR / "label_prob_calibration.svg", bbox_inches="tight")
plt.savefig(FIGURES_DIR / "label_prob_calibration.png", bbox_inches="tight")
plt.show()
```

## LaTeX Tables

### Invalid Answers by Model and Prompt

Number of questions per model per prompt where no probability mass was assigned
to any answer choice, derived from `cal_data["invalid_answers"]`.

```python
invalid_answers_counter = defaultdict(lambda: defaultdict(int))
for model, prompt, dataset in product(models, prompt_designs, all_datasets):
    m_id, pname, ds_id = model["id"], prompt["basename"], dataset["id"]
    entry = (cal_data[pname][ds_id][m_id] or {}).get("norm_chosen")
    if entry is not None:
        invalid_answers_counter[m_id][pname] += entry["invalid_answers"]
```

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
with open(TABLES_DIR / "invalid_answers.tex", "w", encoding="utf-8") as f:
    f.write(latex_invalid)
```

### Label Probability Sum by Model Type and Prompt

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

table_rows = {
    "Base Models": {},
    "Instruction Tuned Models": {},
    "Reasoning Models": {},
    "Average": {},
}
for pname in prompt_basenames:
    agg = {t: np.mean(means_by_type[t][pname]) if means_by_type[t][pname] else 0
           for t in ["base", "instruct", "reasoning"]}
    agg["avg"] = statistics.fmean(agg.values())
    table_rows["Base Models"][pname]              = f"{agg['base']:.4f}"
    table_rows["Instruction Tuned Models"][pname] = f"{agg['instruct']:.4f}"
    table_rows["Reasoning Models"][pname]         = f"{agg['reasoning']:.4f}"
    table_rows["Average"][pname]                  = f"{agg['avg']:.4f}"

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

print(latex_table)
with open(TABLES_DIR / "prompt_design_label_prob_sum.tex", "w", encoding="utf-8") as f:
    f.write(latex_table)
```

### ECE by Prompt Design

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
                value = cal_data[pname][ds_id][m_id]["norm_chosen"][prop]
            except (KeyError, TypeError):
                value = 0
            table_data_ece[ds_id][m_id][pname] = value
            prop_list.append(value)

        mean_value = np.mean(prop_list)
        deviation  = np.std(prop_list)
        table_data_ece[ds_id][m_id]["Mean"]      = mean_value
        table_data_ece[ds_id][m_id]["Deviation"] = deviation
        means_list.append(mean_value)
        deviations.append(deviation)

    dataset_stats[ds_id] = {
        "mean_deviation": np.mean(deviations),
        "std_deviation":  np.std(deviations),
        "mean_mean":      np.mean(means_list),
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
        row_values    = " & ".join(f"{prop_values[pname]:.4f}" for pname in prompt_basenames)
        mean_value    = f"{prop_values['Mean']:.4f}"
        prop_list     = [prop_values[pname] for pname in prompt_basenames]
        max_minus_min = f"{max(prop_list) - min(prop_list):.4f}"
        if first_row:
            latex_ece += (f"        \\multirow{{{len(models_ece)}}}{{*}}{{{ds_id}}} "
                          f"& {model_id} & {row_values} & {mean_value} & {max_minus_min} \\\\\n")
            first_row = False
        else:
            latex_ece += (f"        & {model_id} & {row_values} & {mean_value}"
                          f" & {max_minus_min} \\\\\n")
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
    f"with Mean and Deviation for each Dataset. Dataset-level statistics: "
    f"{' '.join(caption_parts)}. }}\n"
    "    \\label{tab:ece}\n"
    "\\end{table}\n"
)

print(latex_ece)
with open(TABLES_DIR / "prompt_design_ece.tex", "w", encoding="utf-8") as f:
    f.write(latex_ece)
```

## Reasoning Chain Lengths

For reasoning models, visualise the distribution of answer-token lengths
(`amount_answer_tokens`) per dataset as grouped box plots.  Only values
≤ 10 240 tokens are shown (extreme outliers excluded).

```python
reasoning_models = [m for m in models if m["type"] == "reasoning"]

grouped_data = {d["id"]: [] for d in all_datasets}
model_labels = []
for model in reasoning_models:
    model_labels.append(model.get("shortname", model.get("basename", model["id"])))
    for dataset in all_datasets:
        try:
            result_path = prompt_designs[0]["path"] / dataset["id"] / model["id"]
            from async_graph_bench import CSVDataStore as _CSV
            lp_df = _CSV(result_path, "LabelProbExtractor").to_dataframe()
            series = lp_df["amount_answer_tokens"]
            data_values = series[series <= 10240].dropna().tolist()
            grouped_data[dataset["id"]].append(data_values)
        except Exception as e:
            print(f"  Skipping {model['id']}/{dataset['id']}: {e}")
            grouped_data[dataset["id"]].append([])

num_models   = len(reasoning_models)
num_datasets = len(all_datasets)
x_indexes    = np.arange(num_models)
box_width    = 0.15
ds_colors    = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

apply_matplotlib_defaults()
fig, ax = plt.subplots(figsize=(12, 6))
legend_handles = []
for i, dataset in enumerate(all_datasets):
    positions = x_indexes + (i - (num_datasets - 1) / 2) * box_width
    bp = ax.boxplot(
        grouped_data[dataset["id"]],
        positions=positions, widths=box_width,
        patch_artist=True, showfliers=False,
    )
    color = ds_colors[i % len(ds_colors)]
    for box in bp["boxes"]:
        box.set_facecolor(color)
        box.set_alpha(0.8)
    for median in bp["medians"]:
        median.set_color("black")
    legend_handles.append(mpatches.Patch(color=color, label=dataset["id"]))

ax.set_xticks(x_indexes)
ax.set_xticklabels(model_labels, rotation=45, ha="right")
ax.set_xlabel("Reasoning Model")
ax.set_ylabel("Amount of Reasoning Tokens")
ax.set_title("Reasoning Tokens Distribution by Reasoning Model and Dataset")
ax.legend(handles=legend_handles, title="Dataset")
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
for ext in ["svg", "pdf"]:
    plt.savefig(FIGURES_DIR / f"label_probability_reasoning_lengths.{ext}", bbox_inches="tight")
plt.show()
```

## Uncertainty Score Distributions

Boxplot of the normalised chosen-label probability (uncertainty score) distribution
per model, aggregated across all datasets and prompts.  Five-number summary
(min / Q25 / median / Q75 / max) is read directly from the pre-computed
`norm_chosen` calibration metrics.

```python
# Collect five-number summary per model (aggregated across datasets × prompts).
# Each model gets one box; whiskers = min/max, box = Q25–Q75, line = median.
score_dist_data = {}   # model_id → {"min": ..., "q25": ..., "med": ..., "q75": ..., "max": ...}

for model in models:
    m_id = model["id"]
    mins, q25s, meds, q75s, maxs = [], [], [], [], []
    for prompt, dataset in product(prompt_designs, all_datasets):
        pname, ds_id = prompt["basename"], dataset["id"]
        entry = (cal_data[pname][ds_id][m_id] or {}).get("norm_chosen")
        if entry is None:
            continue
        for key, lst in (("label_prob_min", mins), ("label_prob_q25", q25s),
                         ("label_prob_median", meds), ("label_prob_q75", q75s),
                         ("label_prob_max", maxs)):
            v = entry.get(key)
            if isinstance(v, (int, float)):
                lst.append(v)
    if meds:
        score_dist_data[m_id] = {
            "min":    np.min(mins),
            "q25":    np.mean(q25s),
            "median": np.mean(meds),
            "q75":    np.mean(q75s),
            "max":    np.max(maxs),
            "model_type": model["type"],
        }
```

```python
apply_matplotlib_defaults()

model_ids_ord = [m["id"] for m in models if m["id"] in score_dist_data]
x = np.arange(len(model_ids_ord))
fig, ax = plt.subplots(figsize=(max(12, len(model_ids_ord) * 1.2), 6))

for i, m_id in enumerate(model_ids_ord):
    d = score_dist_data[m_id]
    mtype = d["model_type"]
    cmap_name = {"instruct": "Blues", "base": "Oranges", "reasoning": "Greens"}.get(mtype, "Blues")
    color = plt.cm.get_cmap(cmap_name)(0.55)

    # Draw box (Q25–Q75) + median line + whiskers (min/max)
    box_lo, box_hi = d["q25"], d["q75"]
    med = d["median"]
    whi_lo, whi_hi = d["min"], d["max"]
    box_w = 0.4

    # Whisker lines
    ax.plot([i, i], [whi_lo, box_lo], color="black", linewidth=1)
    ax.plot([i, i], [box_hi, whi_hi], color="black", linewidth=1)
    # Whisker caps
    ax.plot([i - box_w * 0.3, i + box_w * 0.3], [whi_lo, whi_lo], color="black", linewidth=1)
    ax.plot([i - box_w * 0.3, i + box_w * 0.3], [whi_hi, whi_hi], color="black", linewidth=1)
    # Box rectangle
    rect = mpatches.FancyBboxPatch(
        (i - box_w / 2, box_lo), box_w, box_hi - box_lo,
        boxstyle="square,pad=0", facecolor=color, edgecolor="black", linewidth=1,
    )
    ax.add_patch(rect)
    # Median line
    ax.plot([i - box_w / 2, i + box_w / 2], [med, med], color="black", linewidth=2)

ax.set_xticks(x)
shortnames = [next(m["shortname"] for m in models if m["id"] == m_id) for m_id in model_ids_ord]
ax.set_xticklabels(shortnames, rotation=45, ha="right", fontsize=12)
ax.set_ylabel("Normalised Chosen-Label Probability", fontsize=14)
ax.set_title(
    "Uncertainty Score Distribution per Model\n"
    "(min / Q25 / median / Q75 / max, averaged across datasets \u00d7 prompts)",
    fontsize=14,
)
ax.set_ylim(0, 1)
ax.set_yticks(np.arange(0.0, 1.1, 0.1))
ax.grid(axis="y", linestyle="--", alpha=0.4)

# Type legend
legend_handles = [
    mpatches.Patch(color=plt.cm.get_cmap("Blues")(0.55), label="Instruct"),
    mpatches.Patch(color=plt.cm.get_cmap("Oranges")(0.55), label="Base"),
    mpatches.Patch(color=plt.cm.get_cmap("Greens")(0.55), label="Reasoning"),
]
ax.legend(handles=legend_handles, fontsize=12)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "uncertainty_score_distributions.svg", bbox_inches="tight")
plt.savefig(FIGURES_DIR / "uncertainty_score_distributions.png", bbox_inches="tight")
plt.show()
```

## Distance-to-1 Statistics

Mean and std of `(1 \u2212 certainty)` per model \u2014 a proxy for how far the model\u2019s
chosen-label confidence is from full certainty.  Computed here for verification;
not yet plotted.

```python
# dist_to_1_data[model_id][prompt_basename] = (mean, std)
# Averaged across all datasets for each (model, prompt) pair.
dist_to_1_data = {}
for model in models:
    m_id = model["id"]
    dist_to_1_data[m_id] = {}
    for prompt in prompt_designs:
        pname = prompt["basename"]
        means, stds = [], []
        for dataset in all_datasets:
            entry = (cal_data[pname][dataset["id"]][m_id] or {}).get("norm_chosen")
            if entry is None:
                continue
            m_val = entry.get("dist_to_1_mean")
            s_val = entry.get("dist_to_1_std")
            if isinstance(m_val, float) and isinstance(s_val, float):
                means.append(m_val)
                stds.append(s_val)
        dist_to_1_data[m_id][pname] = (
            float(np.mean(means)) if means else float("nan"),
            float(np.mean(stds))  if stds  else float("nan"),
        )

# Quick verification printout
print("Distance-to-1 stats (mean / std), norm_chosen, averaged across datasets:")
print(f"{'Model':<50} {'Prompt':<20} {'mean(1-p)':>10} {'std(1-p)':>10}")
print("-" * 95)
for model in models:
    m_id = model["id"]
    for prompt in prompt_designs:
        pname = prompt["basename"]
        mean_d, std_d = dist_to_1_data[m_id][pname]
        print(f"{m_id:<50} {pname:<20} {mean_d:>10.4f} {std_d:>10.4f}")
```

## Per-Dataset Uncertainty Score Distributions

Combined figure: one row per dataset (MMLU, ARC Reasoning, GSM8K-MC, GPQA),
shared x-axis.  Family rectangles span each row.  A dotted line connects the
median values of base → instruct → reasoning models within each family.
Legend: lower right, order base / instruct / reasoning.

```python
from util.models import model_families as _model_families
from analysis_utils.plot_score_distributions import plot_score_distributions

_dist_target_datasets = [
    {"id": "MMLU",         "label": "MMLU"},
    {"id": "ArcReasoning", "label": "ARC Reasoning"},
    {"id": "GSM8KMC",      "label": "GSM8K-MC"},
    {"id": "GPQA",         "label": "GPQA"},
]

apply_matplotlib_defaults()
_fig_dist = plot_score_distributions(
    cal_data=cal_data,
    models=models,
    model_families=_model_families,
    prompt_designs=prompt_designs,
    target_datasets=_dist_target_datasets,
    figures_dir=FIGURES_DIR,
)
plt.show()
```

## Uncertainty Score ECDFs (per Model Type)

Three figures (base / instruct / reasoning), each with one approximate ECDF
line per model.  The ECDF is constructed from the five-number summary
(min=0 %, Q25=25 %, median=50 %, Q75=75 %, max=100 %) averaged across all
four target datasets.  Model shortnames are listed in the legend to the right.

```python
from analysis_utils.plot_score_ecdf import plot_score_ecdf

apply_matplotlib_defaults()
_ecdf_figs = plot_score_ecdf(
    cal_data=cal_data,
    models=models,
    prompt_designs=prompt_designs,
    target_datasets=_dist_target_datasets,
    figures_dir=FIGURES_DIR,
)
for _mtype, _fig in _ecdf_figs:
    print(f"ECDF – {_mtype}")
    plt.show()
    apply_matplotlib_defaults()
```

## Joint IQR Comparison (Model Type × Dataset)

Single figure: for each dataset, three offset IQR markers (Q25–Q75 + median
dot) coloured by model type.  Values are aggregated by averaging Q25 / median
/ Q75 across all models of the same type per dataset.

```python
from analysis_utils.plot_score_iqr import plot_score_iqr

apply_matplotlib_defaults()
_fig_iqr = plot_score_iqr(
    cal_data=cal_data,
    models=models,
    prompt_designs=prompt_designs,
    target_datasets=_dist_target_datasets,
    figures_dir=FIGURES_DIR,
)
plt.show()
```
