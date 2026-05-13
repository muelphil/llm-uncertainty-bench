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
    display_name: Python 3
    language: python
    name: python3
---

# Data Preparation – seq\_ue\_calibration

Loads raw benchmark data, computes all calibration and accuracy metrics,
and pickles the result to `resources/prepared_data.pkl` for later use by
the visualization notebook.

**Run order**: this notebook must be executed before the visualization notebook.

```python
%load_ext autoreload
%autoreload 2
```

## Imports

```python
import json
import os
import pickle
import re
from collections import defaultdict
from pathlib import Path
from functools import reduce

import numpy as np
import pandas as pd
import relplot as rp
from async_graph_bench.stores import CSVDataStore, DiskCacheStore
from sklearn.metrics import roc_auc_score
```

```python
from calibration_visualization import (
    calculate_calibration_data,
    calculate_calibration_data_discrete,
    calculate_ece,
    calculate_normalized_entropy,
)

from util.config import apply_matplotlib_defaults
from util.numpy_utils import NumpyEncoder

from datasets_exp2 import (
    arithmetic_datasets,
    arithmetic_datasets_dict,
    datasets,
    datasets_combined,
    mc_datasets,
    mc_datasets_dict,
)

# Timestamped print helper (used throughout to track long-running loops)
from seq_ue_calibration.analysis_utils.misc_utils import print_with_time, extract_number

# Filters out rows with failed answer extraction; counts unique answers per question ID
from seq_ue_calibration.analysis_utils.dataframe_utils import filter_valid_answers, unique_count_distribution

# Shared configuration: model list, UQ method list, and output paths
from seq_ue_calibration.analysis_config import (
    models,
    uq_methods,
    DATA_BASE_PATH,
    RESOURCES_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    PREPARED_DATA_PATH,
)
```

## Setup

```python
for d in [RESOURCES_DIR, FIGURES_DIR, TABLES_DIR]:
    os.makedirs(d, exist_ok=True)

print("Running data preparation for", ", ".join([m["shortname"] for m in models]))
```

## Data loading

```python
def get_merged_dataset(directory):
    """Load and merge all per-item estimations for a given benchmark output directory.

    Merges AnsweredCorrectly, ClaimConditionedProbability, PTrueOriginal, and
    Verbalized2SUEExtractor output files on (id, iter).

    Args:
        directory: Path string to the model/dataset output directory.

    Returns:
        pd.DataFrame: Merged DataFrame with columns is_correct, ccp, p_true, verbalized.
    """
    answered_correctly_df = (DiskCacheStore if "SciBench" in directory else CSVDataStore)(
        directory, "AnsweredCorrectly"
    ).to_dataframe()

    claim_prob_df, p_true_df, verbalized_df = [
        CSVDataStore(directory, src).to_dataframe().rename(columns={"estimations": name})
        for name, src in [
            ("ccp", "ClaimConditionedProbability"),
            ("p_true", "PTrueOriginal"),
            ("verbalized", "Verbalized2SUEExtractor"),
        ]
    ]
    dfs = [answered_correctly_df, claim_prob_df, p_true_df, verbalized_df]
    print(f"\t\t{'/'.join(str(len(d)) for d in dfs)}")
    lengths = [len(df) for df in dfs]
    merged_df = reduce(
        lambda left, right: pd.merge(left, right, on=["id", "iter"], how="inner"),
        dfs,
    )
    assert len(merged_df) == min(lengths) and len(merged_df) != 0, "Merge failed!"
    return merged_df
```

```python
# data[dataset_id][model_id] = merged DataFrame
data = {}

for dataset in datasets:
    dataset_id = dataset["id"]
    data[dataset_id] = {}
    print(f"Loading data for dataset {dataset_id} per model...")
    for model in models:
        print(f"\tLoading model {model['basename']}...")
        data[dataset_id][model["id"]] = get_merged_dataset(
            f"{DATA_BASE_PATH}/{dataset_id}/{model['basename']}"
        )
```

## Metric computation functions

```python
def compute_arithmetic_metrics(df, dataset_id):
    """Compute accuracy and answer-diversity metrics for arithmetic/free-form datasets.

    Args:
        df: Merged DataFrame for a single model+dataset combination.
        dataset_id: Dataset identifier string (used to select the correct answer column).

    Returns:
        dict with keys:
            totally_correct_questions (int): Questions answered correctly in all 10 iterations.
            accuracy_questions (float): Fraction of fully-correct questions.
            accuracy (float): Overall per-item accuracy.
            different_answer_count (dict): Frequency distribution of unique-answer counts per question.
            different_answer_count_mean (float): Weighted mean of the distribution.
            different_answer_count_std (float): Weighted std of the distribution.
    """
    # overall accuracy
    accuracy = float(df["is_correct"].mean())

    # per-question correctness
    counts = df.groupby("id")["is_correct"].sum()
    totally_correct = int((counts == 10).sum())
    accuracy_questions = totally_correct / len(counts)

    # distribution of different answers
    count_col = "cluster_id" if dataset_id == "SciBench" else "extracted_number"
    diff_counts = unique_count_distribution(df, count_col)

    values = np.fromiter(diff_counts.keys(), dtype=float)
    freqs = np.fromiter(diff_counts.values(), dtype=float)

    mean = np.average(values, weights=freqs)
    std = np.sqrt(np.average((values - mean) ** 2, weights=freqs))

    return {
        "totally_correct_questions": totally_correct,
        "accuracy_questions": float(accuracy_questions),
        "accuracy": accuracy,
        "different_answer_count": diff_counts,
        "different_answer_count_mean": float(mean),
        "different_answer_count_std": float(std),
    }
```

```python
def compute_binary_metrics(y_true: pd.Series, y_pred: pd.Series):
    """Compute binary classification metrics from boolean prediction and ground-truth series.

    Args:
        y_true: Boolean ground-truth labels.
        y_pred: Boolean predicted labels.

    Returns:
        dict with keys: tp, fp, tn, fn, precision, recall, accuracy, f1.
    """
    tp = (y_pred & y_true).sum()
    fp = (y_pred & ~y_true).sum()
    tn = (~y_pred & ~y_true).sum()
    fn = (~y_pred & y_true).sum()

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
    }


def compute_choice_question_accuracy(df):
    """Compute choice-level and question-level accuracy for multiple-choice datasets.

    A choice is 'totally correct' if all 10 iterations answered it correctly.
    A question is 'totally correct' if all 4 choices are totally correct.

    Args:
        df: DataFrame with columns 'id' (choice-level, last character is choice letter)
            and 'is_correct'.

    Returns:
        dict with keys:
            totally_correct_choices (int), accuracy_choices (float),
            totally_correct_questions (int), accuracy_questions (float).
    """
    # count correct answers per choice id
    counts = df.groupby("id")["is_correct"].sum()

    totally_correct_choices = (counts == 10).sum()
    accuracy_choices = totally_correct_choices / len(counts)

    # derive base question id (strip last character = choice letter)
    base_counts = counts.rename_axis("id").reset_index(name="correct_count")
    base_counts["base_id"] = base_counts["id"].str[:-1]

    # count how many fully-correct choices per question
    question_counts = (
        base_counts.groupby("base_id")["correct_count"]
        .apply(lambda x: (x == 10).sum())
    )

    totally_correct_questions = (question_counts == 4).sum()
    accuracy_questions = totally_correct_questions / len(question_counts)

    return {
        "totally_correct_choices": int(totally_correct_choices),
        "accuracy_choices": float(accuracy_choices),
        "totally_correct_questions": int(totally_correct_questions),
        "accuracy_questions": float(accuracy_questions),
    }


def extend_mc_df(df):
    """Add y_pred and y_true boolean columns to a multiple-choice DataFrame.

    Drops rows where yes_no_probabilities is NaN (failed extraction) and adds:
    - ``y_pred``: True when P(yes) > P(no) for the given choice.
    - ``y_true``: True when the choice letter matches correct_answer_idx.

    Args:
        df: Raw merged DataFrame for an MC dataset.

    Returns:
        Filtered DataFrame with new boolean columns y_pred and y_true.
    """
    mask = df["yes_no_probabilities"].notna()
    if (~mask).any():
        print(f"Warning: {(~mask).sum()} rows dropped due to missing yes_no_probabilities")
    df = df[mask].copy()

    df["answer_idx"] = df["id"].apply(lambda x: x[-1])
    df["y_true"] = (df["answer_idx"] == df["correct_answer_idx"]).astype(bool)
    df["y_pred"] = df["yes_no_probabilities"].apply(lambda x: bool(x and x[0] > x[1]))

    return df
```

```python
def compute_calibration_metrics(df, uq_method):
    """Compute calibration and discrimination metrics for one UQ method on one dataset split.

    Also calls ``relplot.prepare_rel_diagram`` so the diagram object can later be fed
    directly to ``relplot.plot_rel_diagram`` in the visualization notebook.

    Args:
        df: Filtered DataFrame (invalid answers already removed).
        uq_method: UQ method dict with keys 'id', 'type', and 'n_bins'.

    Returns:
        dict with keys:
            bin_confidences, bucket_accuracies, bucket_counts,
            correct, certainties,
            ece (float), auroc (float), accuracy (float),
            average_certainty (float), normalized_entropy (float),
            invalid_uq_method_scores (int),
            relplot_diagram (relplot diagram object).

    Raises:
        ValueError: if ``df`` has no valid rows for the given UQ method.
    """
    uq_method_id = uq_method["id"]
    need_to_invert = uq_method["type"] == "uncertainty"

    cleared = df.dropna(subset=[uq_method_id])
    invalid_uq_method_scores = len(df) - len(cleared)

    correct = cleared["is_correct"].to_numpy()
    certainties = cleared[uq_method_id].to_numpy()

    if need_to_invert:
        certainties = 1.0 - certainties

    n_bins = uq_method.get("n_bins", 15)
    calc_fn = calculate_calibration_data_discrete if uq_method.get("discrete", False) \
              else calculate_calibration_data
    bin_confs, bucket_accs, bucket_counts = calc_fn(correct, certainties, n_bins)

    # https://github.com/apple/ml-calibration/blob/main/src/relplot/diagrams.py
    relplot_diagram = rp.prepare_rel_diagram(
        f=certainties,
        y=correct.astype(float),
        plot_confidence_band=True,
        report_CE_std=True,
        kde_bandwidth=0.02
    )

    return {
        "bin_confidences": bin_confs,
        "bucket_accuracies": bucket_accs,
        "bucket_counts": bucket_counts,
        "correct": correct,
        "certainties": certainties,
        "ece": calculate_ece(bin_confs, bucket_accs, bucket_counts),
        "auroc": roc_auc_score(correct, certainties),
        "accuracy": float(correct.mean()) if len(correct) else 0.0,
        "average_certainty": float(certainties.mean()),
        "normalized_entropy": calculate_normalized_entropy(bucket_counts),
        "invalid_uq_method_scores": int(invalid_uq_method_scores),
        "relplot_diagram": relplot_diagram,
    }
```

## Main population loop

```python
# prepared_data[dataset_id][model_id] = ModelDatasetEntry (see analysis_types.py)
prepared_data = {}

for dataset in datasets:
    dataset_id = dataset["id"]
    is_arithmetic = dataset_id in arithmetic_datasets_dict

    prepared_data[dataset_id] = {}

    for model in models:
        model_id = model["id"]
        print_with_time(f"Processing {dataset_id}/{model_id} ...")

        raw_df = data[dataset_id][model_id]

        entry = {
            "is_arithmetic": is_arithmetic,
            "total_items": len(raw_df),
        }

        # Dataset-type-specific metrics and df extension
        if is_arithmetic:
            entry["df"] = raw_df
            entry["arithmetic_metrics"] = compute_arithmetic_metrics(raw_df, dataset_id)
        else:
            extended_df = extend_mc_df(raw_df)
            entry["df"] = extended_df
            entry["mc_metrics"] = {
                **compute_binary_metrics(extended_df["y_true"], extended_df["y_pred"]),
                **compute_choice_question_accuracy(extended_df),
            }

        # Calibration metrics per UQ method
        clean_df = filter_valid_answers(entry["df"]).dropna(subset=["is_correct"])
        invalid_answers = len(entry["df"]) - len(clean_df)
        entry["invalid_answers"] = int(invalid_answers)

        entry["cal"] = {}
        for uq_method in uq_methods:
            uq_method_id = uq_method["id"]
            try:
                cal = compute_calibration_metrics(clean_df, uq_method)
                entry["cal"][uq_method_id] = cal
            except Exception as e:
                print_with_time(f"  No data for {uq_method_id}: {e}")
                entry["cal"][uq_method_id] = None

        prepared_data[dataset_id][model_id] = entry
```

## Token Length Statistics

For each model/dataset pair compute mean and standard deviation of
`answer_token_len`, `reasoning_token_len`, and their sum (`combined`).
Persisted to JSON so the visualization notebook can load it without rerunning
the full preparation loop.

```python
token_length_stats = {}

for dataset in datasets:
    ds_id = dataset["id"]
    token_length_stats[ds_id] = {}
    for model in models:
        raw_df = data[ds_id][model["id"]]
        answer   = raw_df["answer_token_len"]
        reasoning = raw_df["reasoning_token_len"] if "reasoning_token_len" in raw_df else pd.Series([0])
        combined  = answer + reasoning
        token_length_stats[ds_id][model["id"]] = {
            "answer":    {"mean": float(answer.mean()),    "std": float(answer.std())},
            "reasoning": {"mean": float(reasoning.mean()), "std": float(reasoning.std())},
            "combined":  {"mean": float(combined.mean()),  "std": float(combined.std())},
        }

with open(RESOURCES_DIR / "token_length_stats.json", "w", encoding="utf-8") as f:
    json.dump(token_length_stats, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)

print("Saved token length stats.")
```

## Save pickle

```python
with open(PREPARED_DATA_PATH, "wb") as fh:
    pickle.dump(prepared_data, fh)

print(f"Saved prepared data to {PREPARED_DATA_PATH}")
```

## Accuracy tables (JSON artefacts)

Persist accuracy dicts as JSON so the visualization notebook can load them without
re-running the full loop.

```python
accuracy_per_ds_per_model_arithmetic = {
    dataset_id: {
        model_id: prepared_data[dataset_id][model_id]["arithmetic_metrics"]
        for model_id in prepared_data[dataset_id]
    }
    for dataset_id in prepared_data
    if prepared_data[dataset_id] and next(iter(prepared_data[dataset_id].values()))["is_arithmetic"]
}

accuracy_per_ds_per_model_mc = {
    dataset_id: {
        model_id: prepared_data[dataset_id][model_id]["mc_metrics"]
        for model_id in prepared_data[dataset_id]
    }
    for dataset_id in prepared_data
    if prepared_data[dataset_id] and not next(iter(prepared_data[dataset_id].values()))["is_arithmetic"]
}

with open(RESOURCES_DIR / "accuracy_per_ds_per_model_mc.json", "w", encoding="utf-8") as f:
    json.dump(accuracy_per_ds_per_model_mc, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)

with open(RESOURCES_DIR / "accuracy_per_ds_per_model_arithmetic.json", "w", encoding="utf-8") as f:
    json.dump(accuracy_per_ds_per_model_arithmetic, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)

print("Saved accuracy tables.")
```

```python

```
