"""LaTeX table generation utilities for benchmark result reporting.

All functions return a complete LaTeX snippet as a string; the caller is
responsible for writing it to a file.
"""


def make_accuracy_table_latex(data_dict, properties_labels, model_ids, caption):
    """Generate a LaTeX booktabs table comparing accuracy metrics across models and datasets.

    Produces a ``tabular`` with one row per (dataset × metric) pair and one
    column per model.  Datasets are visually grouped with ``\\multirow``.

    Args:
        data_dict: Nested dict ``data_dict[dataset_id][model_id][prop] = value``.
        properties_labels: Ordered dict mapping ``property_key`` → display label
            (determines both row order and label strings).
        model_ids: Ordered list of model ID strings (become rotated column headers).
        caption: Complete LaTeX caption string used verbatim inside ``\\caption``.

    Returns:
        str: Complete LaTeX table code ready to paste into a ``.tex`` file.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\makebox[\textwidth][c]{%",
        r"\begin{tabular}{ll" + "c" * len(model_ids) + "}",
    ]
    header = ["Dataset", "Metric"] + [
        r"\parbox[t]{0mm}{\rotatebox{60}{" + m + "}}" for m in model_ids
    ]
    lines += [" & ".join(header) + r" \\", r"\hline"]

    for ds_id, props_for_model in data_dict.items():
        n_props = len(properties_labels)
        for i, (prop, label) in enumerate(properties_labels.items()):
            row = []
            row.append(
                r"\multirow{" + str(n_props) + r"}{*}{" + ds_id.replace("_", " ") + "}"
                if i == 0 else ""
            )
            row.append(label)
            for mid in model_ids:
                val = props_for_model.get(mid, {}).get(prop, "N/A")
                row.append(f"{val:.4f}" if isinstance(val, float) else str(val))
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\hline")

    lines += [r"\end{tabular}", r"}", caption, r"\label{tab:your_label}", r"\end{table}"]
    return "\n".join(lines)


def make_length_table_latex(length_metadata, model_ids, dataset_ids,
                            caption="Response Token Lengths by Model and Dataset"):
    """Generate a LaTeX booktabs table of mean ± std response token lengths.

    Columns are grouped by model (Mean / ±Std sub-columns); rows are datasets.

    Args:
        length_metadata: Nested dict ``length_metadata[dataset_id][model_id]``
            with ``"mean"`` and ``"std"`` float values.
        model_ids: Ordered list of model ID strings (become rotated column group headers).
        dataset_ids: Ordered list of dataset ID strings (become row labels).
        caption: Table caption string used inside ``\\caption``.

    Returns:
        str: Complete LaTeX table code.
    """
    lines = [
        r"\begin{table}[ht]", r"  \centering", r"  \makebox[\textwidth][c]{%",
        r"    \begin{tabular}{l " + " ".join(["r r"] * len(model_ids)) + "}",
        r"      \toprule",
    ]
    header_cells = ["Dataset"] + [
        rf"\multicolumn{{2}}{{c}}{{\parbox[t]{{0mm}}{{\rotatebox{{60}}{{{mid}}}}}}}"
        for mid in model_ids
    ]
    lines.append("      " + " & ".join(header_cells) + r" \\")
    lines.append(r"      \midrule")
    subcells = [""] + [item for _ in model_ids for item in [r"Mean", r"$\pm$Std"]]
    lines.append("      " + " & ".join(subcells) + r" \\")
    lines.append(r"      \midrule")
    for dataset_id in dataset_ids:
        row = [dataset_id.replace("_", " ")]
        for mid in model_ids:
            stats = length_metadata[dataset_id][mid]
            row += [f"{stats['mean']:.2f}", f"{stats['std']:.2f}"]
        lines.append("      " + " & ".join(row) + r" \\")
    lines += [
        r"      \bottomrule", r"    \end{tabular}%", r"  }",
        rf"  \caption{{{caption}}}", r"\end{table}",
    ]
    return "\n".join(lines)


def make_scalar_table_latex(row_ids, model_ids, values, caption, label=None):
    """Generate a LaTeX center/footnotesize table for one scalar property per cell.

    Produces a ``tabular`` with one row per entry in *row_ids* and one column
    per model.  Intended for single-number metrics such as ECE, AUROC, or
    normalised entropy.

    Args:
        row_ids: Ordered list of row label strings (e.g. dataset IDs).
            Underscores are replaced with spaces in the table.
        model_ids: Ordered list of model ID strings (become rotated column headers).
        values: Nested dict ``values[row_id][model_id] = float``.
            Use ``float("nan")`` for missing entries.
        caption: Caption string used in ``\\captionof``.
        label: Short label for the optional argument of ``\\captionof``;
            defaults to *caption*.

    Returns:
        str: Complete LaTeX table code.
    """
    label = label or caption
    lines = [
        r"\begin{center}", r"  \footnotesize", r"  \centering",
        r"  \makebox[\textwidth][c]{%",
        r"    \begin{tabular}{l " + " ".join(["c"] * len(model_ids)) + "}",
        r"      \toprule",
    ]
    header = ["Dataset"] + [
        fr"\parbox[t]{{0mm}}{{\rotatebox{{60}}{{{mid}}}}}" for mid in model_ids
    ]
    lines += ["      " + " & ".join(header) + r" \\", r"      \midrule"]
    for row_id in row_ids:
        row = [row_id.replace("_", " ")]
        for mid in model_ids:
            val = values[row_id][mid]
            row.append(f"{val:.4f}")
        lines.append("      " + " & ".join(row) + r" \\")
    lines += [
        r"      \bottomrule", r"    \end{tabular}%", r"  }",
        fr"  \captionof{{table}}[{label}]{{{caption}}}",
        r"\end{center}",
    ]
    return "\n".join(lines)
