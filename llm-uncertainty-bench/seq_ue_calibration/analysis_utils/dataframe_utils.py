"""Pandas DataFrame utilities for filtering and summarising benchmark results."""

from collections import Counter


def filter_valid_answers(df):
    """Return only rows where answer extraction succeeded.

    Dispatches on which dataset-specific extraction column is present in *df*:

    * ``yes_no_probabilities`` – keeps rows where the probability sum is > 0.
    * ``cluster_id`` – keeps non-null rows.
    * ``extracted_number`` – keeps non-null rows.

    Args:
        df: Merged benchmark DataFrame produced by ``get_merged_dataset``.

    Returns:
        pd.DataFrame: Filtered copy with invalid-answer rows removed.

    Raises:
        ValueError: If none of the recognised extraction columns is present.
    """
    if "yes_no_probabilities" in df.columns:
        return df[df["yes_no_probabilities"].apply(lambda x: x is not None and sum(x) > 0)]
    elif "cluster_id" in df.columns:
        return df[df["cluster_id"].notna()]
    elif "extracted_number" in df.columns:
        return df[df["extracted_number"].notna()]
    raise ValueError("Unknown dataset format – no recognised extraction column found!")


def unique_count_distribution(df, column="extracted_number"):
    """Count how many unique non-null values each question ID has in *column*.

    Computes, per question ID, the number of distinct non-null values in
    *column*, then returns a frequency distribution: unique-count → number
    of IDs that produced that many distinct values.  IDs with no valid
    entries are counted with zero unique values.

    Args:
        df: Benchmark DataFrame with an ``id`` column.
        column: Column to count unique non-null values in.

    Returns:
        dict[int, int]: Mapping unique-count → frequency
        (how many question IDs have that many unique values).
    """
    counts = (
        df.dropna(subset=[column])
        .groupby("id")[column]
        .nunique()
    )
    counts = counts.reindex(df["id"].unique(), fill_value=0)
    return dict(Counter(counts))
