def split_dataframe(df, num_subsets, subset_index):
    total_items = len(df)
    base_size = total_items // num_subsets
    remainder = total_items % num_subsets

    # Some subsets get an extra item to evenly distribute remainder
    if subset_index < remainder:
        start_idx = subset_index * (base_size + 1)
        end_idx = start_idx + base_size + 1
    else:
        start_idx = remainder * (base_size + 1) + (subset_index - remainder) * base_size
        end_idx = start_idx + base_size

    return df.iloc[start_idx:end_idx]