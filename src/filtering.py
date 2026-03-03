# src/filtering.py

def apply_soft_filter(df, percentile=0.70, similarity_threshold=0.10):
    cutoff = df["base_score_adj_norm"].quantile(percentile)

    df["is_filtered_out"] = (
        (df["base_score_adj_norm"] < cutoff)
        & (df["similarity_to_star"] < similarity_threshold)
    )

    return df[~df["is_filtered_out"]]