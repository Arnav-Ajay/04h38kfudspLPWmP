# src/pipeline.py
from .data_loader import load_data
from .scoring import compute_semantic_score, compute_baseline
from .adaptive import SimilarityEngine, rerank
from .filtering import apply_soft_filter


def rank(data_path):
    df = load_data(data_path)
    df = compute_semantic_score(df)
    df = compute_baseline(df)
    return df.sort_values("base_score_adj_norm", ascending=False)


# def rerank_pipeline(df, star_id):
#     similarity_engine = SimilarityEngine(df)
#     df = rerank(df, similarity_engine, star_id)
#     df = apply_soft_filter(df)
#     return df

def rerank_pipeline(df, star_ids):
    similarity_engine = SimilarityEngine(df)
    df = rerank(df, similarity_engine, star_ids)
    df = apply_soft_filter(df)
    return df