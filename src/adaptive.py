# src/adaptive.py

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# def rerank_with_similarity(df, star_id, alpha=0.4):
#     vectorizer = TfidfVectorizer(stop_words="english")
#     tfidf_matrix = vectorizer.fit_transform(df["job_title_n"])

#     star_index = df.index[df["id"] == star_id][0]
#     star_vector = tfidf_matrix[star_index]

#     similarity = cosine_similarity(star_vector, tfidf_matrix).flatten()

#     df["similarity_to_star"] = similarity
#     df["adaptive_score"] = (
#         (1 - alpha) * df["base_score_adj_norm"]
#         + alpha * df["similarity_to_star"]
#     )

#     return df.sort_values("adaptive_score", ascending=False)



# src/adaptive.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityEngine:
    """
    Builds and stores TF-IDF representation.
    Should be instantiated once per ranking session.
    """

    def __init__(self, df):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(df["job_title_n"])
        self.df_index_map = {id_: idx for idx, id_ in enumerate(df["id"])}

    def compute_similarity(self, df, star_id):
        if star_id not in self.df_index_map:
            raise ValueError(f"Star ID {star_id} not found in dataset.")

        star_index = self.df_index_map[star_id]
        star_vector = self.tfidf_matrix[star_index]

        similarity = cosine_similarity(star_vector, self.tfidf_matrix).flatten()

        df = df.copy()
        df["similarity_to_star"] = similarity
        return df


def fuse_scores(df, alpha=0.4):
    """
    Combines baseline score with similarity score.
    """
    df = df.copy()
    df["adaptive_score"] = (
        (1 - alpha) * df["base_score_adj_norm"]
        + alpha * df["similarity_to_star"]
    )
    return df


def rerank(df, similarity_engine, star_id, alpha=0.4):
    """
    Performs full adaptive reranking.
    """
    df = similarity_engine.compute_similarity(df, star_id)
    df = fuse_scores(df, alpha)
    return df.sort_values("adaptive_score", ascending=False)