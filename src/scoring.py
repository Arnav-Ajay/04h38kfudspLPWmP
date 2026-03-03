# src/scoring.py

HR_TERMS = {
    "human resources": 3,
    " hr ": 3,
    "generalist": 2,
    "specialist": 2,
    "coordinator": 2,
    "people": 1,
    "development": 1,
    "talent": 1
}

SENIOR_TERMS = {
    "senior": 2,
    "manager": 2,
    "director": 3,
    "vp": 3,
    "chro": 4,
    "specialist": 1,
    "coordinator": 1
}

INTENT_TERMS = {
    "aspiring": 1,
    "seeking": 1
}

def score_text(text, term_dict):
    score = 0
    for term, weight in term_dict.items():
        if term in text:
            score += weight
    return score

def compute_semantic_score(df):
    df["hr_score"] = df["job_title_n"].apply(lambda x: score_text(x, HR_TERMS))
    df["seniority_score"] = df["job_title_n"].apply(lambda x: score_text(x, SENIOR_TERMS))
    df["intent_score"] = df["job_title_n"].apply(lambda x: score_text(x, INTENT_TERMS))

    df["semantic_score"] = (
        0.6 * df["hr_score"]
        + 0.35 * df["seniority_score"]
        + 0.05 * df["intent_score"]
    )

    return df

def compute_baseline(df):
    df["base_score"] = (
        0.90 * df["semantic_score"]
        + 0.07 * df["connections_norm"]
    )

    # Duplicate penalty
    counts = df["job_title_n"].value_counts()
    df["dup_penalty"] = df["job_title_n"].map(counts)

    df["base_score_adj"] = df["base_score"] - 0.05 * (df["dup_penalty"] - 1)

    # Normalize
    df["base_score_adj_norm"] = (
        (df["base_score_adj"] - df["base_score_adj"].min())
        / (df["base_score_adj"].max() - df["base_score_adj"].min())
    )

    return df