# Potential Talents – Adaptive Candidate Ranking System

##  Problem Statement

As a talent sourcing and management company, we aim to automate the ranking of candidates for specific roles using limited profile information:

* `job_title`
* `location`
* `connections`

We want to:

1. Rank candidates by fitness for a given role (e.g., “Aspiring Human Resources”)
2. Re-rank candidates dynamically when a candidate is starred (selected as ideal)
3. Filter out clearly irrelevant candidates
4. Avoid overfitting or bias
5. Maintain interpretability

No labeled `fit` values are provided.

Therefore, this is approached as a **ranking + adaptive relevance feedback problem**, not a supervised prediction task.

---

#  System Architecture Overview

The system consists of three layers:

1. **Deterministic Baseline Ranking**
2. **Adaptive Re-ranking via Similarity Feedback**
3. **Soft Filtering with Dynamic Cutoff**

---

# 1 Baseline Deterministic Ranking

## Feature Signals

Based on EDA, we construct structured signals from `job_title`:

* HR relevance signal
* Seniority signal
* Intent signal

Baseline rule-based score:

```
semantic_score = 0.6 * HR
               + 0.35 * Seniority
               + 0.05 * Intent
```

### Minimal Additional Signals

To preserve realism while minimizing bias:

* Connections are normalized and weighted lightly
* Location is optionally matched and weighted minimally

Final baseline score:

```
base_score = 0.90 * semantic_score
           + 0.07 * normalized_connections
           + 0.03 * location_match
```

Then:

* Duplicate job title penalty applied
* Final baseline normalized to [0,1]

### Why minimal weighting?

Connections and location can correlate with privilege, geography, or seniority bias.
Therefore, they are included only as tie-breakers, not primary ranking signals.

---

# 2 Adaptive Re-Ranking via Similarity Feedback

When a candidate is starred:

1. Job titles are vectorized using TF-IDF.
2. Cosine similarity is computed between the starred candidate and all others.
3. Similarity is fused with the baseline score.

Final adaptive score:

```
adaptive_score = (1 - α) * baseline_score
               + α * similarity_to_star
```

Where:

```
α = 0.4
```

This allows the system to mimic recruiter behavior:

* Star an aspirant → similar aspirants move up
* Star a senior specialist → senior roles dominate
* Star a director → executive-level roles rise

This implements a classical relevance feedback mechanism.

---

# Quantitative Behavioral Validation

For each starring scenario, we measure:

* Rank of starred candidate before vs after
* Mean similarity of top 10 candidates before vs after
* Average rank movement of similar candidates
* Average rank movement of dissimilar candidates

## Result

| Scenario (id) | Star Rank Before | Star Rank After | Mean Top10 Similarity Before | Mean After | Similar Candidates Movement| Disimilar Candidates Movement |
| -------- | ---------------- | --------------- | ---------------------------- | ---------- | ----- | ------ |
| Starred Director HR | 19 | 1 | 0.1233 | 0.2375 | 8.7 | -0.93 |
| Starred Senior HR | 45 | 5 | 0.1233 | 0.6823 | -1.6 | 0.17 |
| Starred Aspiring HR Professional (3) | 70 | 8 | 0.1233 | 0.7621 | 41.7 | -4.44 |
| Non HR (80) | 87 | 40 | 0.1233 | 0.0 | -0.7 | 0.7 |

This confirms measurable improvement in semantic alignment.

The system does not just change ranking — it increases semantic coherence.

---

# Soft Filtering Strategy

We do **not** hard-remove candidates.

Instead, we apply soft filtering after adaptive ranking.

A candidate is filtered out only if:

* Their baseline score falls below a dynamic percentile cutoff, AND
* Their similarity to the starred candidate is below a similarity threshold.

```
base_score < 70th_percentile
AND
similarity < 0.10
```

This ensures:

* Clearly irrelevant profiles are removed
* Borderline candidates can survive if semantically similar
* No premature elimination occurs

---

# Why Percentile-Based Cutoff?

Absolute thresholds (e.g., score > 0.6) do not generalize across roles.

Score distributions vary by:

* Keyword density
* Role specificity
* Domain overlap

Using percentiles ensures:

* Distribution awareness
* Role-agnostic behavior
* Stable shortlist sizes
* No reliance on arbitrary score assumptions

---

# Exploratory Data Analysis Summary

* 104 candidates
* 77 HR-related profiles
* 20 senior profiles
* 54 entry-level / aspirational profiles
* Significant duplicate job titles
* No labeled `fit`

Conclusion:

This is a ranking-within-domain problem, not classification.

Supervised learning was intentionally avoided due to lack of ground truth.

---

# Bias Mitigation Strategy

To reduce systemic bias:

* Location is minimally weighted
* Connections are minimally weighted
* No hard filtering by domain
* Adaptive learning shifts control to recruiter behavior
* Duplicate penalty prevents monoculture dominance

Future improvements:

* Blind ranking mode
* Multi-star consensus weighting
* Embedding-based semantic models
* Fairness auditing if demographic data becomes available

---

# Repository Structure

```
.
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_base_model.ipynb
│
├── src/                (planned)
│   ├── ranker.py
│   ├── adaptive.py
│   ├── filter.py
│
├── Dockerfile          (planned)
├── requirements.txt
└── README.md
```

---

# Reproducibility Plan

The project will be containerized using Docker to ensure:

* Deterministic ranking pipeline
* Clean dependency management
* Environment consistency

Note:

The dataset cannot be publicly shared due to company restrictions.
Users must mount the dataset locally when running the container.


# Status

✔ Exploratory phase complete
✔ Baseline ranking implemented
✔ Adaptive re-ranking validated
✔ Soft filtering added
➡ Transitioning to modular `src/` implementation

---