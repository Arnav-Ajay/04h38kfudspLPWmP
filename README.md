# Potential Talents – Adaptive Candidate Ranking System

## Problem Statement

As a talent sourcing and management company, we aim to automate the ranking of candidates for specific roles using limited profile information:

* `job_title`
* `location`
* `connections`

We want to:

1. Rank candidates by fitness for a given role (e.g., “Aspiring Human Resources”)
2. Re-rank candidates dynamically when one or multiple candidates are starred (cumulative relevance feedback)
3. Filter out clearly irrelevant candidates
4. Avoid overfitting or bias
5. Maintain interpretability

No labeled `fit` values are provided.

Therefore, this is approached as a **ranking + adaptive relevance feedback problem**, not a supervised prediction task.

---

## System Architecture Overview

The system consists of three logical layers:

1. **Deterministic Baseline Ranking**
2. **Adaptive Re-Ranking via Similarity Feedback**
3. **Soft Filtering with Dynamic Cutoff**

The implementation is modular and production-aligned via a `src/` package and Docker container.

---

### 1 Baseline Deterministic Ranking

#### Feature Signals

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

#### Minimal Additional Signals

To preserve realism while minimizing bias:

* Connections are normalized and lightly weighted
* Location matching is optionally applied and minimally weighted

Final baseline score:

```
base_score = 0.90 * semantic_score
           + 0.07 * normalized_connections
           + 0.03 * location_match
```

Then:

* Duplicate job title penalty applied
* Final baseline normalized to [0,1]

#### Why Minimal Weighting?

Connections and location can correlate with privilege, geography, or seniority bias.

They are included only as tie-breakers — not primary ranking signals — to reduce systemic bias risk.

---

### 2 Adaptive Re-Ranking via Similarity Feedback

Ranking and re-ranking are separated:

* **Ranking** happens once per query.
* **Adaptive re-ranking** happens each time one or more candidates are starred.
* Stars are cumulative and influence ranking jointly.

#### Multi-Star Relevance Feedback

When one or more candidates are starred:
* Job titles are vectorized using TF-IDF.
* Cosine similarity is computed between each starred candidate and all others.
* Similarities are aggregated.
* Aggregated similarity is fused with the baseline score.

If multiple stars exist:

```
combined_similarity = mean(similarity_to_each_star)
```

Final adaptive score:

```
adaptive_score = (1 - α) * baseline_score
               + α * combined_similarity
```

Where:

```
α = 0.4
```

#### Why Multi-Star?

This enables progressive refinement:
* Star one aspirant → aspirant profiles rise
* Star multiple senior specialists → senior roles dominate
* Star both director + senior → system converges toward executive-level profiles

The ranking becomes more semantically coherent with each starring action, without retraining.

---

### Quantitative Behavioral Validation

For each starring scenario, we measure:

* Rank of starred candidate before vs after
* Mean similarity of top 10 candidates before vs after
* Average rank movement of similar candidates
* Average rank movement of dissimilar candidates

Validation was performed both for single-star scenarios and to verify that ranking progressively improves when multiple stars are applied cumulatively.

#### Example Results

| Scenario    | Star Rank Before | Star Rank After | Mean Top10 Similarity Before | Mean After | Similar Movement | Dissimilar Movement |
| ----------- | ---------------- | --------------- | ---------------------------- | ---------- | ---------------- | ------------------- |
| Director HR | 19               | 1               | 0.1233                       | 0.2375     | 8.7              | -0.93               |
| Senior HR   | 45               | 5               | 0.1233                       | 0.6823     | -1.6             | 0.17                |
| Aspiring HR | 70               | 8               | 0.1233                       | 0.7621     | 41.7             | -4.44               |
| Non-HR      | 87               | 40              | 0.1233                       | 0.0        | -0.7             | 0.7                 |

This confirms measurable improvement in semantic alignment and adaptive behavior.

---

### 3 Soft Filtering Strategy

We do **not** hard-remove candidates.

Filtering occurs **after adaptive re-ranking**.

A candidate is filtered out only if:

* Their baseline score falls below a dynamic percentile cutoff, AND
* Their similarity to the starred candidate is below a similarity threshold.

```
base_score < 70th_percentile
AND
combined_similarity < 0.10
```

This ensures:

* Clearly irrelevant profiles are removed
* Borderline candidates can survive if semantically aligned
* No premature elimination occurs

---

## Why Percentile-Based Cutoff?

Absolute score thresholds (e.g., score > 0.6) do not generalize across roles.

Score distributions vary by:

* Keyword density
* Role specificity
* Domain overlap

Using percentiles ensures:

* Distribution awareness
* Role-agnostic behavior
* Stable shortlist sizes
* No arbitrary thresholds

---

## Exploratory Data Analysis Summary

* 104 candidates
* 77 HR-related profiles
* 20 senior profiles
* 54 entry-level / aspirational profiles
* Significant duplicate job titles
* No labeled `fit` values

Conclusion:

This is a ranking-within-domain problem.
Supervised learning was intentionally avoided.

---

## Bias Mitigation Strategy

To reduce systemic bias:

* Location minimally weighted
* Connections minimally weighted
* No hard domain exclusion
* Recruiter starring drives adaptation
* Duplicate penalty prevents monoculture dominance

Future enhancements:

* Blind ranking mode
* Multi-star consensus weighting (reduces single-star bias amplification)
* Embedding-based semantic similarity
* Fairness auditing if demographic signals become available

---

## Repository Structure

```
.
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_base_model.ipynb
│
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── data_loader.py
│   ├── scoring.py
│   ├── adaptive.py
│   ├── filtering.py
│   ├── pipeline.py
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Running with Docker

### Build Image

```
docker build -t candidate-ranker .
```

### Run Ranking

```
docker run \
  -v $(pwd)/data:/data \
  candidate-ranker \
  --data /data/potential-talents.csv \
  --output /data/baseline.csv
```

### Run Re-Ranking (Single or Multi-Star)

```
docker run \
  -v $(pwd)/data:/data \
  candidate-ranker \
  --data /data/potential-talents.csv \
  --star_ids 89 61 \ # Multiple IDs can be passed. The system aggregates similarity across all starred candidates before re-ranking.
  --output /data/reranked_multi.csv
```

Internally, this runs:

```
python -m src.main
```

---

## Reproducibility

* Fully containerized pipeline
* Deterministic ranking
* Modular architecture
* No notebook dependency for execution

Note:
The dataset cannot be publicly shared due to company restrictions.
Users must mount the dataset locally when running the container.

---