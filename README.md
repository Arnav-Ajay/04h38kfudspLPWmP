# Potential Talents – Adaptive Candidate Ranking System

## 📌 Problem Statement

As a talent sourcing and management company, we aim to automate the ranking of candidates for specific roles based on textual profile information.

Given:

* `job_title`
* `location`
* `connections`

We want to:

1. Rank candidates by fitness for a given role (e.g., “Aspiring Human Resources”)
2. Re-rank candidates dynamically when a candidate is starred (selected as ideal)
3. Reduce manual effort while preserving interpretability
4. Avoid introducing bias or overfitting to small data

No labeled `fit` values are provided.

Therefore, this problem is approached as a ranking + adaptive relevance feedback system rather than a supervised prediction problem.

---

# Solution Overview

The system consists of two main layers:

## 1. Baseline Deterministic Ranking

Based on exploratory analysis, we construct three structured signals:

* **HR Relevance Score**
* **Seniority Score**
* **Intent Score**

These signals are derived from domain-specific keywords identified during EDA.

Baseline score:

```
base_score = 0.6 * HR + 0.35 * Seniority + 0.05 * Intent
```

Additionally:

* Duplicate job title penalty applied to prevent ranking saturation.
* Scores normalized to 0–1 scale.

This ensures:

* HR-related roles rank above non-HR.
* Seniority and intent are distinguishable.
* Duplicate titles do not dominate the ranking.

---

## 2. Adaptive Re-Ranking via Similarity Feedback

When a candidate is starred:

1. Job titles are vectorized using TF-IDF.
2. Cosine similarity is computed between the starred candidate and all others.
3. Similarity is fused with baseline score.

Final adaptive score:

```
adaptive_score = 0.6 * normalized_base_score
               + 0.4 * similarity_to_star
```

This allows:

* Starring an aspirant → promotes aspirant profiles
* Starring a senior → promotes senior roles
* Starring a director → promotes executive profiles

This mimics recruiter behavior and acts as a relevance feedback loop.

---

# Exploratory Analysis Summary

Key findings from EDA:

* Dataset contains 104 candidates.
* 77 HR-related profiles.
* 20 senior profiles.
* 54 entry-level / aspirational profiles.
* Significant duplicate job titles.
* No labeled `fit` column provided.

Implication:

This is a ranking problem within a partially filtered HR domain.

Supervised modeling is inappropriate due to lack of ground truth labels.

---

# Behavioral Demonstration

The adaptive model demonstrates measurable ranking shifts:

### Star “Aspiring HR”

* Aspirant profiles rise.
* Senior roles slightly drop.

### Star “HR Senior Specialist”

* Senior specialists dominate.
* Entry-level profiles move lower.

### Star “Director HR”

* Director-level roles move to top.
* Junior and aspirant profiles drop further.

This confirms adaptive re-ranking behavior.

---

# Bias Considerations

To reduce bias:

* No hard filtering removes borderline candidates.
* Location is not heavily weighted.
* Connections are not used as primary ranking signal.
* Ranking remains explainable and traceable.

Future improvements could include:

* Blind ranking without location
* Diversity constraints
* Embedding-based semantic models

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
│
├── Dockerfile          (planned)
├── requirements.txt
└── README.md
```

---

# Reproducibility Plan

The project will be containerized using Docker to ensure:

* Reproducible ranking pipeline
* Clean dependency management
* Environment consistency

Note:

The dataset cannot be publicly shared due to company restrictions.
Users must mount the dataset locally when running the container.

---

# Next Steps

* Refactor notebook logic into modular `src/` implementation
* Build CLI interface for ranking & starring
* Containerize entire pipeline
* Add evaluation metrics (rank improvement tracking)
* Explore embedding-based ranking extensions

---

# Key Takeaways

* Structured scoring provides stable baseline ranking.
* Similarity-based re-ranking enables adaptive learning.
* The system is explainable and production-aligned.
* Works without labeled data.
* Easily extensible to other roles beyond HR.

---

# Status

Exploratory phase complete.
Transitioning to production implementation.

---