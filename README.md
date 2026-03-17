🧠 NLP-Assisted Product Category Mapper

A hybrid NLP pipeline that maps cleaned retail product titles to
structured master categories using similarity-based ranking, confidence
scoring, and review routing.

------------------------------------------------------------------------

📌 Overview

This project simulates a real-world catalog enrichment problem:

Given a product title, what is the correct category?

Instead of forcing a single prediction, the system:

-   Suggests top 3 candidate categories
-   Assigns a confidence score
-   Routes uncertain cases to manual review

This makes it suitable for semi-automated workflows, not just static
classification.

------------------------------------------------------------------------

🧱 Pipeline overview
```text

Raw Product Title
        ↓
(Phase 2) Cleaning & Normalization
        ↓
Normalized Title
        ↓
TF-IDF Vectorization
        ↓
Cosine Similarity vs Category Index
        ↓
Top-K Candidate Ranking
        ↓
Confidence Scoring
        ↓
Review Routing (if needed)
        ↓
Structured Output
```

------------------------------------------------------------------------

⚙️ Features

Text Processing: - Unicode normalization - Accent removal - Punctuation
handling

Category Matching: - TF-IDF vectorization - Cosine similarity - Top-3
retrieval

Domain Enrichment: - Category aliases - Brand-aware matching

Confidence & Review: - Score-based confidence - Margin-based ambiguity
detection - Review routing

Evaluation: - Top-1 accuracy - Top-3 accuracy - Review rate -
Auto-approved accuracy

------------------------------------------------------------------------

📂 Project Structure

```
product_category_mapper/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── src/
│   ├── io_utils.py
│   ├── text_normalization.py
│   ├── category_index.py
│   ├── ranker.py
│   ├── confidence.py
│   ├── evaluate.py
│   └── pipeline.py
│
├── outputs/
│   ├── predictions.csv
│   ├── needs_review.csv
│   └── evaluation_report.txt
│
├── config/
├── tests/
├── README.md
└── requirements.txt
```

------------------------------------------------------------------------

🧪 How to Run

pip install -r requirements.txt python -m src.pipeline

Outputs: - predictions.csv - needs_review.csv - evaluation_report.txt

------------------------------------------------------------------------

🧩 Key Takeaway

The goal is not just to predict — but to know when not to predict.