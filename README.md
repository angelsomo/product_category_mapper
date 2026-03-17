# NLP-Assisted Product Category Mapper

A hybrid NLP pipeline that maps cleaned retail product titles to
structured master categories using similarity-based ranking, confidence
scoring, and review routing.

------------------------------------------------------------------------

## Overview

This project simulates a real-world catalog enrichment problem:

Given a product title, what is the correct category?

Instead of forcing a single prediction, the system:

-   Suggests top 3 candidate categories
-   Assigns a confidence score
-   Routes uncertain cases to manual review

This makes it suitable for semi-automated workflows, not just static
classification.

------------------------------------------------------------------------

## Pipeline overview

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

## Features

### Text Processing
- Unicode normalization
- Accent removal (Greek/Latin compatibility)
- Punctuation handling
- Consistent normalization across products and categories

### Category Matching
- TF-IDF vectorization (unigrams + bigrams)
- Cosine similarity scoring
- Top-3 candidate retrieval

### Domain Enrichment
- Category aliases (e.g. *“chips”, “crisps”, “lays”*)
- Brand-aware matching via alias expansion
- Improved recall for real-world product titles

### Confidence & Review Logic
- Score-based confidence estimation
- Margin-based ambiguity detection
- Automatic routing of low-confidence cases

### Evaluation
- Top-1 accuracy
- Top-3 accuracy
- Review rate
- Accuracy of auto-approved predictions

------------------------------------------------------------------------

## Project Structure

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
## Example Input

```
product_id,clean_title
p1,lays oven baked paprika 125g
p2,coca cola zero 330ml
p3,fage greek yogurt 2 percent
p4,gourmet gold cat wet food salmon
p5,tuc crackers classic
p6,flora original 1lt
```

------------------------------------------------------------------------
## Example Output

```
| product_id | predicted_category | confidence | review_flag                     |
| ---------- | ------------------ | ---------- | ------------------------------- |
| p2         | Cola Soft Drinks   | 0.59       | False                           |
| p3         | Greek Yogurt       | 0.81       | False                           |
| p6         | Margarine          | 0.19       | True *(after threshold tuning)* |
```

------------------------------------------------------------------------
## Sample Evaluation Results

```text
Top-1 accuracy: 1.0
Top-3 accuracy: 1.0
Review rate: 0.33
Auto-approved Top-1 accuracy: 1.0
```
Note: Metrics are based on a small synthetic dataset for demonstration.
------------------------------------------------------------------------
## How to Run

```bash
pip install -r requirements.txt python -m src.pipeline
```

Outputs: 
- predictions.csv 
- needs_review.csv 
- evaluation_report.txt