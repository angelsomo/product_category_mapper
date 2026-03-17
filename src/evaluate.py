import pandas as pd

from src.io_utils import ensure_columns


REQUIRED_EVAL_COLUMNS = [
    "product_id",
    "master_category_id",
    "candidate_1_id",
    "candidate_2_id",
    "candidate_3_id",
    "review_flag",
]


def evaluate_predictions(predictions_df: pd.DataFrame) -> dict:
    """
    Evaluate category suggestion quality using labeled examples.

    Expected columns
    ----------------
    - product_id
    - master_category_id
    - candidate_1_id
    - candidate_2_id
    - candidate_3_id
    - review_flag

    Returns
    -------
    dict
        Dictionary of evaluation metrics.
    """
    ensure_columns(predictions_df, REQUIRED_EVAL_COLUMNS, df_name="Evaluation DataFrame")

    working_df = predictions_df.copy()
    working_df["master_category_id"] = working_df["master_category_id"].astype(str)

    top_1_correct = (
        working_df["candidate_1_id"].astype(str) == working_df["master_category_id"]
    )

    top_3_correct = (
        (working_df["candidate_1_id"].astype(str) == working_df["master_category_id"]) |
        (working_df["candidate_2_id"].astype(str) == working_df["master_category_id"]) |
        (working_df["candidate_3_id"].astype(str) == working_df["master_category_id"])
    )

    auto_approved_df = working_df[~working_df["review_flag"]].copy()

    if len(auto_approved_df) > 0:
        auto_approved_accuracy = (
            auto_approved_df["candidate_1_id"].astype(str) ==
            auto_approved_df["master_category_id"]
        ).mean()
    else:
        auto_approved_accuracy = 0.0

    metrics = {
        "n_products": len(working_df),
        "top_1_accuracy": round(float(top_1_correct.mean()), 4),
        "top_3_accuracy": round(float(top_3_correct.mean()), 4),
        "review_rate": round(float(working_df["review_flag"].mean()), 4),
        "auto_approved_count": int(len(auto_approved_df)),
        "auto_approved_top_1_accuracy": round(float(auto_approved_accuracy), 4),
    }

    return metrics


def format_metrics_report(metrics: dict) -> str:
    """
    Convert evaluation metrics into a readable text report.
    """
    lines = [
        "Evaluation Report",
        "-----------------",
        f"Number of products: {metrics['n_products']}",
        f"Top-1 accuracy: {metrics['top_1_accuracy']}",
        f"Top-3 accuracy: {metrics['top_3_accuracy']}",
        f"Review rate: {metrics['review_rate']}",
        f"Auto-approved predictions: {metrics['auto_approved_count']}",
        f"Auto-approved Top-1 accuracy: {metrics['auto_approved_top_1_accuracy']}",
    ]
    return "\n".join(lines)