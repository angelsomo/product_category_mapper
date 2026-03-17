import pandas as pd


def apply_confidence_rules(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add confidence, predicted category, and review routing fields
    based on top candidate scores.
    """
    working_df = predictions_df.copy()

    working_df["predicted_category_id"] = working_df["candidate_1_id"]
    working_df["predicted_category_name"] = working_df["candidate_1_name"]
    working_df["confidence"] = working_df["candidate_1_score"]

    review_flags = []
    review_reasons = []

    for _, row in working_df.iterrows():
        top_score = float(row["candidate_1_score"])
        second_score = float(row["candidate_2_score"])
        margin = top_score - second_score
        title = str(row["clean_title"]).strip()

        reasons = []

        if top_score == 0.0:
            reasons.append("no_match_signal")
        elif top_score < 0.25:
            reasons.append("low_top_score")

        if margin < 0.05:
            reasons.append("small_margin_between_top_candidates")

        if len(title.split()) <= 1:
            reasons.append("short_or_ambiguous_title")

        if reasons:
            review_flags.append(True)
            review_reasons.append("; ".join(reasons))
        else:
            review_flags.append(False)
            review_reasons.append("")

    working_df["review_flag"] = review_flags
    working_df["review_reason"] = review_reasons

    return working_df