from src.category_index import build_category_index, load_categories
from src.confidence import apply_confidence_rules
from src.evaluate import evaluate_predictions, format_metrics_report
from src.io_utils import load_csv, save_csv
from src.ranker import rank_products_to_categories


def main() -> None:
    categories_path = "data/raw/master_categories.csv"
    products_path = "data/raw/labeled_products.csv"
    predictions_path = "outputs/predictions.csv"
    review_path = "outputs/needs_review.csv"
    report_path = "outputs/evaluation_report.txt"

    categories_df = load_categories(categories_path)
    category_index_df = build_category_index(categories_df)

    products_df = load_csv(products_path)
    predictions_df = rank_products_to_categories(
        products_df=products_df,
        category_index_df=category_index_df,
        top_k=3,
    )

    predictions_df = predictions_df.merge(
        products_df[["product_id", "master_category_id"]],
        on="product_id",
        how="left",
    )

    predictions_df = apply_confidence_rules(predictions_df)

    review_df = predictions_df[predictions_df["review_flag"]].copy()

    metrics = evaluate_predictions(predictions_df)
    report_text = format_metrics_report(metrics)

    print("\nTop category suggestions:")
    print(predictions_df)

    print("\nNeeds review:")
    print(review_df[["product_id", "clean_title", "confidence", "review_reason"]])

    print("\n" + report_text)

    save_csv(predictions_df, predictions_path)
    save_csv(review_df, review_path)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nSaved predictions to: {predictions_path}")
    print(f"Saved review cases to: {review_path}")
    print(f"Saved evaluation report to: {report_path}")


if __name__ == "__main__":
    main()