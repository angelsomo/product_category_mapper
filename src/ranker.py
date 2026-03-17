import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.io_utils import ensure_columns
from src.text_normalization import normalize_mapper_text


REQUIRED_PRODUCT_COLUMNS = ["product_id", "clean_title"]


def rank_products_to_categories(
    products_df: pd.DataFrame,
    category_index_df: pd.DataFrame,
    top_k: int = 3,
) -> pd.DataFrame:
    """
    Rank the most likely categories for each product title using
    TF-IDF + cosine similarity.
    """
    ensure_columns(products_df, REQUIRED_PRODUCT_COLUMNS, df_name="Products DataFrame")
    ensure_columns(
        category_index_df,
        ["master_category_id", "category_name", "category_path", "match_text"],
        df_name="Category index DataFrame",
    )

    working_products = products_df.copy()
    working_products["clean_title"] = (
        working_products["clean_title"]
        .fillna("")
        .astype(str)
        .apply(normalize_mapper_text)
    )

    category_texts = category_index_df["match_text"].fillna("").astype(str).tolist()
    product_texts = working_products["clean_title"].tolist()

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    category_matrix = vectorizer.fit_transform(category_texts)
    product_matrix = vectorizer.transform(product_texts)

    similarity_matrix = cosine_similarity(product_matrix, category_matrix)

    results = []

    for row_position, (_, product_row) in enumerate(working_products.iterrows()):
        similarity_scores = similarity_matrix[row_position]
        ranked_indices = similarity_scores.argsort()[::-1][:top_k]

        result_row = {
            "product_id": product_row["product_id"],
            "clean_title": product_row["clean_title"],
        }

        for rank_position, category_idx in enumerate(ranked_indices, start=1):
            category_row = category_index_df.iloc[category_idx]
            score = float(similarity_scores[category_idx])

            result_row[f"candidate_{rank_position}_id"] = category_row["master_category_id"]
            result_row[f"candidate_{rank_position}_name"] = category_row["category_name"]
            result_row[f"candidate_{rank_position}_path"] = category_row["category_path"]
            result_row[f"candidate_{rank_position}_score"] = round(score, 4)

        results.append(result_row)

    return pd.DataFrame(results)