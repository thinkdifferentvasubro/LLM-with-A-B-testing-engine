import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.stats_.preprocessing import Preprocessor
from src.stats_.ab_test_selector import ABTestSelector
from src.RAG.rag_pipeline import VectorDBManager

def run_ab_test_analysis(
    episodes: list[dict],
    Covariate_cols: list[str],
    num_col_with_str_vals: dict,
    date_and_formats: dict,
    cat_cols: list[str],
    num_cols: list[str],
    dataset: pd.DataFrame,
    user_id: str
):
    
    dataset_p, df_warnings = Preprocessor(
    episodes=episodes,
    Covariate_cols=Covariate_cols,
    num_col_with_str_vals=num_col_with_str_vals,
    dataset=dataset,
    date_and_formats=date_and_formats,
    cat_cols=cat_cols,
    num_cols=num_cols
    ).run_pipeline()

    result = ABTestSelector(
    episodes = episodes,
    dataset=dataset_p,
    cat_columns=cat_cols,
    num_columns=num_cols,
    smd_warnings=df_warnings,
    user_id=user_id
    ).run_pipeline()

    return result


def generate_csv_schema(csv_path: str, sample_size: int = 10, random_state: int = 42) -> str:
    df = pd.read_csv(csv_path)
    if df.empty:
        return "dataset that user has provided is empty", None
    total_rows = df.shape[0]

    lines = []

    for col in df.columns:
        series = df[col].dropna()
        unique_vals = series.unique()
        total_unique_vals = series.nunique()
        n_unique = len(unique_vals)

        lines.append(f"\nColumn: {col}")
        lines.append(f"  Type: {df[col].dtype}")
        lines.append(f"  Non-null count: {series.shape[0]} / {total_rows}")
        lines.append(f"  Unique values: {total_unique_vals}")

        if n_unique <= sample_size:
            values = unique_vals.tolist()
            lines.append(f"  All unique values: {values}")
        else:
            sample_values = (
                pd.Series(unique_vals)
                .sample(n=sample_size, random_state=random_state)
                .tolist()
            )
            lines.append(f"  Sample values ({sample_size} of {total_unique_vals}): {sample_values}")

    return "\n".join(lines), df

def rag_search(query: str, n_results: int, user_id: str) -> str:

    rag_results = VectorDBManager().retrieve_data(
        query=query,
        n_results=n_results,
        user_id=user_id
        )
    final_rag_result = "Rag results: "
    for rag_result in rag_results["documents"]:
        if not rag_result:
            return "could not find the related test/experiment"
        final_rag_result += rag_result[0]
    return final_rag_result
