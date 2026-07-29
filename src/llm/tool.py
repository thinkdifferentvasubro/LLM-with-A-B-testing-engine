import sys
import os
import pandas as pd
from langchain_core.tools import tool
from typing import Union, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from stats_.preprocessing import Preprocessor
from stats_.ab_test_selector import ABTestSelector
from RAG.rag_pipeline import VectorDBManager

@tool
def run_ab_test_analysis(
    episodes: list[dict],
    Covariate_cols: list[str],
    num_col_with_str_vals: dict,
    date_and_format: dict,
    cat_cols: list[str],
    num_cols: list[str],
    csv_path: str
):
    """
Run an A/B test by preprocessing the data and executing statistical tests.
Decision rules (which test, what counts as a covariate, tail semantics, etc.)
live in the system prompt — this docstring only defines shapes/types.

Args:
    episodes (list[dict]): One comparison per episode.

        pairs (dict):
            {
                column: {
                    "control_value": str,
                    "treatment": [[column, value], ...]
                }
            }
            Schema values only. See system prompt for how many
            allocation columns/episodes to create.
        
        Covariate_cols (list[str]):
        Pre-treatment covariate columns to adjust for or balance-check.
        Contains only schema column names. See the system prompt for the
        eligibility rules (pre-treatment variables only; never allocation
        columns, outcome metrics, identifiers, or free-text columns).

        metrics (list[str]):
            Metric column(s) for this episode.

        p_value (float):
            User's significance level. Default 0.05 if unspecified.

        test (str):
            "", "ttest", "welchttest", "mannwhitneyu",
            "anova", "kruskalwallis", "fisherexact",
            "ztest", "chisquare", "manova", "gtest"
            See system prompt for selection rule.

        tail (str):
            "", "greater", "less"
            Always control vs treatment. See system prompt for
            normalization table.

    covariate_imbalance (dict[str, float]):
        Standardized Mean Difference (SMD) per covariate between control
        and treatment groups, computed during preprocessing (df_warnings).
        Raw values, not a pre-written warning string — severity is reasoned
        about downstream rather than assumed. See system prompt for the
        flagging threshold and how to phrase severity.

    num_col_with_str_vals (dict[str, str]):
        Regex for extracting numbers from mixed text/numeric columns
        (e.g. "10 kg", "3 apples").

    date_and_formats (dict[str, str]):
        Detected date columns and inferred formats.

    cat_cols (list[str]):
        Schema columns (allocation, metric, or covariate) classified as
        categorical. See system prompt for classification rule.

    num_cols (list[str]):
        Schema columns (allocation, metric, or covariate) classified as
        numeric. See system prompt for classification rule.

Returns:
    Statistical test results for each episode as a string.
    """
    dataset, df_warnings = Preprocessor(
    episodes=episodes,
    Covariate_cols=Covariate_cols,
    num_col_with_str_vals=num_col_with_str_vals,
    csv_paths=csv_path,
    date_and_formats=date_and_format,
    cat_cols=cat_cols,
    num_cols=num_cols
    ).run_pipeline()

    result = ABTestSelector(
    episodes = episodes,
    dataset=dataset,
    cat_columns=cat_cols,
    num_columns=num_cols
    ).run_pipeline()

    return {"test_results": result, "SMD_results": df_warnings}

@tool
def generate_csv_schema(csv_path: str):
    """
    For each column, returns a dict containing:
        - column_name: name of the column
        - column_type: pandas dtype as string
        - If n_unique <= 10:
            - "unique values": list of all unique values in the column
        - If n_unique > 10:
            - "number of unique values": number of unique values
            - "total number of values": total row count in the combined dataframe
            - "subset": random sample of 10 unique values (random_state=42)

    Returns:
        list[dict]
    """
    df = pd.read_csv(csv_path)
    total_rows = df.shape[0]
    sample_size = 10
    random_state = 42

    schema = []
    for col in df.columns:
        series = df[col].dropna()
        unique_vals = series.unique()
        total_unique_vals = series.nunique()
        n_unique = len(unique_vals)

        if n_unique <= sample_size:
            values = unique_vals.tolist()
            schema.append({
                        "column_name": col,
                        "column_type": str(df[col].dtype),
                        "unique values": values
                    })
        else:
            values = (
                pd.Series(unique_vals)
                .sample(n=sample_size, random_state=random_state)
                .tolist()
            )
            schema.append({
                        "column_name": col,
                        "column_type": str(df[col].dtype),
                        "number of unique values": total_unique_vals,
                        "total number of values": total_rows,
                        "subset": values
                    })

    return schema

@tool
def rag_search(query: str, n_results: int) -> str:
    """
    Retrieve A/B test / covariate balance results from the vector store.
    Embedding model (all-MiniLM-L6-v2) is weak — do NOT pass the raw user
    question. Rewrite it into a short keyword query matching stored doc
    vocabulary before calling.
    Stored doc vocabulary: "Covariate Balance Results", "standardized mean
    difference (SMD)", "A/B Test Episode", "control group", "treatment group",
    "metric", "Statistical Test Results", test name (e.g. mann_whitney_u),
    "p value", "significant".

    Rewrite rules:
        - Map vague terms to exact keywords above.
        - Keep group/metric values if mentioned.
        - 5-15 keywords, no filler, no full sentences.

    Note: each stored entry is one comparison. If the user wants multiple
    tests, they are stored and retrieved as separate entries, not merged.

    Args:
        query (str): Refined keyword query, not raw user text.
        n_results (int): Number of result entries to retrieve (1-5). Use 1
            for a single test; use more if the user wants multiple tests.

    Returns:
        str
    """
    rag_results = VectorDBManager().retrieve_data(
        query=query,
        n_results=n_results
        )
    final_rag_result = ""
    for rag_result in rag_results["documents"]:
        final_rag_result += rag_result[0]
    return final_rag_result