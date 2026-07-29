import os
import sys
from preprocessing import Preprocessor
from ab_test_selector import ABTestSelector

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from RAG.rag_pipeline import VectorDBManager

episodes=[
    {
      "p_value": 0.05,
      "test": "chisquare",
      "metrics": [
        "converted"
      ],
      "pairs": {
        "test group": {
          "control_value": "psa",
          "treatment": [
            [
              "test group",
              "ad"
            ]
          ]
        }
      },
      "tail": "less"
    },
    {
    "p_value": 0.05,
          "test": "",
          "metrics": [
            "total ads"
          ],
          "pairs": {
            "test group": {
              "control_value": "psa",
              "treatment": [
                [
                  "test group",
                  "ad"
                ]
              ]
            }
          },
          "tail": ""
    }
  ]
dataset, df_warnings = Preprocessor(
    episodes=episodes,
    Covariate_cols=[
    "most ads day",
    "most ads hour"
  ],
  csv_path=r"C:\projects\resume\marketing_AB.csv",
  num_col_with_str_vals={},
  date_and_formats={},
  cat_cols=[
    "test group",
    "converted",
    "most ads day"
  ],
  num_cols=[
    "total ads",
    "most ads hour"
  ]

).run_pipeline()

selec = ABTestSelector(
    episodes = episodes,
    dataset=dataset,
    cat_columns=[
    "test group",
    "converted",
    "most ads day"
  ],
    num_columns=[
    "total ads",
    "most ads hour"
  ],
  smd_warnings=df_warnings
)

result = selec.run_pipeline()

rag_results = VectorDBManager().retrieve_data(
    query=result
)

final_rag_result = ""
for rag_result in rag_results["documents"]:
  final_rag_result += rag_result[0]

print(final_rag_result)