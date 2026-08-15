import pandas as pd
from preprocessing import Preprocessor
from ab_test_selector import ABTestSelector


df = pd.read_csv(r"C:\projects\resume\marketing_AB.csv")
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
  dataset=df,
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
  smd_warnings=df_warnings,
  user_id="dedfef"
)

result = selec.run_pipeline()
print(result)
