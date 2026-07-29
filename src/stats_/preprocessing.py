import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from feature_engine.outliers import Winsorizer
import numpy as np
from typing import Union, List

class Preprocessor:
    def __init__(self,
                 episodes: list[dict],
                 Covariate_cols: list[str],
                 num_col_with_str_vals: dict,
                 csv_path: str,
                 date_and_formats: dict,
                 cat_cols: list[str],
                 num_cols: List[str]):
        
        
        self.Covariate_cols = Covariate_cols
        self.episodes = episodes
        self.num_col_with_str_vals = num_col_with_str_vals
        self.csv_path = csv_path
        self.Assignment_cols = []
        self.Final_metric_col = []
        self.date_and_formats = date_and_formats
        self.cat_cols = cat_cols
        self.num_cols = num_cols

    def run_pipeline(self):
        dataset = self.load_ab_test_data()
        self.Assignment_cols, self.Final_metric_col =  self.extract_episode_columns()
        cols = self.Assignment_cols + self.Final_metric_col + self.Covariate_cols
        dataset = dataset[cols]
        dataset = self.remove_duplicates(dataset)
        dataset = self.handling_num_col_with_str_val(dataset)
        dataset = self.extract_date_features(df=dataset)
        dataset = self.null_values_handling(dataset=dataset)
        dataset = self.outlier_handling(dataset=dataset)
        warnings = self.Covariate_Balance_Check(dataset=dataset)

        return dataset, warnings
    
    def load_ab_test_data(self):
        return pd.read_csv(self.csv_path)

    def remove_duplicates(self, dataset: pd.DataFrame):
        return dataset.drop_duplicates()
    
    def handling_num_col_with_str_val(self, dataset: pd.DataFrame):
        dataset = dataset.copy()
        if not self.num_col_with_str_vals:
            return dataset

        for column, pattern in self.num_col_with_str_vals.items():
            dataset[column] = (
                dataset[column]
                .str.extract(f"({pattern})", expand=False)
                .astype(float)
            )

        return dataset
    
    def extract_episode_columns(self):
        allocation_col = set()
        metrices = set()

        for episode in self.episodes:
            controls = list(episode["pairs"].keys())
            for control in controls:
                allocation_col.update([control])

                for treatment in episode["pairs"][control]["treatment"]:
                    allocation_col.update([treatment[0]])

            metric = episode["metrics"]
            metrices.update(metric)

        return list(allocation_col), list(metrices)

    def extract_date_features(self, df: pd.DataFrame):
        df = df.copy()
        if not self.date_and_formats:
            return df

        columns = list(self.date_and_formats.keys())
        for col in columns:
            fmt = self.date_and_formats[col]
            valid = pd.to_datetime(df[col], format=fmt, errors="raise")

            is_unreliable = (
                valid.empty
                or (valid.dropna().dt.day == 1).all()
            )

            if not is_unreliable:
                day_name = valid.dt.day_name().astype("string")
                is_weekend = valid.dt.dayofweek.isin([5, 6])

                # preserve NA where the original date was NaT
                day_name = day_name.where(valid.notna(), pd.NA)
                is_weekend = is_weekend.astype("string").where(valid.notna(), pd.NA)

                df[f'day_of_week_{col}'] = day_name
                df[f'is_weekend_{col}'] = is_weekend

                self.Covariate_cols.extend([f'day_of_week_{col}', f'is_weekend_{col}'])

            try:
                df.drop(columns=[col], inplace=True)
                self.Covariate_cols.remove(col)
            except:
                pass

        return df

    def null_values_handling(self, dataset: pd.DataFrame):
        dataset = dataset.copy()

        assignment_and_metric_cols = self.Assignment_cols + self.Final_metric_col
        dataset = dataset.dropna(subset=assignment_and_metric_cols)

        if dataset.empty:
            return dataset

        cat_cov, num_cov = self.cat_cols, self.num_cols

        less_than_15 = []
        between_15_and_60 = []
        greater_than_60 = []

        if self.Covariate_cols:
            for column in self.Covariate_cols:
                null_values_percentage = (
                    dataset[column].isna().sum() / len(dataset)
                ) * 100

                if null_values_percentage <= 15:
                    less_than_15.append(column)
                elif null_values_percentage <= 60:
                    between_15_and_60.append(column)
                else:
                    greater_than_60.append(column)

            # Drop columns with >60% missing
            if greater_than_60:
                dataset = dataset.drop(columns=greater_than_60)
                self.Covariate_cols = [
                    x for x in self.Covariate_cols if x not in greater_than_60
                ]

                self.cat_cols = [x for x in cat_cov if x not in greater_than_60]
                self.num_cols = [x for x in num_cov if x not in greater_than_60]

            # <=15% missing
            if less_than_15:

                cat_columns_ = [x for x in less_than_15 if x in cat_cov]
                if cat_columns_:
                    cat_imputer = SimpleImputer(strategy="most_frequent")
                    dataset[cat_columns_] = cat_imputer.fit_transform(dataset[cat_columns_])

                num_columns_ = [x for x in less_than_15 if x in num_cov]
                if num_columns_:
                    num_imputer = SimpleImputer(strategy="median")
                    dataset[num_columns_] = num_imputer.fit_transform(dataset[num_columns_])

            # 15-60% missing
            if between_15_and_60:

                num_columns_ = [x for x in between_15_and_60 if x in num_cov]
                if num_columns_:
                    scaler = MinMaxScaler()
                    knn_imputer = KNNImputer(n_neighbors=5)

                    dataset[num_columns_] = scaler.fit_transform(dataset[num_columns_])
                    dataset[num_columns_] = knn_imputer.fit_transform(dataset[num_columns_])
                    dataset[num_columns_] = scaler.inverse_transform(dataset[num_columns_])

                cat_columns_ = [x for x in between_15_and_60 if x in cat_cov]
                if cat_columns_:
                    dataset[cat_columns_] = dataset[cat_columns_].fillna("Missing")

        return dataset
    
    def outlier_handling(self, dataset: pd.DataFrame):
        dataset=dataset.copy()
        num_covariate_cols = [x for x in self.Covariate_cols if x in self.num_cols]
        if not num_covariate_cols:
            return dataset
        
        win_ = Winsorizer(
            capping_method="iqr",
            tail="both"
        )

        num_covariate_cols = [x for x in self.Covariate_cols if x in self.num_cols]
        
        dataset[num_covariate_cols] = win_.fit_transform(dataset[num_covariate_cols])
        return dataset
    

    def _smd_numeric(self, control_vals, treatment_vals):
        m1, m0 = np.mean(treatment_vals), np.mean(control_vals)
        v1, v0 = np.var(treatment_vals, ddof=1), np.var(control_vals, ddof=1)
        pooled_sd = np.sqrt((v1 + v0) / 2)
        return float((m1 - m0) / pooled_sd) if pooled_sd > 0 else np.nan

    def _smd_categorical(self, control_series, treatment_series):
        levels = set(control_series.unique()) | set(treatment_series.unique())
        max_smd = 0.0

        for level in levels:
            p1 = (treatment_series == level).mean()
            p0 = (control_series == level).mean()

            denom = np.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2)

            if denom > 0:
                smd = abs((p1 - p0) / denom)
                max_smd = max(max_smd, smd)

        return float(max_smd)

    def Covariate_Balance_Check(self, dataset: pd.DataFrame):
        if not self.Covariate_cols:
            return "no covariates to check SMD"
        dataset=dataset.copy()
        allocation_pairs = []
        results = {}

        for episode in self.episodes:
            controls = list(episode["pairs"].keys())
            for control in controls:
                for treatment in episode["pairs"][control]["treatment"]:
                    treatment_name = treatment[0]
                    control_value = episode["pairs"][control]["control_value"]
                    treatment_value = treatment[1]
                    treatment_value_type = type(treatment_value).__name__
                    control_value_type = type(control_value).__name__

                    pair = f"{control}||{control_value}||{control_value_type}|||{treatment_name}||{treatment_value}||{treatment_value_type}"
                    if pair in allocation_pairs:
                        continue
                    allocation_pairs.append(pair)

                    control_df = dataset[dataset[control] == control_value]
                    treatment_df = dataset[dataset[treatment_name] == treatment_value]

                    num_cols = [x for x in self.Covariate_cols if x in self.num_cols]
                    cat_cols = [x for x in self.Covariate_cols if x in self.cat_cols]

                    pair_result = {}
                    for col in num_cols:
                        pair_result[col] = self._smd_numeric(
                            control_df[col], treatment_df[col]
                        )
                    for col in cat_cols:
                        pair_result[col] = self._smd_categorical(
                            control_df[col], treatment_df[col]
                        )

                    results[pair] = pair_result

        return results