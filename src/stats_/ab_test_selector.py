import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.multivariate.manova import MANOVA
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportions_ztest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.RAG.rag_pipeline import VectorDBManager


class ABTestSelector:

    def __init__(self,
        episodes: list[dict],
        dataset: pd.DataFrame,
        cat_columns: list,
        num_columns: list,
        smd_warnings: dict,
        user_id: str
    ):
        self.episodes = episodes
        self.dataset = dataset
        self.cat_columns = cat_columns
        self.num_columns = num_columns
        self.smd_warnings = smd_warnings
        self.user_id = user_id

    def run_pipeline(self):
        results = ""
        for episode in self.episodes:
            if not episode:
                result = "could not find proper schema"
            df = self.dataset.copy()
            mask, allocation_columns = self.extract_mask_from_episode(episode=episode, dataset=df)
            df = df[mask]
            metrics = episode["metrics"]
            p_value = episode["p_value"]
            tail = episode["tail"]
            test = episode["test"]

            if test:
                result = self.run_selected_test(
                    test_name = test,
                    df = df,
                    episode=episode,
                    allocation_col = allocation_columns,
                    metric_col = metrics,
                    alpha = p_value,
                    tail=tail
                )

            elif len(metrics) == 1:
                metric = metrics[0]
                if metric in self.cat_columns:
                    if tail:
                        result = self.one_tailed_categorical_test(
                            df=df,
                            allocation_col=allocation_columns,
                            episode=episode,
                            metric_col=metric,
                            alpha=p_value
                        )
                    else:
                        result = self.two_tailed_categorical_test(
                            df=df,
                            allocation_col=allocation_columns,
                            metric_col=metric,
                            alpha=p_value
                        )
                
                elif metric in self.num_columns:
                    if tail:
                        result = self.one_tailed_numeric_test(
                            df=df,
                            allocation_col=allocation_columns,
                            metric_col=metric,
                            episode=episode,
                            alpha=p_value
                        )
                    else:
                        result = self.two_tailed_numeric_test(
                            df=df,
                            allocation_col=allocation_columns,
                            metric_col=metric,
                            alpha=p_value
                        )
            elif len(metrics) > 1:
                if all(x in self.cat_columns for x in metrics):
                    result = self.multi_categorical_gtest(
                        df=df,
                        allocation_col=allocation_columns,
                        metric_col=metrics,
                        alpha=p_value
                    )
                
                elif(all(x in self.num_columns for x in metrics)):
                    result = self.mannova_test(
                        df=df,
                        allocation_col=allocation_columns,
                        metric_col=metrics,
                        alpha=p_value

                    )

            if not isinstance(result, str):
                out = self.convert_to_str(episode=episode, result=result)
                results += out
                VectorDBManager().save_data(
                    document=out,
                    user_id=self.user_id
                    )
        return results

    def extract_mask_from_episode(self, episode: dict, dataset: pd.DataFrame):
        allocation_columns = set()
        controls = list(episode["pairs"].keys())
        allocation_columns.update(controls)
        mask = pd.Series(False, index=dataset.index)
        for control in controls:
            control_value = episode["pairs"][control]["control_value"]
            mask = mask | (dataset[control] == control_value)

            for treatment in episode["pairs"][control]["treatment"]:
                treatment_name = treatment[0]
                treatment_value = treatment[1]
                mask = mask | (dataset[treatment_name] == treatment_value)
                allocation_columns.update([treatment_name])

        return mask, list(allocation_columns)
    
    def is_normal(self, series: pd.Series, alpha=0.05):
        if len(series) < 3:
            return False
        stat, p = stats.shapiro(series)
        return p > alpha

    def convert_to_str(self, episode: dict, result: dict):
        out = ""
        control_name = next(iter(episode["pairs"]))
        control_value = episode["pairs"][control_name]["control_value"]

        if type(self.smd_warnings).__name__ != "str":
            out += "Test results:\n\nCovariate Balance Results:\n"

            for smd_pair, values in self.smd_warnings.items():
                smd_control_pair, smd_treatment_pair = smd_pair.split("|||")
                smd_control_name, smd_control_value, smd_control_type = smd_control_pair.split("||")
                smd_treatment_name, smd_treatment_value, smd_treatment_type = smd_treatment_pair.split("||")

                smd_control_value = int(smd_control_value) if smd_control_type=="int" else smd_control_value
                smd_control_value = float(smd_control_value) if smd_control_type=="float" else smd_control_value

                smd_treatment_value = int(smd_treatment_value) if smd_treatment_type=="int" else smd_treatment_value
                smd_treatment_value = float(smd_treatment_value) if smd_treatment_type=="float" else smd_treatment_value

                treatments = episode["pairs"][control_name]["treatment"]
                for treatment in treatments:
                    treatment_name = treatment[0]
                    treatment_value = treatment[1]
                    if (control_name==smd_control_name and control_value==smd_control_value and treatment_name==smd_treatment_name and treatment_value==smd_treatment_value):
                        out += f"Comparison of {smd_control_name} for value {smd_control_value} with {smd_treatment_name} for value {smd_treatment_value}\n"
                        for key, value in values.items():
                            out += (
                                f"The covariate '{key}' has a standardized mean difference "
                                f"(SMD) value of {value}.\n"
                            )

                    out += "\n"
        else:
            out += self.smd_warnings + "\n"

        out += "A/B Test Episode:\n\n"

        metric = episode["metrics"]
        metric = " ".join(metric)
        p_value = episode["p_value"]

        out += (
            f"This experiment compares the control group "
            f"{control_name} with value '{control_value}' and the metric is {metric} with p value {p_value}.\n"
        )
        test = episode["test"] 
        if test:
            out += f"the user selects {test}"

        treatments = episode["pairs"][control_name]["treatment"]

        for treatment in treatments:
            treatment_name = treatment[0]
            treatment_value = treatment[1]

            out += (
                f"The treatment group is {treatment_name} "
                f"with value '{treatment_value}'.\n"
            )

        out += "\nStatistical Test Results:\n\n"
        for key, value in result.items():
            readable_key = key.replace("_", " ")
            out += f"The {readable_key} was {value}.\n"

        return out
    
    def two_tailed_categorical_test(self, df: pd.DataFrame, allocation_col: list, metric_col: str, alpha=0.05):
        sub = df[allocation_col + [metric_col]]
        table = pd.crosstab(
            [sub[c] for c in allocation_col],
            sub[metric_col]
        )
        n_groups, n_categories = table.shape

        if n_groups < 2 or n_categories < 2:
            return f"Need >=2 groups and >=2 categories; got {n_groups}x{n_categories}."


        chi2, p_value, dof, expected = stats.chi2_contingency(table)
        low_expected = (expected < 5).any()

        if n_groups == 2 and n_categories == 2:

            if low_expected:
                odds_ratio, p_value_fisher = stats.fisher_exact(table, alternative="two-sided")

                return {
                    "test": "fisher_exact",
                    "statistic": float(odds_ratio),
                    "p_value": float(p_value_fisher),
                    "significant": bool(p_value_fisher < alpha),
                }

        return {
            "test": "chi_square",
            "statistic": float(chi2),
            "p_value": float(p_value),
            "dof": dof,
            "low_expected_counts": low_expected, 
            "significant": bool(p_value < alpha)
        }

    def one_tailed_categorical_test(self, df: pd.DataFrame, allocation_col: list, metric_col: str, episode: dict, alpha=0.05):
        df = df[allocation_col + [metric_col]]
        control_col = list(episode["pairs"].keys())[0]
        control_value = episode["pairs"][control_col]["control_value"]
        treatment_col, treatment_value = episode["pairs"][control_col]["treatment"][0]
        tail = episode["tail"]

        control_mask = df[control_col] == control_value
        treatment_mask = df[treatment_col] == treatment_value

        sub = df[control_mask | treatment_mask][[control_col, treatment_col, metric_col]].copy()
        sub["group"] = None
        sub.loc[control_mask[control_mask | treatment_mask], "group"] = "control"
        sub.loc[treatment_mask[control_mask | treatment_mask], "group"] = "treatment"

        table = pd.crosstab(sub["group"], sub[metric_col])

        n_groups, n_categories = table.shape
        if n_groups != 2 or n_categories != 2:
            return "groups and categories should be equal to 2 for one tailed test"

        table = table.loc[["control", "treatment"]]
        odds_ratio, p_value_fisher = stats.fisher_exact(table.values, alternative=tail)

        return {
            "test": "fisher_exact",
            "statistic": float(odds_ratio),
            "p_value": float(p_value_fisher),
            "control": {"column": control_col, "value": control_value},
            "treatment": {"column": treatment_col, "value": treatment_value},
            "category_compared": table.columns[0],
            "tail": tail,
            "significant": bool(p_value_fisher < alpha),
        }
        
    def two_tailed_numeric_test(self, df: pd.DataFrame, allocation_col: list, metric_col: str, alpha=0.05):
        sub = df[allocation_col + [metric_col]]

        grouped = [
            group[metric_col]
            for _, group in sub.groupby(allocation_col)
        ]

        n_groups = len(grouped)

        if n_groups < 2:
            return f"Need >=2 groups; got {n_groups}."

        normal = all(self.is_normal(series) for series in grouped)

        if n_groups == 2:
            
            if normal:
                statistic, p_value = stats.ttest_ind(
                    grouped[0],
                    grouped[1],
                    equal_var=False,
                    alternative="two-sided"
                )

                return {
                    "test": "welch_t_test",
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "significant": bool(p_value < alpha),
                }

            statistic, p_value = stats.mannwhitneyu(
                grouped[0],
                grouped[1],
                alternative="two-sided"
            )

            return {
                "test": "mann_whitney_u",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "significant": bool(p_value < alpha),
            }

        if normal:
            statistic, p_value = stats.f_oneway(*grouped)

            return {
                "test": "one_way_anova",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "significant": bool(p_value < alpha),
            }

        statistic, p_value = stats.kruskal(*grouped)

        return {
            "test": "kruskal_wallis",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": bool(p_value < alpha),
        }

    def one_tailed_numeric_test(self, df: pd.DataFrame, allocation_col: list, metric_col: str, episode: dict, alpha=0.05):
        df = df[allocation_col + [metric_col]]
        control_col = list(episode["pairs"].keys())[0]
        control_value = episode["pairs"][control_col]["control_value"]
        treatment_col, treatment_value = episode["pairs"][control_col]["treatment"][0]
        tail = episode["tail"]

        control_mask = df[control_col] == control_value
        treatment_mask = df[treatment_col] == treatment_value

        sub = df[control_mask | treatment_mask][[control_col, treatment_col, metric_col]].copy()
        sub["group"] = None
        sub.loc[control_mask[control_mask | treatment_mask], "group"] = "control"
        sub.loc[treatment_mask[control_mask | treatment_mask], "group"] = "treatment"

        group1 = sub[sub["group"] == "control"][metric_col]
        group2 = sub[sub["group"] == "treatment"][metric_col]
        
        if group1.empty or group2.empty:
            return "control and treatment groups must both be non-empty for one tailed test"

        group1_norm = self.is_normal(group1)
        group2_norm = self.is_normal(group2)


        if group1_norm and group2_norm:
            statistic, p_value = stats.ttest_ind(
                group1,
                group2,
                alternative=tail,
                equal_var=False
            )

            return {
                "test": "welch_t_test",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "significant": bool(p_value < alpha)
            }

        statistic, p_value = stats.mannwhitneyu(
            group1,
            group2,
            alternative=tail
        )

        return {
            "test": "mann_whitney_u",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": bool(p_value < alpha)
        }

    def mannova_test(self, df: pd.DataFrame, allocation_col: list, metric_col: list, alpha=0.05):
        
        sub = df[allocation_col + metric_col]

        if len(allocation_col) == 1:
            sub = sub.copy()
            sub["_group"] = sub[allocation_col[0]].astype(str)
        else:
            sub = sub.copy()
            sub["_group"] = (
                sub[allocation_col]
                .astype(str)
                .agg("_".join, axis=1)
            )

        if sub["_group"].nunique() < 2:
            return f"Need >=2 groups; got {sub['_group'].nunique()}."

        normal = True

        for _, group in sub.groupby("_group"):
            for metric in metric_col:

                if len(group[metric]) >= 3:
                    if not self.is_normal(group[metric]):
                        normal = False
                        break

            if not normal:
                break

        if not normal:
            return {
                "test": "mannova",
                "warning": (
                    "One or more metrics failed the normality test. "
                    "MANOVA assumptions may be violated."
                )
            }

        formula = (
            " + ".join(metric_col)
            + " ~ _group"
        )

        model = MANOVA.from_formula(
            formula,
            data=sub
        )

        result = model.mv_test()

        stats_table = result.results["_group"]["stat"]

        # Pillai's Trace (most robust)
        pillai = stats_table.loc["Pillai's trace"]

        statistic = float(pillai["Value"])
        f_value = float(pillai["F Value"])
        p_value = float(pillai["Pr > F"])

        return {
            "test": "MANOVA",
            "multivariate_statistic": "Pillai's Trace",
            "statistic": float(statistic),
            "F": float(f_value),
            "p_value": float(p_value),
            "significant": bool(p_value < alpha),
            "normality_passed": normal,
            "groups": int(sub["_group"].nunique()),
            "metrics": metric_col,
        }

    def multi_categorical_gtest(self, df: pd.DataFrame, allocation_col: list, metric_col: list, alpha=0.05):
        
        sub = df[allocation_col + metric_col].copy()

        if len(allocation_col) == 1:
            sub["_group"] = sub[allocation_col[0]].astype(str)
        else:
            sub["_group"] = sub[allocation_col].astype(str).agg("_".join, axis=1)

        
        sub["_metric"] = sub[metric_col].astype(str).agg("_".join, axis=1)

        if sub["_group"].nunique() < 2:
            return f"Need >=2 groups; got {sub['_group'].nunique()}."

        table = pd.crosstab(sub["_group"], sub["_metric"])
        counts = table.stack().rename("count").reset_index()
        counts.columns = ["_group", "_metric", "count"]

        n_groups, n_categories = table.shape

        model = smf.glm(
            formula="count ~ C(_group) + C(_metric)",
            data=counts,
            family=sm.families.Poisson()
        ).fit()

        g_stat = float(model.deviance) 
        dof = int(model.df_resid)              
        p_value = float(stats.chi2.sf(g_stat, dof))

        expected = model.fittedvalues.values
        low_expected = bool((expected < 5).any())

        return {
            "test": "g_test_loglinear_glm",
            "statistic": g_stat,
            "p_value": p_value,
            "dof": dof,
            "low_expected_counts": low_expected,
            "significant": bool(p_value < alpha),
            "groups": int(n_groups),
            "joint_categories": int(n_categories),
            "metrics": metric_col,
        }
    def run_selected_test(
    self,
    test_name: str,
    episode: dict,
    tail: str,
    df: pd.DataFrame,
    allocation_col,
    metric_col,
    alpha: float = 0.05
    ):
        sub = df[allocation_col + metric_col]
        n_groups = sub.groupby(allocation_col).ngroups

        if n_groups < 2:
            return "there should be at least two groups for the test"

        if test_name in ("ttest", "welchttest", "mannwhitneyu", "anova", "kruskalwallis", "fisherexact", "ztest", "chisquare") and len(metric_col) > 1:
            return f"'{test_name}' only supports a single metric column; got {len(metric_col)}."

        # these tests have no directional variant - tail isn't valid for them
        if test_name in ("chisquare", "anova", "kruskalwallis", "manova", "gtest") and tail:
            return f"'{test_name}' does not support a one-tailed test; only two-sided is available for it."
            

        if test_name in ("fisherexact", "ztest"):
            n_categories = sub[metric_col[0]].nunique()
            if n_categories != 2:
                return f"'{test_name}' requires exactly 2 categories in '{metric_col[0]}'; got {n_categories}."

        if test_name == "chisquare":
            n_categories = sub[metric_col[0]].nunique()
            if n_categories < 2:
                return f"'chisquare' requires at least 2 categories in '{metric_col[0]}'; got {n_categories}."
            

        try:
            if test_name in ("ttest", "welchttest"):
                if tail:
                    control_col = list(episode["pairs"].keys())[0]
                    control_value = episode["pairs"][control_col]["control_value"]
                    treatment_col, treatment_value = episode["pairs"][control_col]["treatment"][0]
                    metric_col = episode["metrics"][0]
                    tail = episode["tail"]
            
                    control_mask = df[control_col] == control_value
                    treatment_mask = df[treatment_col] == treatment_value
            
                    sub = df[control_mask | treatment_mask][[control_col, treatment_col, metric_col]].copy()
                    sub["group"] = None
                    sub.loc[control_mask[control_mask | treatment_mask], "group"] = "control"
                    sub.loc[treatment_mask[control_mask | treatment_mask], "group"] = "treatment"
            
                    group1 = sub[sub["group"] == "control"][metric_col]
                    group2 = sub[sub["group"] == "treatment"][metric_col]
            
                    if group1.empty or group2.empty:
                        return "control and treatment groups must both be non-empty for one tailed test" 
                else:
                    grouped = [g[metric_col[0]] for _, g in sub.groupby(allocation_col)]
                    group1, group2 = grouped[0], grouped[1]
                    tail = "two-sided"

                normal = self.is_normal(group1) and self.is_normal(group2)
                statistic, p_value = stats.ttest_ind(
                    group1, group2, equal_var=False, alternative=tail
                )
                result = {
                    "test": "welch_t_test",
                    "conducted": True,
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "significant": bool(p_value < alpha),
                }
                if not normal:
                    result["warning"] = (
                        "Normality assumption looks violated (Shapiro-Wilk); "
                        "consider 'mannwhitneyu' instead."
                    )
                return result

            elif test_name == "mannwhitneyu":
                if tail:
                    control_col = list(episode["pairs"].keys())[0]
                    control_value = episode["pairs"][control_col]["control_value"]
                    treatment_col, treatment_value = episode["pairs"][control_col]["treatment"][0]
                    metric_col = episode["metrics"][0]
                    tail = episode["tail"]
            
                    control_mask = df[control_col] == control_value
                    treatment_mask = df[treatment_col] == treatment_value
            
                    sub = df[control_mask | treatment_mask][[control_col, treatment_col, metric_col]].copy()
                    sub["group"] = None
                    sub.loc[control_mask[control_mask | treatment_mask], "group"] = "control"
                    sub.loc[treatment_mask[control_mask | treatment_mask], "group"] = "treatment"
            
                    group1 = sub[sub["group"] == "control"][metric_col]
                    group2 = sub[sub["group"] == "treatment"][metric_col]
            
                    if group1.empty or group2.empty:
                        return "control and treatment groups must both be non-empty for one tailed test" 
                else:
                    grouped = [g[metric_col[0]] for _, g in sub.groupby(allocation_col)]
                    group1, group2 = grouped[0], grouped[1]
                    tail = "two-sided"

                statistic, p_value = stats.mannwhitneyu(group1, group2, alternative=tail)
                return {
                    "test": "mann_whitney_u",
                    "conducted": True,
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "significant": bool(p_value < alpha),
                }

            elif test_name == "anova":
                grouped = [g[metric_col[0]] for _, g in sub.groupby(allocation_col)]
                normal = all(self.is_normal(g) for g in grouped)
                statistic, p_value = stats.f_oneway(*grouped)
                result = {
                    "test": "one_way_anova",
                    "conducted": True,
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "significant": bool(p_value < alpha),
                }
                if not normal:
                    result["warning"] = (
                        "Normality assumption looks violated in at least one group; "
                        "consider 'kruskalwallis' instead."
                    )
                return result

            elif test_name == "kruskalwallis":
                grouped = [g[metric_col[0]] for _, g in sub.groupby(allocation_col)]
                statistic, p_value = stats.kruskal(*grouped)
                return {
                    "test": "kruskal_wallis",
                    "conducted": True,
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "significant": bool(p_value < alpha),
                }

            elif test_name == "chisquare":
                table = pd.crosstab([sub[c] for c in allocation_col], sub[metric_col[0]])
                chi2, p_value, dof, expected = stats.chi2_contingency(table)
                result = {
                    "test": "chi_square",
                    "conducted": True,
                    "statistic": float(chi2),
                    "p_value": float(p_value),
                    "dof": dof,
                    "significant": bool(p_value < alpha),
                }
                if (expected < 5).any():
                    result["warning"] = (
                        "Some expected cell counts are below 5; the chi-square "
                        "approximation may be unreliable. Consider 'fisherexact' "
                        "if this is a 2x2 table."
                    )
                return result
            
            elif test_name == "fisherexact":

                if not tail:
                    table = pd.crosstab(
                        [sub[c] for c in allocation_col],
                        sub[metric_col[0]]
                    )

                    if table.shape != (2, 2):
                        return f"'fisherexact' requires a 2x2 table; got shape {table.shape}."

                    odds_ratio, p_value = stats.fisher_exact(
                        table.values,
                        alternative="two-sided",
                    )

                else:
                    control_col = list(episode["pairs"].keys())[0]
                    control_value = episode["pairs"][control_col]["control_value"]
                    treatment_col, treatment_value = episode["pairs"][control_col]["treatment"][0]
                    metric = episode["metrics"][0]

                    control_mask = df[control_col] == control_value
                    treatment_mask = df[treatment_col] == treatment_value

                    sub = df[control_mask | treatment_mask][
                        [control_col, treatment_col, metric]
                    ].copy()

                    sub["group"] = None
                    sub.loc[control_mask[control_mask | treatment_mask], "group"] = "control"
                    sub.loc[treatment_mask[control_mask | treatment_mask], "group"] = "treatment"

                    table = pd.crosstab(sub["group"], sub[metric])

                    if table.shape != (2, 2):
                        return f"'fisherexact' requires a 2x2 table; got shape {table.shape}."

                    table = table.loc[["control", "treatment"]]

                    odds_ratio, p_value = stats.fisher_exact(
                        table.values,
                        alternative=tail,
                    )

                return {
                    "test": "fisher_exact",
                    "conducted": True,
                    "statistic": float(odds_ratio),
                    "p_value": float(p_value),
                    "significant": bool(p_value < alpha),
                }

            elif test_name == "ztest":

                if tail:
                    control_col = list(episode["pairs"].keys())[0]
                    control_value = episode["pairs"][control_col]["control_value"]
                    treatment_col, treatment_value = episode["pairs"][control_col]["treatment"][0]
                    metric_col = episode["metrics"][0]
                    tail = episode["tail"]

                    control_mask = df[control_col] == control_value
                    treatment_mask = df[treatment_col] == treatment_value

                    sub = df[control_mask | treatment_mask][[control_col, treatment_col, metric_col]].copy()
                    sub["group"] = None
                    sub.loc[control_mask[control_mask | treatment_mask], "group"] = "control"
                    sub.loc[treatment_mask[control_mask | treatment_mask], "group"] = "treatment"

                    group1 = sub[sub["group"] == "control"][metric_col]
                    group2 = sub[sub["group"] == "treatment"][metric_col]

                    if group1.empty or group2.empty:
                        return "control and treatment groups must both be non-empty for one tailed test"

                    categories = sorted(sub[metric_col].unique())
                    if len(categories) != 2:
                        return "ztest requires a binary categorical metric"

                    success_label = categories[0]
                    counts = [
                        int((group1 == success_label).sum()),
                        int((group2 == success_label).sum()),
                    ]
                    nobs = [len(group1), len(group2)]

                else:
                    categories = sorted(sub[metric_col[0]].unique())
                    success_label = categories[0]
                    tail = episode["tail"]

                    groups = list(sub.groupby(allocation_col))
                    counts = [int((g[metric_col[0]] == success_label).sum()) for _, g in groups]
                    nobs = [len(g) for _, g in groups]

                statistic, p_value = proportions_ztest(
                    count=counts,
                    nobs=nobs,
                    alternative=tail,
                )

                return {
                    "test": "z_test_proportions",
                    "conducted": True,
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "significant": bool(p_value < alpha),
                    "success_category": success_label,
                }
    
            elif test_name == "manova":
                result = self.mannova_test(df=sub, allocation_col=allocation_col, metric_col=metric_col, alpha=alpha)
                if isinstance(result, dict):
                    result["conducted"] = True
                return result

            elif test_name == "gtest":
                result = self.multi_categorical_gtest(df=sub, allocation_col=allocation_col, metric_col=metric_col, alpha=alpha)
                if isinstance(result, dict):
                    result["conducted"] = True
                return result

        except Exception as e:
            return f"'{test_name}' raised during execution despite passing structural checks: {e}"