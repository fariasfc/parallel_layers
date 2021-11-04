import re
from time import sleep

import numpy as np
import pandas as pd
import wandb

# import plotly.express as px
# import plotly.figure_factory as ff
from scipy.stats import wilcoxon

pd.set_option("display.max_columns", None)
metric_name = "overall_acc"
# metric_name = "matthews_corrcoef"
# metric = "roc_micro"


def wandb_to_df(project_names, metric="overall_acc"):
    project_runs = {}
    for project_name in project_names:
        print(f"Downloading project {project_name}")
        try_again = True
        while try_again:
            try:
                # Moving api inside loop since I need to `reset` the session when having timeout errors...
                api = wandb.Api()
                project_runs[project_name] = [
                    r for r in api.runs(project_name, {}) if r.state == "finished"
                ]
                try_again = False
            except Exception as e:
                print(f"Trying again {project_name} - {e}")
                sleep(5)

    dicts = []
    for k in project_runs.keys():
        runs = project_runs[k]
        for r in runs:
            metrics = {
                k: r.summary[k]
                for k in r.summary.keys()
                if re.search(
                    # f"(valid|test)_(drl|trained|untrained|raw).*{metric}.*",
                    f"test.*{metric}.*",
                    k,
                )
            }
            d = {
                "project": r.project,
                "dataset_name": r.config["training"]["dataset"],
                # "config": r.config,
                **metrics,
            }
            dicts.append(d)

    df = pd.DataFrame(dicts)
    return df


def wilcoxon_tests(df, metric):
    results = []
    dataset_names = df["dataset_name"].unique()
    df_melted = df.melt(["project", "dataset_name"])
    df_melted["group"] = df_melted["project"] + "-" + df_melted["variable"]
    groups = df_melted["variable"].unique()
    projects = df_melted["project"].unique()
    # groups = df_melted["group"].unique()
    # df_melted = df_melted.drop(["project", "variable"], axis=1)
    df_melted = df_melted.dropna()
    # groups = df_melted["group"].unique()
    for dataset_name in dataset_names:
        df_current_dataset = df_melted[df_melted["dataset_name"] == dataset_name]
        for project in projects:
            df_current_dataset_project = df_current_dataset[
                df_current_dataset["project"] == project
            ]
            for g1 in groups:
                for g2 in groups:
                    if re.match(f".*_autoconstructive_.*{metric}.*", g1) and re.match(
                        r".*[knn\-1|knn\-3|svm|xgboost|rf|drl].*", g2
                    ):
                        if g1 != g2:
                            df1 = df_current_dataset_project[
                                df_current_dataset_project["variable"] == g1
                            ]
                            df2 = df_current_dataset_project[
                                df_current_dataset_project["variable"] == g2
                            ]
                            if df1.shape[0] < 2 or df2.shape[0] < 2:
                                continue
                            g1_over_g2 = "d"
                            stat = 999
                            p = 999

                            values1 = df1["value"].to_numpy()
                            values2 = df2["value"].to_numpy()

                            if any(values1 != values2):
                                stat, p = wilcoxon(values1, values2)
                            if p < 0.05:
                                if values1.mean() > values2.mean():
                                    g1_over_g2 = "w"
                                else:
                                    g1_over_g2 = "l"
                            results.append(
                                {
                                    "project": project,
                                    "dataset_name": dataset_name,
                                    "g1_count": len(values1),
                                    "g1": g1,
                                    "g1_mean": values1.mean(),
                                    "g1_std": values1.std(),
                                    "g2_count": len(values2),
                                    "g2": g2,
                                    "g2_mean": values2.mean(),
                                    "g2_std": values2.std(),
                                    "wilcoxon_result": g1_over_g2,
                                    "statistic": stat,
                                    "p-value": p,
                                }
                            )
    df = pd.DataFrame(
        results,
        columns=[
            "project",
            "dataset_name",
            "g1_count",
            "g1",
            "g1_mean",
            "g1_std",
            "g2_count",
            "g2",
            "g2_mean",
            "g2_std",
            "wilcoxon_result",
            "statistic",
            "p-value",
        ],
    )
    df = df.sort_values(["g1", "dataset_name", "wilcoxon_result"])
    return df


def wandb_to_csv(proj_name, save_file_path):
    df = wandb_to_df([proj_name], metric_name)
    df.to_csv(save_file_path, index=False)


def plot_html(df, filename="analysis.html"):
    def highlight_max(x, props=""):
        return np.where(x.to_numpy() > 0.1, props, None)

    with open(filename, "w") as html_file:
        html_file.write(df.style.apply(highlight_max, props=f"colors: blue;").render())

        # html_file.write(
        #     df.style.highlight_max(
        #         color="lightgreen", axis=1, subset=["g1_mean", "g2_mean"]
        #     ).render()
        # )


if __name__ == "__main__":
    df = wandb_to_df(
        [
            # "exp_14_stacking",
            # "exp_14_stacking_rtol0.01",
            # "exp_11",
            # "exp_11_boosting",
            # # "exp_12_rtol",
            # "exp_13_rol_boosting",
            # "exp_004_clean"
            # "exp_011_rtol0.01",
            # "exp_011_rtol0.001",
            # "exp_011_autoencoders",
            # "exp_011_autoencoders_50",
            # "exp_012_rtol0.01",
            # "exp_013_12foiautoencoders_rtol0.01",
            # "exp_014_rtol_defato_0.01",
            "exp0007",
            "exp0009_stack_hidden_maxlayers2_noappend",
            "exp0009_maxlayers1",
            "exp0009_maxlayers2",
            "exp0009_stack_hidden_maxlayers2",
            "exp0016",
            "exp0016_tanh",
            "exp0016_relu",
            "exp0016_max_layers1_tanh",
            "exp0016_max_layers1_relu",
        ],
        metric_name,
    )
    df.to_csv('/tmp/csv.csv')
    # df = pd.read_csv('/tmp/csv.csv')
    # df = df.drop(columns=["Unnamed: 0"])
    df = df.sort_index(axis=1)
    metric_columns = [
        c
        for c in df.columns
        if re.match(r"test_.*[1\-nn|3\-nn|svm|xgboost|rf|autoconstructive]_*" + metric_name, c)
        # if re.match("test_.*[drl]_" + metric, c)
    ]

    df_filtered_tmp = df[metric_columns + ["project", "dataset_name"]]
    df_filtered = pd.DataFrame()
    for project in df_filtered_tmp['project'].unique():
        df_p = df[df['project'] == project]
        for dataset_name in df_p['dataset_name'].unique():
            df_p_d = df_p[df_p['dataset_name'] == dataset_name]
            tmp_df = df_p_d.iloc[:30, :]

            df_filtered = pd.concat((df_filtered, tmp_df))

    df = wilcoxon_tests(df_filtered, metric_name)
    df.to_csv("analysis_wilcoxon.csv")

    # df.groupby(['project', 'g1', 'g2', 'wilcoxon_result']).count().unstack(2).to_csv("final.csv")
    df.groupby(['project', 'wilcoxon_result'])['wilcoxon_result'].count().unstack(1).to_csv("final.csv")

    # df_pivot = df[df["g1"] == f"test_drl_untrained_{metric_name}"].pivot(
    #     index=["project", "dataset_name"],
    #     columns="g2",
    #     values=["g1_mean", "g2_mean", "wilcoxon_result"],
    # )
    # df_pivot.columns = df_pivot.columns.swaplevel(0, 1)
    # df_pivot.sort_index(1).to_csv("pivot_untrained.csv")

    df_pivot = df[df["g1"] == f"test_autoconstructive_{metric_name}"].pivot(
        index=["project", "dataset_name"],
        columns="g2",
        values=["g1_mean", "g2_mean", "wilcoxon_result"],
    )
    df_pivot.columns = df_pivot.columns.swaplevel(0, 1)
    df_pivot.sort_index(1).to_csv("pivot_trained.csv")

    # plot_html(df)
    # df[["project", "dataset_name", "g2", "wilcoxon_result"]].pivot(
    #     "project", "dataset_name", "g2", "wilcoxon_result"
    # ).to_csv("pivot.csv")
    df_filtered.melt(["project", "dataset_name"]).groupby(
        ["project", "dataset_name", "variable"]
    ).mean().unstack([0, 2]).to_csv("analysis2.csv")
    avg = df.groupby(["project", "dataset_name"]).mean()
    avg.to_csv("analysis.csv")
    with open("analysis.html", "w") as html_file:
        html_file.write(avg.style.highlight_max(color="lightgreen", axis=1).render())
    print(df)
