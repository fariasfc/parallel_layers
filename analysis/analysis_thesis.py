from sklearn.linear_model import LinearRegression
from scipy.stats import wilcoxon, ttest_rel, ttest_ind


import pandas as pd
from plotly.subplots import make_subplots

import plotly.graph_objects as go


import plotly.express as px
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pymcdm

use_mcdm = True
metric = "overall_acc"
train_metric = f"train_{metric}"
validation_metric = f"validation_{metric}"
holdout_metric = f"holdout_{metric}"
test_metric = f"test_{metric}"
# metric = "matthews_corrcoef"
# folder that contains the nosbss.csv
folder = Path("experiments/exp0090/")
policies_folder = folder / f"{metric}/policies"
folder.absolute()
df_path = folder / "politica_1_oracle_1m1l_and_nosbss.csv"
df_parquet_path = folder / "politica_1_oracle_1m1l_and_nosbss.parquet"
datasets = [
    "ionosphere",
    "diabetes",
    "Australian",
    "car(3)",
    "credit-g",
    "climate-model-simulation-crashes(4)",
    "ilpd",
    "balance-scale",
    "libras_move",
    # "hill-valley",
    "blood-transfusion-service-center",
    "lsvt",
    "wdbc",
    "satimage",
    "vowel(2)",
    # "musk",
]
experiments = [
    "exp0090_politica_1_oracle_1m1l",
    "exp0090_politica_1_oracle_1m1l_nosbss",
]

policies = [
    "oracle",
    "holdout",
    "Train",
    "Validation",
    # "pareto_second_best_holdout_num_neurons",
    # "0.01_smallest_mean_dist_holdout",
    # "smallest_euclidian_train_holdout_test_utopic",
    # "smallest_euclidian_holdout_test_utopic",
    # "best_holdout_in_best_architecture_10fold",
    "Best Architectural Holdout",
    "Best Architectural Validation",
    "Best Architectural Holdout Over All Runs",
    "Best Architectural Validation Over All Runs",
    "Utopic Euclidean Train Validation Holdout Test",
    "Utopic Euclidean Train Validation Holdout",
    "topsis_pareto_oracle",
    "topsis_pareto_holdout",
]

if use_mcdm:
    policies += [
        "topsis_pareto_oracle",
        "topsis_pareto_holdout",
        "topsis_pareto_validation",
        "topsis_pareto_train",
        "topsis_pareto_holdout_no_epochs",
        "topsis_pareto_holdout_no_num_neurons",
        "topsis_pareto_holdout_weighted_neu-p3_epc-p1_hold-p6",
        "topsis_pareto_holdout_weighted_neu-p1_epc-p1_hold-p8",
        "topsis_pareto_holdout_mean_diffs",
        "all_except_test",
        "topsis_all_overall_acc_except_test",
        "topsis_train_validation_holdout_no_num_neurons_no_epochs",
        "topsis_pareto_train_validation_holdout",
        "topsis_pareto_train_validation_holdout_no_epochs",
        "topsis_best_rank_architecture_pareto_train_validation_holdout_no_epochs",
        "topsis_pareto_oracle_holdout_no_epochs_no_num_neurons",
        "topsis_pareto_oracle_holdout_train",
        "topsis_pareto_oracle_holdout_train_noepochs_nonum_neurons",
        "topsis_pareto_oracle_holdout_validation_train",
        "topsis_pareto_oracle_holdout_validation_train_noepochs_nonum_neurons",
        "topsis_holdout_in_best_median_architecture_10fold_nopareto",
        "mairca_pareto_holdout",
        "mairca_pareto_holdout_reverse",
        "moora_pareto_holdout",
        "moora_pareto_holdout_inverse",
        "wilcoxon_holdout",
        "ttest_ind_holdout",
        "ttest_rel_holdout",
    ]
    analysis_folder = folder / f"{metric}/plots"
else:
    analysis_folder = folder / f"{metric}/plots_no_topsis"

analysis_folder.mkdir(parents=True, exist_ok=True)
policies_folder.mkdir(parents=True, exist_ok=True)


def csv_to_parquet():
    df = pd.read_csv(df_path)
    df.to_parquet(df_parquet_path)


def load_df():
    df = pd.read_parquet(df_parquet_path)
    # Removing repeated experiments
    non_duplicated_index = (
        df[["experiment", "dataset", "run", "model_id"]].drop_duplicates()
    ).index
    df = df.loc[non_duplicated_index].reset_index()
    df["mean_diffs"] = (
        abs(df[holdout_metric] - df[train_metric])
        + abs(df[holdout_metric] - df[validation_metric])
        + abs(df[train_metric] - df[validation_metric])
    ) / 3
    print(df.keys())
    df = df[(df["experiment"].isin(experiments)) & (df["dataset"].isin(datasets))]
    print(f"df.shape: {df.shape}")

    return df


def load_df_policies(only_sbss=True):
    experiments = [
        "exp0090_politica_1_oracle_1m1l",
    ]
    if not only_sbss:
        experiments.append("exp0090_politica_1_oracle_1m1l_nosbss")

    policies_files = policies_folder.glob("**/*.csv")
    print("prepolicy")
    df_policies = pd.concat(
        [
            pd.read_csv(policy_file_path)
            for policy_file_path in policies_files
            if policy_file_path.name.replace(".csv", "") in policies
        ]
    )
    print("postpolicy")

    df_policies = df_policies[
        (df_policies["experiment"].isin(experiments))
        & (df_policies["dataset"].isin(datasets))
    ]

    print(f"df_policies.shape: {df_policies.shape}")
    print(f"policies: {df_policies['policy'].unique()}")
    return df_policies


def load_df_times():
    df_paths = list(Path("experiments").glob("*times*/*.csv"))
    print(df_paths)
    dfs = []
    for df_path in df_paths:
        if "bs32" in str(df_path.absolute()):
            batch = 32
        if "bs128" in str(df_path.absolute()):
            batch = 128
        if "bs256" in str(df_path.absolute()):
            batch = 256
        if "cpu" in str(df_path.absolute()):
            device = "cpu"
        else:
            device = "cuda"
        # print(f"opening {df_path} with batch {batch} device {device}")
        tmp_df = pd.read_csv(df_path)
        tmp_df["parallel/sequential"] *= 100
        tmp_df = tmp_df.drop(
            columns=[
                "min_neurons",
                "max_neurons",
                "epochs",
                "repetitions",
                "activation_functions",
                "repetitions",
                "num_models",
            ]
        )
        tmp_df["batch"] = batch
        tmp_df["device"] = device
        dfs.append(tmp_df)
    # for df_device_key, df in dfs.items():
    # #,num_samples,num_features,min_neurons,max_neurons,epochs,num_models,activation_functions,repetitions,sequential,parallel,device,parallel/sequential
    #     df = df.drop(columns=["min_neurons", "max_neurons", "epochs", "repetitions", "activation_functions", "repetitions"])
    #     df = df.melt(id_vars=["num_samples", "num_features", "num_models", "device"])
    # df_times_cpu = pd.read_csv(Path("experiments/times_cuda/times_cpu.csv"))
    df = pd.concat(dfs)

    df = df.drop(columns=["Unnamed: 0"])
    # print(df.columns)
    df.to_csv(analysis_folder / "all_times.csv")

    return df


df_times = load_df_times()
df_times = df_times.melt(["num_features", "num_samples", "batch", "device"])
df_times_csv = (
    df_times[["num_features", "num_samples", "variable", "value", "batch", "device"]]
    .pivot_table(index=["device", "variable", "num_features", "num_samples", "batch"])
    .unstack(["num_samples", "batch"])
    .reindex([100, 1000, 10000], axis=1, level=1)
    .reindex([32, 128, 256], axis=1, level=2)
    .reindex(["parallel", "sequential", "parallel/sequential"], level=1)
).round(3)
df_times_csv.to_csv(analysis_folder / "df_times_paper.csv")
# df_times
# df_times = plot_times(df_times)
# df_times.unstack("strategy")

# csv_to_parquet()


def post_process_dfs(df):
    df["mean_metrics"] = df[
        [train_metric, validation_metric, holdout_metric, test_metric]
    ].mean(axis=1)
    df["std_metrics"] = df[
        [train_metric, validation_metric, holdout_metric, test_metric]
    ].std(axis=1)
    return df


def plot_holdout_test(df):
    fig = px.scatter(
        df.sort_values(by=[test_metric, holdout_metric]),
        x=holdout_metric,
        y=test_metric,
        color="dataset",
        symbol="experiment",
    )

    fig.write_html(analysis_folder / "holdout__test.html")


def debug_ilpd(df):
    df = df.copy()
    df[df["dataset"] == "ilpd"].to_csv(analysis_folder / "debug_ilpd.csv")


def num_models(df):
    # should be 	1,612,800
    df.groupby(["experiment", "dataset", "run"]).size().to_csv(
        analysis_folder / "num_models.csv"
    )


def is_pareto_efficient(costs, return_mask=True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array of costs to be minimized
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


def get_ranked_pmlps_df(
    pmlps_df,
    mcdm_tuples,
    theoretical_best=None,
    theoretical_worst=None,
    only_pareto_solutions=True,
    sort_by_rank=False,
    weights=None,
    mcdm_method_name="topsis",
):

    pmlps_df = pmlps_df.copy()

    mcdm_keys = [k[0] for k in mcdm_tuples]

    types = np.array([k[1] for k in mcdm_tuples])
    mcdm_method_map = {
        "topsis": pymcdm.methods.TOPSIS(pymcdm.normalizations.minmax_normalization),
        "moora": pymcdm.methods.MOORA(),
        "mairca": pymcdm.methods.MAIRCA(),
    }
    mcdm_method = mcdm_method_map[mcdm_method_name]

    decision_matrix = pmlps_df[mcdm_keys].to_numpy()
    pareto_matrix = decision_matrix.copy()

    for c in range(pareto_matrix.shape[1]):
        # types inform if is profit (1 - should be maximized) or cost (-1 - should be minimized)
        # is_pareto_efficient expectes a matrix of costs to minimize
        # that's why i am multipying by -types[c]
        pareto_matrix[:, c] *= -types[c]

    pareto_mask = is_pareto_efficient(pareto_matrix)
    pmlps_df["dominant_solution"] = pareto_mask

    if theoretical_best is not None:
        if theoretical_worst is None:
            raise ValueError(
                "Both theoretical_best and theoretical_worst must be None or have values."
            )
        decision_matrix = np.vstack(
            (decision_matrix, theoretical_best, theoretical_worst)
        )

    if weights is None:
        weights = pymcdm.weights.equal_weights(decision_matrix)
    # # weights = np.array([0.5, 0.4, 0.1])
    ranks = mcdm_method(decision_matrix, weights, types)
    # removing best_and_worst_theoretical_mlps
    if theoretical_best is not None:
        ranks = ranks[:-2]

    pmlps_df["rank"] = ranks

    if only_pareto_solutions:
        pmlps_df = pmlps_df.loc[pareto_mask]

    if sort_by_rank:
        pmlps_df = pmlps_df.sort_values(by=["rank"], ascending=False).reset_index()

    return pmlps_df


def apply_policies(df):
    # datasets = df["dataset"].unique()
    # datasets = ["diabetes", "credit-g"]
    runs = df["run"].unique()

    for policy in policies:
        policy_file_path = policies_folder / f"{policy}.csv"
        if policy_file_path.exists():
            print(f"{policy_file_path.absolute()} exists.")
            continue
        print(f"Executing {policy_file_path.absolute()}.")
        choices = []
        for experiment in experiments:
            pmlps_df_experiment = df[df["experiment"] == experiment]
            for dataset in tqdm(datasets):
                pmlps_df_dataset = pmlps_df_experiment[
                    pmlps_df_experiment["dataset"] == dataset
                ]
                # Global == all runs in the dataset
                df_avg_by_architecture_global = (
                    pmlps_df_dataset.groupby(["num_neurons", "activation_name"])
                    .mean()
                    .reset_index()
                )
                best_holdout_architecture_global = (
                    df_avg_by_architecture_global.sort_values(
                        by=[holdout_metric, "num_neurons"], ascending=[False, True]
                    ).iloc[0]
                )
                best_validation_architecture_global = (
                    df_avg_by_architecture_global.sort_values(
                        by=[validation_metric, "num_neurons"], ascending=[False, True]
                    ).iloc[0]
                )
                for run in runs:
                    df_avg_by_architecture_run = (
                        pmlps_df_dataset.groupby(["num_neurons", "activation_name"])
                        .mean()
                        .reset_index()
                    )
                    best_holdout_architecture_run = (
                        df_avg_by_architecture_run.sort_values(
                            by=[holdout_metric, "num_neurons"], ascending=[False, True]
                        ).iloc[0]
                    )
                    best_validation_architecture_run = (
                        df_avg_by_architecture_run.sort_values(
                            by=[validation_metric, "num_neurons"],
                            ascending=[False, True],
                        ).iloc[0]
                    )
                    pmlps_df = pmlps_df_dataset[
                        (pmlps_df_dataset["run"] == run)
                        & (pmlps_df_dataset["experiment"] == experiment)
                    ]
                    pmlps_df = pmlps_df.sort_values(
                        by=["mean_diffs", "num_neurons", holdout_metric],
                        ascending=[True, True, False],
                    )
                    mcdm_tuples = [
                        ("num_neurons", -1),
                        ("epoch", 1),
                        (holdout_metric, 1),
                    ]
                    ranked_pmlps_df_original = get_ranked_pmlps_df(
                        pmlps_df, mcdm_tuples, only_pareto_solutions=False
                    )
                    ranked_pmlps_df = ranked_pmlps_df_original.copy()

                    if policy == "oracle":
                        ranked_pmlps_df = ranked_pmlps_df.sort_values(
                            by=[test_metric, "num_neurons"],
                            ascending=[False, True],
                        )
                    elif policy == "holdout":
                        ranked_pmlps_df = ranked_pmlps_df.sort_values(
                            by=[holdout_metric, "num_neurons"],
                            ascending=[False, True],
                        )
                    elif policy == "Train":
                        ranked_pmlps_df = ranked_pmlps_df.sort_values(
                            by=[train_metric, "num_neurons"],
                            ascending=[False, True],
                        )
                    elif policy == "Validation":
                        ranked_pmlps_df = ranked_pmlps_df.sort_values(
                            by=[validation_metric, "num_neurons"],
                            ascending=[False, True],
                        )
                    elif policy == "Best Architectural Validation":
                        activation = best_validation_architecture_run["activation_name"]
                        num_neurons = best_validation_architecture_run["num_neurons"]
                        ranked_pmlps_df = ranked_pmlps_df[
                            (ranked_pmlps_df["activation_name"] == activation)
                            & (ranked_pmlps_df["num_neurons"] == num_neurons)
                        ].sort_values(
                            by=[validation_metric, "num_neurons"],
                            ascending=[False, True],
                        )
                    elif policy == "Best Architectural Holdout":
                        activation = best_holdout_architecture_run["activation_name"]
                        num_neurons = best_holdout_architecture_run["num_neurons"]
                        ranked_pmlps_df = ranked_pmlps_df[
                            (ranked_pmlps_df["activation_name"] == activation)
                            & (ranked_pmlps_df["num_neurons"] == num_neurons)
                        ].sort_values(
                            by=[holdout_metric, "num_neurons"], ascending=[False, True]
                        )
                    elif policy == "Best Architectural Holdout Over All Runs":
                        activation = best_holdout_architecture_global["activation_name"]
                        num_neurons = best_holdout_architecture_global["num_neurons"]
                        ranked_pmlps_df = ranked_pmlps_df[
                            (ranked_pmlps_df["activation_name"] == activation)
                            & (ranked_pmlps_df["num_neurons"] == num_neurons)
                        ].sort_values(
                            by=[holdout_metric, "num_neurons"], ascending=[False, True]
                        )
                    elif policy == "Best Architectural Validation Over All Runs":
                        activation = best_validation_architecture_global[
                            "activation_name"
                        ]
                        num_neurons = best_validation_architecture_global["num_neurons"]
                        ranked_pmlps_df = ranked_pmlps_df[
                            (ranked_pmlps_df["activation_name"] == activation)
                            & (ranked_pmlps_df["num_neurons"] == num_neurons)
                        ].sort_values(
                            by=[validation_metric, "num_neurons"],
                            ascending=[False, True],
                        )

                    elif policy == "pareto_second_best_holdout_num_neurons":
                        mcdm_tuples = [
                            ("num_neurons", -1),
                            (holdout_metric, 1),
                        ]
                        ranked_pmlps_df = get_ranked_pmlps_df(
                            pmlps_df,
                            mcdm_tuples,
                            only_pareto_solutions=True,
                            sort_by_rank=True,
                        )
                        ranked_pmlps_df = ranked_pmlps_df.drop(
                            columns=["level_0"]
                        ).sort_values(
                            by=[holdout_metric, "num_neurons"],
                            ascending=[False, True],
                        )
                        if ranked_pmlps_df.shape[0] > 1:
                            ranked_pmlps_df = (
                                ranked_pmlps_df.iloc[1:].copy().reset_index()
                            )
                    elif policy == "best_holdout_in_best_architecture_10fold":
                        best_architecture = (
                            ranked_pmlps_df.groupby(["architecture_id"])
                            .mean()
                            .sort_values(by=[holdout_metric], ascending=[False])
                            .reset_index()
                            .iloc[0]["architecture_id"]
                        )
                        ranked_pmlps_df = ranked_pmlps_df[
                            ranked_pmlps_df["architecture_id"] == best_architecture
                        ]
                        ranked_pmlps_df = ranked_pmlps_df.sort_values(
                            by=[holdout_metric, "num_neurons"],
                            ascending=[False, True],
                        )
                    elif (
                        policy
                        == "topsis_holdout_in_best_median_architecture_10fold_nopareto"
                    ):
                        mcdm_tuples = [
                            ("num_neurons", -1),
                            ("epoch", 1),
                            (holdout_metric, 1),
                        ]
                        ranked_pmlps_df = get_ranked_pmlps_df(
                            pmlps_df,
                            mcdm_tuples,
                            # only_pareto_solutions=True,
                            sort_by_rank=True,
                        )
                        best_architecture = (
                            ranked_pmlps_df.groupby(["architecture_id"])
                            .median()
                            .sort_values(by=["rank"], ascending=[False])
                            .reset_index()
                            .iloc[0]["architecture_id"]
                        )
                        ranked_pmlps_df = ranked_pmlps_df[
                            ranked_pmlps_df["architecture_id"] == best_architecture
                        ]
                        ranked_pmlps_df = ranked_pmlps_df.sort_values(
                            by=[holdout_metric, "num_neurons"],
                            ascending=[False, True],
                        )

                    elif policy == "0.01_smallest_mean_dist_holdout":
                        ranked_pmlps_df = ranked_pmlps_df.iloc[
                            : int(ranked_pmlps_df.shape[0] * 0.01)
                        ]
                        ranked_pmlps_df = ranked_pmlps_df.sort_values(
                            by=[holdout_metric, "num_neurons"],
                            ascending=[False, True],
                        )
                    elif policy == "topsis_pareto_oracle":
                        mcdm_tuples = [
                            ("num_neurons", -1),
                            ("epoch", 1),
                            (test_metric, 1),
                        ]
                        ranked_pmlps_df = get_ranked_pmlps_df(
                            pmlps_df,
                            mcdm_tuples,
                            only_pareto_solutions=True,
                            sort_by_rank=True,
                        )
                    elif (
                        policy
                        == "topsis_train_validation_holdout_no_num_neurons_no_epochs"
                    ):
                        mcdm_tuples = [
                            (train_metric, 1),
                            (validation_metric, 1),
                            (holdout_metric, 1),
                        ]
                        ranked_pmlps_df = get_ranked_pmlps_df(
                            pmlps_df,
                            mcdm_tuples,
                            only_pareto_solutions=True,
                            sort_by_rank=True,
                        )
                    elif policy == "topsis_pareto_train_validation_holdout":
                        mcdm_tuples = [
                            ("num_neurons", -1),
                            ("epoch", 1),
                            (train_metric, 1),
                            (validation_metric, 1),
                            (holdout_metric, 1),
                        ]
                        ranked_pmlps_df = get_ranked_pmlps_df(
                            pmlps_df,
                            mcdm_tuples,
                            only_pareto_solutions=True,
                            sort_by_rank=True,
                        )
                    elif policy == "topsis_pareto_train_validation_holdout_no_epochs":
                        mcdm_tuples = [
                            ("num_neurons", -1),
                            (train_metric, 1),
                            (validation_metric, 1),
                            (holdout_metric, 1),
                        ]
                        ranked_pmlps_df = get_ranked_pmlps_df(
                            pmlps_df,
                            mcdm_tuples,
                            only_pareto_solutions=True,
                            sort_by_rank=True,
                        )
                    elif (
                        policy
                        == "topsis_best_rank_architecture_pareto_train_validation_holdout_no_epochs"
                    ):
                        mcdm_tuples = [
                            ("num_neurons", -1),
                            (train_metric, 1),
                            (validation_metric, 1),
                            (holdout_metric, 1),
                        ]
                        ranked_pmlps_df = get_ranked_pmlps_df(
                            pmlps_df,
                            mcdm_tuples,
                            only_pareto_solutions=False,
                            sort_by_rank=True,
                        )
                        best_architecture_id = (
                            ranked_pmlps_df.groupby(["rank"])
                            .median()
                            .sort_values(by=["rank"], ascending=[False])
                            .iloc[0]["architecture_id"]
                        )
                        ranked_pmlps_df = ranked_pmlps_df[
                            ranked_pmlps_df["architecture_id"] == best_architecture_id
                        ].sort_values(by=["rank"], ascending=[False])
                    elif (
                        policy
                        == "topsis_pareto_oracle_holdout_no_epochs_no_num_neurons"
                    ):
                        mcdm_tuples = [
                            (holdout_metric, 1),
                            (test_metric, 1),
                        ]
                        ranked_pmlps_df = get_ranked_pmlps_df(
                            pmlps_df,
                            mcdm_tuples,
                            only_pareto_solutions=True,
                            sort_by_rank=True,
                        )
                    elif policy == "topsis_pareto_oracle_holdout_train":
                        mcdm_tuples = [
                            ("num_neurons", -1),
                            ("epoch", 1),
                            (train_metric, 1),
                            (holdout_metric, 1),
                            (test_metric, 1),
                        ]
                        ranked_pmlps_df = get_ranked_pmlps_df(
                            pmlps_df,
                            mcdm_tuples,
                            only_pareto_solutions=True,
                            sort_by_rank=True,
                        )
                    elif policy == "topsis_pareto_oracle_holdout_validation_train":
                        mcdm_tuples = [
                            ("num_neurons", -1),
                            ("epoch", 1),
                            (train_metric, 1),
                            (validation_metric, 1),
                            (holdout_metric, 1),
                            (test_metric, 1),
                        ]
                        ranked_pmlps_df = get_ranked_pmlps_df(
                            pmlps_df,
                            mcdm_tuples,
                            only_pareto_solutions=True,
                            sort_by_rank=True,
                        )
                    elif (
                        policy
                        == "topsis_pareto_oracle_holdout_validation_train_noepochs_nonum_neurons"
                    ):
                        mcdm_tuples = [
                            (train_metric, 1),
                            (validation_metric, 1),
                            (holdout_metric, 1),
                            (test_metric, 1),
                        ]
                        ranked_pmlps_df = get_ranked_pmlps_df(
                            pmlps_df,
                            mcdm_tuples,
                            only_pareto_solutions=True,
                            sort_by_rank=True,
                        )
                    elif (
                        policy
                        == "topsis_pareto_oracle_holdout_train_noepochs_nonum_neurons"
                    ):
                        mcdm_tuples = [
                            (train_metric, 1),
                            (holdout_metric, 1),
                            (test_metric, 1),
                        ]
                        ranked_pmlps_df = get_ranked_pmlps_df(
                            pmlps_df,
                            mcdm_tuples,
                            only_pareto_solutions=True,
                            sort_by_rank=True,
                        )
                    elif policy == "topsis_pareto_holdout":
                        mcdm_tuples = [
                            ("num_neurons", -1),
                            ("epoch", 1),
                            (holdout_metric, 1),
                        ]
                        ranked_pmlps_df = get_ranked_pmlps_df(
                            pmlps_df,
                            mcdm_tuples,
                            only_pareto_solutions=True,
                            sort_by_rank=True,
                        )
                    elif policy == "topsis_pareto_validation":
                        mcdm_tuples = [
                            ("num_neurons", -1),
                            ("epoch", 1),
                            (validation_metric, 1),
                        ]
                        ranked_pmlps_df = get_ranked_pmlps_df(
                            pmlps_df,
                            mcdm_tuples,
                            only_pareto_solutions=True,
                            sort_by_rank=True,
                        )
                    elif policy == "topsis_pareto_train":
                        mcdm_tuples = [
                            ("num_neurons", -1),
                            ("epoch", 1),
                            (train_metric, 1),
                        ]
                        ranked_pmlps_df = get_ranked_pmlps_df(
                            pmlps_df,
                            mcdm_tuples,
                            only_pareto_solutions=True,
                            sort_by_rank=True,
                        )
                    elif policy == "topsis_pareto_holdout_no_epochs_no_num_neurons":
                        mcdm_tuples = [
                            ("num_neurons", -1),
                            (holdout_metric, 1),
                        ]
                        ranked_pmlps_df = get_ranked_pmlps_df(
                            pmlps_df,
                            mcdm_tuples,
                            only_pareto_solutions=True,
                            sort_by_rank=True,
                        )
                    elif policy == "mairca_pareto_holdout":
                        mcdm_tuples = [
                            ("num_neurons", -1),
                            ("epoch", 1),
                            (holdout_metric, 1),
                        ]
                        ranked_pmlps_df = get_ranked_pmlps_df(
                            pmlps_df,
                            mcdm_tuples,
                            only_pareto_solutions=True,
                            sort_by_rank=True,
                            mcdm_method_name="mairca",
                        )
                    elif policy == "mairca_pareto_holdout_reverse":
                        mcdm_tuples = [
                            ("num_neurons", -1),
                            ("epoch", 1),
                            (holdout_metric, 1),
                        ]
                        ranked_pmlps_df = get_ranked_pmlps_df(
                            pmlps_df,
                            mcdm_tuples,
                            only_pareto_solutions=True,
                            sort_by_rank=True,
                            mcdm_method_name="mairca",
                        )
                        ranked_pmlps_df = ranked_pmlps_df.sort_values(
                            by=["rank"], ascending=True
                        )
                    elif policy == "moora_pareto_holdout":
                        mcdm_tuples = [
                            ("num_neurons", -1),
                            ("epoch", 1),
                            (holdout_metric, 1),
                        ]
                        ranked_pmlps_df = get_ranked_pmlps_df(
                            pmlps_df,
                            mcdm_tuples,
                            only_pareto_solutions=True,
                            sort_by_rank=True,
                            mcdm_method_name="moora",
                        )
                    elif policy == "moora_pareto_holdout_inverse":
                        mcdm_tuples = [
                            ("num_neurons", -1),
                            ("epoch", 1),
                            (holdout_metric, 1),
                        ]
                        ranked_pmlps_df = get_ranked_pmlps_df(
                            pmlps_df,
                            mcdm_tuples,
                            only_pareto_solutions=True,
                            sort_by_rank=True,
                            mcdm_method_name="moora",
                        )
                        ranked_pmlps_df = ranked_pmlps_df.sort_values(
                            by=["rank"], ascending=True
                        )
                    elif policy == "topsis_pareto_holdout_no_epochs":
                        mcdm_tuples = [
                            ("num_neurons", -1),
                            (holdout_metric, 1),
                        ]
                        ranked_pmlps_df = get_ranked_pmlps_df(
                            pmlps_df,
                            mcdm_tuples,
                            only_pareto_solutions=True,
                            sort_by_rank=True,
                        )
                    elif policy == "topsis_pareto_holdout_no_num_neurons":
                        mcdm_tuples = [
                            ("epoch", 1),
                            (holdout_metric, 1),
                        ]
                        ranked_pmlps_df = get_ranked_pmlps_df(
                            pmlps_df,
                            mcdm_tuples,
                            only_pareto_solutions=True,
                            sort_by_rank=True,
                        )
                    elif (
                        policy == "topsis_pareto_holdout_weighted_neu-p3_epc-p1_hold-p6"
                    ):
                        mcdm_tuples = [
                            ("num_neurons", -1),
                            ("epoch", 1),
                            (holdout_metric, 1),
                        ]
                        ranked_pmlps_df = get_ranked_pmlps_df(
                            pmlps_df,
                            mcdm_tuples,
                            only_pareto_solutions=True,
                            sort_by_rank=True,
                            weights=np.array([0.3, 0.1, 0.6])
                            # # weights = np.array([0.5, 0.4, 0.1])
                        )
                    elif (
                        policy == "topsis_pareto_holdout_weighted_neu-p1_epc-p1_hold-p8"
                    ):
                        mcdm_tuples = [
                            ("num_neurons", -1),
                            ("epoch", 1),
                            (holdout_metric, 1),
                        ]
                        ranked_pmlps_df = get_ranked_pmlps_df(
                            pmlps_df,
                            mcdm_tuples,
                            only_pareto_solutions=True,
                            sort_by_rank=True,
                            weights=np.array([0.1, 0.1, 0.8]),
                        )
                    elif policy == "topsis_pareto_holdout_mean_diffs":
                        mcdm_tuples = [
                            ("num_neurons", -1),
                            ("mean_diffs", -1),
                            ("epoch", 1),
                            (holdout_metric, 1),
                        ]
                        ranked_pmlps_df = get_ranked_pmlps_df(
                            pmlps_df,
                            mcdm_tuples,
                            only_pareto_solutions=True,
                            sort_by_rank=True,
                        )
                    elif policy == "smallest_euclidian_holdout_test_utopic":
                        overall_accs = ranked_pmlps_df[
                            [test_metric, holdout_metric]
                        ].values
                        utopic_accs = np.array([[1.0, 1.0]])
                        ranked_pmlps_df.loc[:, "euclidian_to_utopic"] = np.sqrt(
                            ((overall_accs - utopic_accs) ** 2).sum(1)
                        )
                        ranked_pmlps_df = ranked_pmlps_df.sort_values(
                            by=["euclidian_to_utopic", "num_neurons"],
                            ascending=[True, True],
                        )
                        ranked_pmlps_df = ranked_pmlps_df.drop(
                            columns=["euclidian_to_utopic"]
                        )
                    elif policy == "smallest_euclidian_train_holdout_test_utopic":
                        overall_accs = ranked_pmlps_df[
                            [train_metric, test_metric, holdout_metric]
                        ].values
                        utopic_accs = np.array([[1.0, 1.0, 1.0]])
                        ranked_pmlps_df.loc[:, "euclidian_to_utopic"] = np.sqrt(
                            ((overall_accs - utopic_accs) ** 2).sum(1)
                        )
                        ranked_pmlps_df = ranked_pmlps_df.sort_values(
                            by=["euclidian_to_utopic", "num_neurons"],
                            ascending=[True, True],
                        )
                        ranked_pmlps_df = ranked_pmlps_df.drop(
                            columns=["euclidian_to_utopic"]
                        )
                    elif policy == "Utopic Euclidean Train Validation Holdout Test":
                        overall_accs = ranked_pmlps_df[
                            [
                                train_metric,
                                validation_metric,
                                test_metric,
                                holdout_metric,
                            ]
                        ].values
                        utopic_accs = np.array([[1.0, 1.0, 1.0, 1.0]])
                        ranked_pmlps_df.loc[:, "euclidian_to_utopic"] = np.sqrt(
                            ((overall_accs - utopic_accs) ** 2).sum(1)
                        )
                        ranked_pmlps_df = ranked_pmlps_df.sort_values(
                            by=["euclidian_to_utopic", "num_neurons"],
                            ascending=[True, True],
                        )
                        ranked_pmlps_df = ranked_pmlps_df.drop(
                            columns=["euclidian_to_utopic"]
                        )
                    elif policy == "Utopic Euclidean Train Validation Holdout":
                        overall_accs = ranked_pmlps_df[
                            [train_metric, validation_metric, holdout_metric]
                        ].values
                        utopic_accs = np.array([[1.0, 1.0, 1.0]])
                        ranked_pmlps_df.loc[:, "euclidian_to_utopic"] = np.sqrt(
                            ((overall_accs - utopic_accs) ** 2).sum(1)
                        )
                        ranked_pmlps_df = ranked_pmlps_df.sort_values(
                            by=["euclidian_to_utopic", "num_neurons"],
                            ascending=[True, True],
                        )
                        ranked_pmlps_df = ranked_pmlps_df.drop(
                            columns=["euclidian_to_utopic"]
                        )
                    elif policy == "all_except_test":
                        mcdm_tuples = [
                            ("num_neurons", -1),
                            ("epoch", 1),
                            ("holdout_matthews_corrcoef", 1),
                            ("holdout_overall_acc", 1),
                            ("validation_matthews_corrcoef", 1),
                            ("validation_overall_acc", 1),
                            ("train_overall_acc", 1),
                            ("train_matthews_corrcoef", 1),
                        ]
                        ranked_pmlps_df = get_ranked_pmlps_df(
                            pmlps_df,
                            mcdm_tuples,
                            only_pareto_solutions=True,
                            sort_by_rank=True,
                        )
                    elif policy == "topsis_train_dhol":
                        mcdm_tuples = [
                            ("num_neurons", -1),
                            ("epoch", 1),
                            (holdout_metric, 1),
                            (validation_metric, 1),
                            (train_metric, 1),
                        ]
                        ranked_pmlps_df = get_ranked_pmlps_df(
                            pmlps_df,
                            mcdm_tuples,
                            only_pareto_solutions=True,
                            sort_by_rank=True,
                        )
                    elif policy == "topsis_all_overall_acc_except_test":
                        mcdm_tuples = [
                            ("num_neurons", -1),
                            ("epoch", 1),
                            (train_metric, 1),
                            (holdout_metric, 1),
                            (validation_metric, 1),
                        ]
                        ranked_pmlps_df = get_ranked_pmlps_df(
                            pmlps_df,
                            mcdm_tuples,
                            only_pareto_solutions=True,
                            sort_by_rank=True,
                        )
                    elif policy in [
                        "wilcoxon_holdout",
                        "ttest_ind_holdout",
                        "ttest_rel_holdout",
                    ]:
                        pmlps_df = pmlps_df.sort_values(
                            by=["num_neurons", "architecture_id", "fold"]
                        )
                        architecture_ids = pmlps_df["architecture_id"].unique()
                        best_arch = architecture_ids[0]
                        for current_arch in architecture_ids[1:]:
                            best_metrics = pmlps_df[
                                pmlps_df["architecture_id"] == best_arch
                            ][holdout_metric]
                            current_arch_metrics = pmlps_df[
                                pmlps_df["architecture_id"] == current_arch
                            ][holdout_metric]
                            if np.allclose(
                                current_arch_metrics.values, best_metrics.values
                            ):
                                continue
                            if policy == "wilcoxon_holdout":
                                hypothesis_test_method = wilcoxon
                            elif policy == "ttest_ind_holdout":
                                hypothesis_test_method = ttest_ind
                            elif policy == "ttest_rel_holdout":
                                hypothesis_test_method = ttest_rel

                            # H0: median(best_metrics) == median(current_arch_metrics), Ha: median(best_metrics) != median(current_arch_metrics)
                            stat, p = hypothesis_test_method(
                                best_metrics, current_arch_metrics
                            )

                            alpha = 0.05
                            if p < alpha:
                                # The one-sided test has the null hypothesis that the median is positive against the alternative that it is negative (alternative == 'less'), or vice versa (alternative == 'greater.'). https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
                                # H0: median(best_metrics) - median(current_arch_metrics) >= 0, Ha: median(best_metrics) < median(current_arch_metrics)
                                stat_alternative, less_p = hypothesis_test_method(
                                    best_metrics,
                                    current_arch_metrics,
                                    alternative="less",
                                )
                                if less_p < alpha:
                                    best_arch = current_arch

                                    if (
                                        current_arch_metrics.mean()
                                        < best_metrics.mean()
                                    ):
                                        print(
                                            f"{policy} - current_arch_metrics.mean() ({current_arch_metrics.mean()}) < best_metrics.mean() ({best_metrics.mean()}) and H0 b=c, p-value({p}), H0 b<c: less_p-value ({less_p})"
                                        )
                                        # best_arch = current_arch
                        ranked_pmlps_df = pmlps_df[
                            pmlps_df["architecture_id"] == best_arch
                        ].sort_values(by=[holdout_metric], ascending=[False])
                    else:
                        raise NotImplementedError(
                            f"Policy {policy} was not implemented."
                        )

                    choice = ranked_pmlps_df.iloc[0].copy()
                    choice["policy"] = policy
                    choices.append(choice)
        df_policy = pd.DataFrame(choices)
        df_policy.to_csv(policy_file_path)


def plot_policies(df):
    fig = px.scatter(
        df,
        x=holdout_metric,
        y=test_metric,
        color="dataset",
        # symbol="policy",
        facet_row="policy",
    )
    fig.update_layout(height=1500)
    fig.write_html(analysis_folder / "choices_per_policy.html")

    fig = px.scatter_3d(
        df,
        x=holdout_metric,
        y=test_metric,
        z="num_neurons",
        color="policy",
    )
    fig.write_html(analysis_folder / "choices_per_policy_3d.html")


def plot_box_policies(df):
    fig = px.box(
        df,
        x="policy",
        y=test_metric,
        color="dataset",
        # symbol="policy",
        # facet_row="policy",
    )
    # fig.update_layout(height=1500)
    fig.write_html(analysis_folder / "choices_per_policy_box.html")


def plot_times(df):
    df = df.drop(
        columns=[
            "min_neurons",
            "max_neurons",
            "epochs",
            "repetitions",
            "activation_functions",
            "repetitions",
            "parallel/sequential",
        ]
    )
    df = df.melt(id_vars=["num_samples", "num_features", "num_models", "device"])
    # fig = px.scatter_3d(
    #     df,
    #     x="num_features",
    #     y="num_samples",
    #     z="value",
    #     color="variable",
    #     symbol="device",
    # )
    df_parallel = df[df["variable"] == "parallel"]
    df_sequential = df[df["variable"] == "sequential"]
    df = df.rename(columns={"value": "seconds", "variable": "strategy"})
    for device in ["cuda", "cpu"]:
        for strategy in ["sequential", "parallel"]:
            # x = df_parallel[["seconds"]]
            # y = df_sequential["seconds"]
            df_tmp = df[(df["strategy"] == strategy) & (df["device"] == device)]
            x = df_tmp[["num_samples", "num_features"]]
            y = df_tmp[["seconds"]]

            reg = LinearRegression().fit(x, y)
            print(
                f"device {device}, strategy {strategy} - score: {reg.score(x, y)}, coef: {reg.coef_}, incercept: {reg.intercept_}"
            )

    # fig = px.scatter(df, x="num_samples", y="value", facet_row="variable", facet_col="num_features", trendline="ols").show()
    fig = px.scatter(
        df,
        x="num_samples",
        y="seconds",
        facet_row="strategy",
        facet_col="num_features",
        trendline="ols",
        category_orders={"num_features": [5, 10, 50, 100]},
    )  # ,
    fig.update_layout(yaxis_title="Seconds")
    # category_orders={"day": ["Thur", "Fri", "Sat", "Sun"], "time": ["Lunch", "Dinner"]})
    # fig.show()
    #     fig = make_subplots(rows=2, cols=1)
    #     fig.add_trace(
    #         go.Scatter(x=df_parallel["num_samples"], y=df_parallel["num_features"], size="value"), row=1, col=1
    #     )
    #     fig.add_trace(
    #         go.Scatter(x=df_sequential["num_samples"], y=df_sequential["num_features"], size="value"), row=2, col=1
    #     )
    #     fig.update_layout(height=600, width=800, title_text="Epoch time average to train 10,000 models.")
    fig.show()

    fig.write_html(analysis_folder / "times.html")


def distance_to_optimals(original_df, df_policies):
    best_test_df = (
        original_df.sort_values(by=[test_metric], ascending=False)
        .groupby(["experiment", "dataset", "run"])
        .head(1)
        .reset_index()
    )
    best_gap = (
        original_df.sort_values(by=[test_metric], ascending=False)
        .groupby(["experiment", "dataset", "run"])
        .head(100)
        .reset_index()
    )
    best_gap = (
        best_gap.sort_values(by=["mean_diffs"], ascending=True)
        .groupby(["experiment", "dataset", "run"])
        .head(1)
    )
    df_merge = df_policies.merge(
        best_test_df[["experiment", "dataset", "run", test_metric]],
        on=["experiment", "dataset", "run"],
        suffixes=("", "_best_test"),
    )
    df_merge = df_merge.merge(
        best_gap[["experiment", "dataset", "run", "mean_diffs"]],
        on=["experiment", "dataset", "run"],
        suffixes=("", "_best_gap"),
    )

    df_merge["distance_to_test_oracle"] = (
        df_merge[f"test_{metric}_best_test"] - df_merge[test_metric]
    ).abs()
    df_merge["distance_to_best_mean_diffs"] = (
        df_merge["mean_diffs_best_gap"] - df_merge["mean_diffs"]
    ).abs()

    fig = px.scatter(
        df_merge.groupby(["policy"]).mean().reset_index(),
        x="distance_to_best_mean_diffs",
        y="distance_to_test_oracle",
        color="policy",
        size="num_neurons"
        # symbol="policy",
        # facet_row="policy",
    )
    fig.write_html(analysis_folder / "distance_to_bests.html")

    fig = px.scatter(
        df_merge.groupby(["policy"]).mean().reset_index(),
        x=holdout_metric,
        y=test_metric,
        color="policy",
        size="num_neurons"
        # symbol="policy",
        # facet_row="policy",
    )
    fig.write_html(analysis_folder / "mean_holdout_test__policy.html")

    fig = px.scatter_3d(
        df_merge.groupby(["policy"]).mean().reset_index(),
        x=holdout_metric,
        y=test_metric,
        z=train_metric,
        color="policy",
        size="num_neurons"
        # symbol="policy",
        # facet_row="policy",
    )
    fig.write_html(analysis_folder / "3d_mean_train_holdout_test__policy.html")

    fig = px.box(
        df_merge,  # .groupby(["policy"]),#.mean().reset_index(),
        y="distance_to_test_oracle",
        # y="distance_to_best_mean_diffs",
        color="policy",
        # size="num_neurons"
        # symbol="policy",
        # facet_row="policy",
    )
    # order = df_policies.groupby(["policy"]).median().sort_values(by=[test_metric]).reset_index()["policy"].values.tolist()
    fig.write_html(analysis_folder / "policy_boxplots.html")

    # fig = make_subplots(1, 1)
    # fig.add_trace(
    #     go.Scatter(x=optimal_gap_model, y=df_parallel["num_features"], size="value"), row=1, col=1
    # )
    # fig = px.scatter(df_gap, x="num_samples", y="seconds", facet_row="strategy", facet_col="num_features", trendline="ols", category_orders={"num_features": [5, 10,50,100]})#,


def sbss_vs_nosbss_plots(df_policies):
    fig = px.box(
        df_policies,  # .groupby(["policy"]),#.mean().reset_index(),
        y=test_metric,
        x="policy",
        color="experiment",
        # size="num_neurons"
        # symbol="policy",
        # facet_row="policy",
    )
    # order = df_policies.groupby(["policy"]).median().sort_values(by=[test_metric]).reset_index()["policy"].values.tolist()
    fig.write_html(analysis_folder / "sbss_vs_nosbss.html")

    fig = px.scatter(
        df_policies.groupby(["experiment", "policy"]).mean().reset_index(),
        x=holdout_metric,
        y=test_metric,
        color="policy",
        size="num_neurons",
        symbol="experiment",
        # facet_row="policy",
    )
    fig.write_html(analysis_folder / "sbss_vs_nosbss_mean_holdout_test__policy.html")

    d = df_policies.groupby(["experiment", "policy"]).mean().reset_index()
    d2 = d.pivot_table(
        [
            train_metric,
            validation_metric,
            holdout_metric,
            test_metric,
            "num_neurons",
            "mean_diffs",
        ],
        ["policy"],
        "experiment",
    )
    d2.to_csv(analysis_folder / "sbss_vs_nosbss.csv")


# df_times = load_df_times()
# plot_times(df_times)
# df = load_df()
# apply_policies(df)
# df_policies = load_df_policies()

# print(df.shape)


# plot_holdout_test(df)
# # debug_ilpd(df)

# num_models(df)

# df_policies = apply_policies(df)

# sbss_vs_nosbss_plots(df_policies)


# distance_to_optimals(df, df_policies)
# plot_policies(df_policies)

# plot_box_policies(df_policies)

df = load_df()
apply_policies(df)
df_policies = load_df_policies()
sbss_vs_nosbss_plots(df_policies)

# df = post_process_dfs(df)
# df_policies = post_process_dfs(df_policies)
