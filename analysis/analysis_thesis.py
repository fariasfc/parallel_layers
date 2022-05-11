import pandas as pd
import plotly.express as px
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pymcdm


# folder that contains the nosbss.csv
folder = Path("experiments/exp0090/")
folder.absolute()
df_path = folder / "politica_1_oracle_1m1l_and_nosbss.csv"
df_parquet_path = folder / "politica_1_oracle_1m1l_and_nosbss.parquet"

analysis_folder = folder / "plots"
analysis_folder.mkdir(parents=True, exist_ok=True)


def csv_to_parquet():
    df = pd.read_csv(df_path)
    df.to_parquet(df_parquet_path)


csv_to_parquet()


def load_df():
    df = pd.read_parquet(df_parquet_path)
    # Removing repeated experiments
    non_duplicated_index = (
        df[["experiment", "dataset", "run", "model_id"]].drop_duplicates()
    ).index
    df = df.loc[non_duplicated_index].reset_index()
    df["mean_diffs"] = (
        abs(df["holdout_overall_acc"] - df["train_overall_acc"])
        + abs(df["holdout_overall_acc"] - df["validation_overall_acc"])
        + abs(df["train_overall_acc"] - df["validation_overall_acc"])
    ) / 3
    print(df.keys())

    return df


def plot_holdout_test(df):
    fig = px.scatter(
        df.sort_values(by=["test_overall_acc", "holdout_overall_acc"]),
        x="holdout_overall_acc",
        y="test_overall_acc",
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
):

    pmlps_df = pmlps_df.copy()

    mcdm_keys = [k[0] for k in mcdm_tuples]

    types = [k[1] for k in mcdm_tuples]
    mcdm_method = pymcdm.methods.TOPSIS(pymcdm.normalizations.minmax_normalization)

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
    datasets = df["dataset"].unique()
    # datasets = ["diabetes", "credit-g"]
    runs = df["run"].unique()

    policies = [
        "oracle",
        "holdout",
        "validation",
        "0.01_smallest_mean_dist_holdout",
        "topsis_pareto_oracle",
        "topsis_pareto_holdout",
        "topsis_pareto_holdout_mean_diffs",
        "smallest_euclidian_holdout_test_utopic",
        "all_except_test",
        "train_validation_holdout_no_num_neurons_no_epochs",
        "topsis_pareto_train_validation_holdout",
        "topsis_pareto_oracle_holdout_train",
        "topsis_pareto_oracle_holdout_train_noepochs_nonum_neurons",
    ]
    experiments = [
        "exp0090_politica_1_oracle_1m1l",
        "exp0090_politica_1_oracle_1m1l_nosbss",
    ]
    choices = []
    for experiment in experiments:
        for dataset in tqdm(datasets):
            for run in runs:
                pmlps_df = df[
                    (df["dataset"] == dataset)
                    & (df["run"] == run)
                    & (df["experiment"] == experiment)
                ]
                pmlps_df = pmlps_df.sort_values(
                    by=["mean_diffs", "num_neurons", "holdout_overall_acc"],
                    ascending=[True, True, False],
                )
                mcdm_tuples = [
                    ("num_neurons", -1),
                    ("epoch", 1),
                    ("holdout_overall_acc", 1),
                ]
                ranked_pmlps_df_original = get_ranked_pmlps_df(
                    pmlps_df, mcdm_tuples, only_pareto_solutions=False
                )
                for policy in policies:
                    ranked_pmlps_df = ranked_pmlps_df_original.copy()

                    if policy == "oracle":
                        ranked_pmlps_df = ranked_pmlps_df.sort_values(
                            by=["test_overall_acc", "num_neurons"],
                            ascending=[False, True],
                        )
                    elif policy == "holdout":
                        ranked_pmlps_df = ranked_pmlps_df.sort_values(
                            by=["holdout_overall_acc", "num_neurons"],
                            ascending=[False, True],
                        )
                    elif policy == "validation":
                        ranked_pmlps_df = ranked_pmlps_df.sort_values(
                            by=["validation_overall_acc", "num_neurons"],
                            ascending=[False, True],
                        )
                    elif policy == "0.01_smallest_mean_dist_holdout":
                        ranked_pmlps_df = ranked_pmlps_df.iloc[
                            : int(ranked_pmlps_df.shape[0] * 0.01)
                        ]
                        ranked_pmlps_df = ranked_pmlps_df.sort_values(
                            by=["holdout_overall_acc", "num_neurons"],
                            ascending=[False, True],
                        )
                    elif policy == "topsis_pareto_oracle":
                        mcdm_tuples = [
                            ("num_neurons", -1),
                            ("epoch", 1),
                            ("test_overall_acc", 1),
                        ]
                        ranked_pmlps_df = get_ranked_pmlps_df(
                            pmlps_df,
                            mcdm_tuples,
                            only_pareto_solutions=True,
                            sort_by_rank=True,
                        )
                    elif policy == "train_validation_holdout_no_num_neurons_no_epochs":
                        mcdm_tuples = [
                            ("train_overall_acc", 1),
                            ("validation_overall_acc", 1),
                            ("holdout_overall_acc", 1),
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
                            ("train_overall_acc", 1),
                            ("validation_overall_acc", 1),
                            ("holdout_overall_acc", 1),
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
                            ("train_overall_acc", 1),
                            ("holdout_overall_acc", 1),
                            ("test_overall_acc", 1),
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
                            ("train_overall_acc", 1),
                            ("holdout_overall_acc", 1),
                            ("test_overall_acc", 1),
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
                            ("holdout_overall_acc", 1),
                        ]
                        ranked_pmlps_df = get_ranked_pmlps_df(
                            pmlps_df,
                            mcdm_tuples,
                            only_pareto_solutions=True,
                            sort_by_rank=True,
                        )
                    elif policy == "topsis_pareto_holdout_mean_diffs":
                        mcdm_tuples = [
                            ("num_neurons", -1),
                            ("mean_diffs", -1),
                            ("epoch", 1),
                            ("holdout_overall_acc", 1),
                        ]
                        ranked_pmlps_df = get_ranked_pmlps_df(
                            pmlps_df,
                            mcdm_tuples,
                            only_pareto_solutions=True,
                            sort_by_rank=True,
                        )
                    elif policy == "smallest_euclidian_holdout_test_utopic":
                        overall_accs = ranked_pmlps_df[
                            ["test_overall_acc", "holdout_overall_acc"]
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

                    elif policy == "all_except_test":
                        mcdm_tuples = [
                            ("num_neurons", -1),
                            ("epoch", 1),
                            ("holdout_overall_acc", 1),
                            ("holdout_matthews_corrcoef", 1),
                            ("validation_overall_acc", 1),
                            ("validation_matthews_corrcoef", 1),
                            ("train_overall_acc", 1),
                            ("train_matthews_corrcoef", 1),
                        ]
                        ranked_pmlps_df = get_ranked_pmlps_df(
                            pmlps_df,
                            mcdm_tuples,
                            only_pareto_solutions=True,
                            sort_by_rank=True,
                        )
                    choice = ranked_pmlps_df.iloc[0].copy()
                    choice["policy"] = policy
                    choices.append(choice)
    df_policies = pd.DataFrame(choices)
    df_policies.to_csv(analysis_folder / "choices_per_policy.csv")
    return df_policies


def plot_policies(df):
    fig = px.scatter(
        df,
        x="holdout_overall_acc",
        y="test_overall_acc",
        color="dataset",
        # symbol="policy",
        facet_row="policy",
    )
    fig.update_layout(height=1500)
    fig.write_html(analysis_folder / "choices_per_policy.html")

    fig = px.scatter_3d(
        df,
        x="holdout_overall_acc",
        y="test_overall_acc",
        z="num_neurons",
        color="policy",
    )
    fig.write_html(analysis_folder / "choices_per_policy_3d.html")


def plot_box_policies(df):
    fig = px.box(
        df,
        x="policy",
        y="test_overall_acc",
        color="dataset",
        # symbol="policy",
        # facet_row="policy",
    )
    # fig.update_layout(height=1500)
    fig.write_html(analysis_folder / "choices_per_policy_box.html")


df = load_df()

# print(df.shape)


# plot_holdout_test(df)
# # debug_ilpd(df)

# num_models(df)

df_policies = apply_policies(df)
plot_policies(df_policies)

plot_box_policies(df_policies)
