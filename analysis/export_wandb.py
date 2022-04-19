import pandas as pd
from pathlib import Path
import wandb

api = wandb.Api()

# Project is specified by <entity/project-name>
projects = [
    # "exp0087_politica_1_oracle_1m1l",
    # "exp0087_politica_1_oracle_1m1l_nosbss",
    # "exp0087_politica_2_best_holdout_1m1l",
    # "exp0087_politica_2_best_holdout_1m1l_nosbss",
    # "exp0087_politica_3_best_validation_1m1l",
    # "exp0087_politica_3_best_validation_1m1l_nosbss",
    # "exp0087_politica_4_best_diff_holdout_1m1l",
    # "exp0087_politica_4_best_diff_holdout_1m1l_nosbss",
    "exp0090_politica_1_oracle_1m1l",
    "exp0090_politica_2_holdout_1m1l",
    "exp0090_politica_3_best_validation_1m1l",
    "exp0090_politica_4_diff_best_holdout_1m1l",
    "exp0090_politica_1_oracle_1m1l_nosbss",
    "exp0090_politica_2_holdout_1m1l_nosbss",
    "exp0090_politica_3_best_validation_1m1l_nosbss",
    "exp0090_politica_4_diff_best_holdout_1m1l_nosbss",
]
for current_project in projects:
    runs = api.runs(f"brains/{current_project}")

    summary_list, config_list, name_list = [], [], []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k, v in run.config.items() if not k.startswith("_")}
        )

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    runs_df = pd.DataFrame(
        {"summary": summary_list, "config": config_list, "name": name_list}
    )

    df_path = Path(f"experiments/{current_project}/wandb_table.csv")
    df_path.parent.mkdir(exist_ok=True, parents=True)
    runs_df.to_csv(df_path)