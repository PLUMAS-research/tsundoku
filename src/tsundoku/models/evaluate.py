import os
import re
from glob import glob

import click
import dask.dataframe as dd
import pandas as pd
from gensim.utils import deaccent

from tsundoku.models.pipeline import evaluate, prepare_features
from tsundoku.utils.config import TsundokuApp


@click.command("evaluate_classifier")
@click.argument("experiment", type=str)
@click.argument("group", type=str)
@click.option("--n_splits", default=5, type=int)
def main(experiment, group, n_splits):
    app = TsundokuApp("Classifier Evaluation")
    app.read_group_config(group)

    source_path = app.data_path / "raw"

    if not source_path.exists():
        raise FileNotFoundError(source_path)

    experimental_settings = app.experiment_config["experiments"][experiment]
    app.logger.info(f"Experimental settings: {experimental_settings}")

    source_folders = sorted(
        glob(str(source_path / experimental_settings.get("folder_pattern", "*")))
    )
    app.logger.info(
        f"{len(source_folders)} folders with data. {source_folders[0]} up to {source_folders[-1]}"
    )

    key_folders = map(os.path.basename, source_folders)

    if experimental_settings.get("folder_start", None) is not None:
        key_folders = filter(
            lambda x: x >= experimental_settings.get("folder_start"), key_folders
        )

    if experimental_settings.get("folder_end", None) is not None:
        key_folders = filter(
            lambda x: x <= experimental_settings.get("folder_end"), key_folders
        )

    key_folders = list(key_folders)
    app.logger.info(f"{key_folders}")

    # let's go
    processed_path = app.data_path / "processed" / experimental_settings.get("key")

    # these are sorted by tweet count!
    user_ids = (
        dd.read_parquet(processed_path / "user.elem_ids.parquet")
        .set_index("user.id")
        .compute()
    )
    app.logger.info(f"Total users: #{len(user_ids)}")

    # user_ids['rank_quartile'] = pd.qcut(user_ids['row_id'], 20, retbins=False, labels=range(20))

    # we discard noise to evaluate, including in stance classification!
    user_groups = (
        dd.read_parquet(
            processed_path / "relevance.classification.predictions.parquet", lines=True
        )
        .compute()
        .set_index("user.id")
    )
    valid_users = user_groups[
        ~(user_groups["predicted_class"].isin(["noise", "undisclosed"]))
    ].index
    user_ids = user_ids.loc[valid_users].sort_values("row_id")
    app.logger.info(f"Kept users for {group} prediction: #{len(user_ids)}")

    columns = [g for g in app.group_config.keys() if g != "noise"]

    # if 'noise' in group_config:
    #    del group_config['noise']

    labels = pd.DataFrame(
        0, index=user_ids.index, columns=app.group_config.keys(), dtype=int
    )

    def load_user_data():
        return (
            dd.read_parquet(processed_path / "user.unique.parquet")
            .set_index("user.id")
            .loc[user_ids.index]
            .compute()
        )

    # if there are location patterns, tream them specially:
    user_data = None
    for key, meta in app.group_config.items():
        group_re = None
        try:
            print(f'location patterns for {key}, {meta["location"]["patterns"]}')
            group_re = re.compile("|".join(meta["location"]["patterns"]), re.IGNORECASE)
        except KeyError:
            print(f"no location patterns in {key}")
            continue

        if user_data is None:
            user_data = load_user_data()
            user_data["user.location"] = (
                user_data["user.location"].fillna("").map(deaccent)
            )

        group_ids = user_data[user_data["user.location"].str.contains(group_re)].index

        if group == "location":
            # use these as account ids that cannot be modified (let's trust users)
            if not "account_ids" in meta:
                meta["account_ids"] = dict()

            if not "known_users" in meta:
                meta["account_ids"]["known_users"] = list(group_ids)
            else:
                meta["account_ids"]["known_users"].extend(group_ids)
        else:
            # use them as labels
            labels[key].loc[group_ids] = 1

    xgb_parameters = app.experiment_config[group].get(
        "xgb",
        {
            "learning_rate": 0.25,
            "max_depth": 3,
            "subsample": 0.95,
            "n_estimators": 100,
            "max_delta_step": 1,
            "n_jobs": app.config.get("settings", {}).get("n_jobs", 2),
            "random_state": 42,
            "tree_method": "hist",
        },
    )

    pipeline_config = app.experiment_config[group].get("pipeline", {})

    X, labels, feature_names_all = prepare_features(
        processed_path, app.group_config, user_ids, labels
    )

    print("Evaluating...")
    outputs = evaluate(
        processed_path,
        xgb_parameters,
        X,
        labels,
        group,
        training_eval_fraction=pipeline_config.get("eval_fraction", 0.05),
        n_splits=n_splits,
    )
    app.logger.info(f"{str(outputs)}")


if __name__ == "__main__":
    main()
