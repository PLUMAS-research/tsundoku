import copy
import logging
import os
import re
import click
import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import toml

from glob import glob
from multiprocessing.pool import ThreadPool
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from gensim.utils import deaccent

from tsundoku.utils.files import read_toml
from tsundoku.models.pipeline import evaluate, prepare_features


@click.command()
@click.option("--experiment", type=str, default="full")
@click.option("--group", type=str, default="relevance")
@click.option("--n_splits", default=5, type=int)
def main(experiment, group, n_splits):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    config = read_toml(Path(os.environ["TSUNDOKU_PROJECT_PATH"]) / "config.toml")[
        "project"
    ]
    logger.info(str(config))
    dask.config.set(pool=ThreadPool(int(config.get("n_jobs", 2))))

    source_path = Path(config["path"]["data"]) / "raw"
    experiment_file = Path(config["path"]["config"]) / "experiments.toml"

    if not source_path.exists():
        raise FileNotFoundError(source_path)

    if not experiment_file.exists():
        raise FileNotFoundError(experiment_file)

    with open(experiment_file) as f:
        experiment_config = toml.load(f)
        logging.info(f"{experiment_config}")

    experimental_settings = experiment_config["experiments"][experiment]
    logging.info(f"Experimental settings: {experimental_settings}")

    source_folders = sorted(
        glob(str(source_path / experimental_settings.get("folder_pattern", "*")))
    )
    logging.info(
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
    logging.info(f"{key_folders}")

    # let's go

    data_base = Path(config["path"]["data"]) / "interim"
    processed_path = (
        Path(config["path"]["data"]) / "processed" / experimental_settings.get("key")
    )

    with open(Path(config["path"]["config"]) / "groups" / f"{group}.toml") as f:
        group_config = toml.load(f)

    # these are sorted by tweet count!
    user_ids = (
        dd.read_parquet(processed_path / "user.elem_ids.parquet")
        .set_index("user.id")
        .compute()
    )
    logging.info(f"Total users: #{len(user_ids)}")

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
    logging.info(f"Kept users for {group} prediction: #{len(user_ids)}")

    columns = [g for g in group_config.keys() if g != "noise"]

    # if 'noise' in group_config:
    #    del group_config['noise']

    labels = pd.DataFrame(
        0, index=user_ids.index, columns=group_config.keys(), dtype=int
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
    for key, meta in group_config.items():
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

    xgb_parameters = experiment_config[group].get("xgb", {
        'learning_rate': 0.25,
        'max_depth': 3,
        'subsample': 0.95,
        'n_estimators': 100,
        'max_delta_step': 1,
        'n_jobs': config.get('settings', {}).get("n_jobs", 2),
        'random_state': 42,
        'tree_method': 'hist'
    })

    pipeline_config = experiment_config[group].get("pipeline", {})

    X, labels, feature_names_all = prepare_features(
        processed_path, group_config, user_ids, labels
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
    logging.info(f"{str(outputs)}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
