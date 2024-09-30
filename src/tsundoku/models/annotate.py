import datetime
import os
from glob import glob
from pathlib import Path

import click
import dask.dataframe as dd
import pandas as pd

from tsundoku.models.utils import load_model_features
from tsundoku.utils.config import TsundokuApp
from tsundoku.utils.tweets import TWEET_DTYPES

def peek_potentially_relevant_users(path, users, group="stance"):
    features = load_model_features(path, group, users=users)

    whitelisted_user_ids = []
    for idx, row in features[
        features["type"].isin(["RT", "REPLY", "QUOTE"])
    ].iterrows():
        print(row)
        token = row["token"]

        if type(token) == int:
            whitelisted_user_ids.append(token)

        candidate_id = row["label"].split(":")[1]

        try:
            whitelisted_user_ids.append(int(candidate_id))
        except ValueError as e:
            print(e)

    # print(whitelisted_user_ids)

    return users[users["user.id"].isin(whitelisted_user_ids)]


def tweet_dataframe(
    user_ids, data_path, experiment, folder_pattern=None, max_tweets=None
):
    if folder_pattern is None:
        folder_pattern = experiment.get("folder_pattern", "*")

    source_path = data_path / "raw"
    source_folders = sorted(glob(str(source_path / folder_pattern)))

    key_folders = list(source_folders)

    print(
        experiment.get("folder_start", "no folder start"),
        experiment.get("folder_end", "no folder end"),
    )

    if experiment.get("folder_start", None) is not None:
        key_folders = filter(
            lambda x: os.path.basename(x) >= experiment.get("folder_start"), key_folders
        )

    if experiment.get("folder_end", None) is not None:
        key_folders = filter(
            lambda x: os.path.basename(x) <= experiment.get("folder_end"), key_folders
        )

    key_folders = sorted(key_folders)

    all_tweets = []
    total_tweets = 0

    for folder in key_folders:
        try:
            tweets = (
                dd.read_parquet(
                    Path(folder) / "*.parquet", dtype=TWEET_DTYPES, meta=TWEET_DTYPES
                )
                .pipe(lambda x: x[x["user.id"].isin(user_ids)])
                .drop_duplicates("id")
                .assign(date=lambda x: x["created_at"].dt.date)[
                    ["date", "user.id", "user.screen_name", "is_retweet", "text"]
                ]
                .compute()
            )
        except KeyError:
            print(folder, "EMPTY DATE")
            continue

        if not tweets.empty:
            all_tweets.append(tweets)
            total_tweets += len(tweets)

            if max_tweets is not None and total_tweets > max_tweets:
                break

    return pd.concat(all_tweets).reset_index(drop=True)


def user_label_loop(users, options, user_func=None):
    labeled = []

    if not users.empty:
        for idx, row in users.iterrows():
            print("what do to with this user?\n")
            print(row)

            if user_func is not None:
                user_func(row["user.id"])

            action = input(f'options: [{"/".join(options)}] or [skip] or [break]\n>> ')
            if action == "skip":
                continue

            if action == "break":
                break

            if not action in options:
                print("wrong option!")
                continue

            labeled.append(
                (
                    action,
                    row["user.id"],
                    row["user.screen_name"],
                    datetime.datetime.now().isoformat(),
                )
            )

    return labeled


@click.command()
@click.argument("experiment", type=str)
@click.argument("group", type=str)
def main(experiment, group):
    pd.set_option("display.max_colwidth", None)

    app = TsundokuApp("Annotation")
    app.read_group_config(group)

    processed_path = (
        app.data_path
        / "processed"
        / app.experiment_config["experiments"][experiment]["key"]
    )

    folder_pattern = app.config.get("data_folder_pattern", "*")

    users = (
        pd.read_parquet(
            app.data_path / "processed" / experiment / "user.unique.parquet"
        )
        .join(
            pd.read_parquet(
                processed_path / f"{group}.classification.predictions.parquet"
            ).set_index("user.id"),
            on="user.id",
        )
        .pipe(
            lambda x: x[
                pd.isnull(x["reported_label"]) & pd.notnull(x["predicted_class"])
            ]
        )
    )

    annotations_file = app.project_path / "groups" / f"{group}.annotations.csv"

    if annotations_file.exists():
        annotated_user_ids = pd.read_csv(annotations_file)["user.id"].unique()
        print(f"Found {len(annotated_user_ids)} annotated accounts.")
        users = users[~users["user.id"].isin(annotated_user_ids)]

    print(f"# of users that could be labeled: {len(users)}")
    print(users.head())

    labeled = []

    potential_users = peek_potentially_relevant_users(processed_path, users, group)

    if not potential_users.empty:
        print(
            f"See if there are potential relevant users in features of {group} classifier\n"
        )

        labeled.extend(user_label_loop(potential_users, app.group_options))
    else:
        print("No users to label according to classifier features.")

    see_content = input(
        f"See if there are potential relevant users in terms of {group} content? [Y/N]\n>> "
    )
    if see_content.lower() == "y":
        print(app.group_options)

        for group_id in app.group_options:
            candidate_users = users[users["user.dataset_tweets"] >= 5]
            if candidate_users.empty:
                continue
            try:
                group_users = (
                    users[pd.isnull(users["reported_label"])]
                    .pipe(lambda x: x[x[group_id].between(0.55, 0.75)])
                    .sort_values("user.dataset_tweets", ascending=False)
                )
            except KeyError:
                continue

            if len(group_users) < 15:
                continue

            print(f"labeling potential users weakly classified as {group_id}:")
            potential_group_users = group_users.head(
                min(1000, len(group_users))
            ).sample(10)

            potential_group_user_tweets = tweet_dataframe(
                potential_group_users["user.id"].values,
                app.data_path,
                app.experiment_config["experiments"][experiment],
                max_tweets=500,
            )

            def user_func(user_id):
                user_tweets = potential_group_user_tweets.pipe(
                    lambda x: x[x["user.id"] == user_id]
                )
                if len(user_tweets) > 5:
                    user_tweets = user_tweets.sample(5)
                print("\n".join(user_tweets["text"].values))

            labeled.extend(
                user_label_loop(
                    potential_group_users, app.group_options, user_func=user_func
                )
            )

    see_connections = input(
        f"See if there are potential relevant users in terms of {group} connections? [Y/N]\n>> "
    )
    if see_connections.lower() == "y":
        rts = (
            pd.read_parquet(processed_path / "user.retweet_edges.all.parquet")
            .join(
                users.set_index("user.id")[f"predicted_class"].rename(
                    f"source.{group}"
                ),
                on="user.id",
                how="inner",
            )
            .join(
                users.set_index("user.id")[f"predicted_class"].rename(
                    f"target.{group}"
                ),
                on="rt.user.id",
                how="left",
            )
        )

        rts["cross-group"] = rts[f"source.{group}"] != rts[f"target.{group}"]

        cross_group = (
            rts[rts["cross-group"]]
            .groupby("rt.user.id")["frequency"]
            .sum()
            .sort_values(ascending=False)
        )

        cross_group = cross_group.head(min(1000, len(cross_group)))
        cross_group = cross_group.sample(min(10, len(cross_group)))

        potential_group_user_tweets = tweet_dataframe(
            cross_group.index.values,
            app.data_path,
            app.experiment_config["experiments"][experiment],
            max_tweets=1000,
        )

        def user_func(user_id):
            user_tweets = potential_group_user_tweets.pipe(
                lambda x: x[x["user.id"] == user_id]
            )
            if len(user_tweets) > 5:
                user_tweets = user_tweets.sample(5)
            print("\n".join(user_tweets["text"].values))

        labeled.extend(
            user_label_loop(
                users[users["user.id"].isin(cross_group.index)],
                app.group_options,
                user_func=user_func,
            )
        )

    #print(labeled)
    labeled_df = pd.DataFrame.from_records(
        labeled, columns=["class", "user.id", "user.screen_name", "datetime"]
    )
    app.logger.info(labeled_df)

    if annotations_file.exists():
        labeled_df = pd.concat([pd.read_csv(annotations_file), labeled_df])

    labeled_df.to_csv(annotations_file, index=False)


if __name__ == "__main__":
    main()
