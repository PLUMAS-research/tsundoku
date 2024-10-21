import os
import re
from glob import glob

import click
import dask.dataframe as dd
import pandas as pd
from gensim.utils import deaccent

from tsundoku.models.pipeline import classifier_pipeline, save_classifier
from tsundoku.utils.config import TsundokuApp
from tsundoku.utils.timer import Timer


@click.command("classify_users")
@click.argument("experiment", type=str)
@click.argument("group", type=str)
@click.option("--max_group_labels", type=int, default=-1)
def main(experiment, group, max_group_labels):
    app = TsundokuApp('Group Prediction')
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

    data_base = app.data_path / "interim"
    processed_path = (
        app.data_path / "processed" / experimental_settings.get("key")
    )

    group_annotations_file = (
        app.project_path / "groups" / f"{group}.annotations.csv"
    )

    if group_annotations_file.exists():
        app.logger.info("Reading annotations...")
        group_annotations = pd.read_csv(group_annotations_file)

        for key in app.group_config.keys():
            annotated_user_ids = group_annotations[group_annotations["class"] == key]
            if not annotated_user_ids.empty:
                app.logger.info(
                    f'# of annotated "{key}" accounts: {len(annotated_user_ids)}'
                )
                app.group_config[key].get("account_ids", []).extend(
                    annotated_user_ids["user.id"].unique()
                )

    user_ids = (
        dd.read_parquet(processed_path / "user.elem_ids.parquet")
        .set_index("user.id")
        .compute()
    )
    app.logger.info(f"Total users: #{len(user_ids)}")

    relevance_path = processed_path / "relevance.classification.predictions.parquet"

    # use default parameters if not provided
    xgb_parameters = app.experiment_config[group].get("xgb", {
        'learning_rate': 0.25,
        'max_depth': 3,
        'subsample': 0.95,
        'n_estimators': 100,
        'max_delta_step': 1,
        'n_jobs': app.config.get('settings', {}).get("n_jobs", 2),
        'random_state': 42,
        'tree_method': 'hist'
    })
    pipeline_config = app.experiment_config[group].get("pipeline", {})

    if "allow_list" in app.experiment_config[group]:
        allow_list_ids = app.experiment_config[group]["allow_list"].get(
            "user_ids", None
        )
        allow_id_class = app.experiment_config[group]["allow_list"].get(
            "assigned_class", "undisclosed"
        )
        app.logger.info(
            f"Whitelisted accounts: #{len(allow_list_ids)}. Using class {allow_id_class}"
        )
    else:
        allow_list_ids = None
        allow_id_class = None
        app.logger.info(f"No whitelisted accounts")

    if group != "relevance" and relevance_path.exists():
        user_groups = dd.read_parquet(relevance_path).set_index("user.id")
        # TODO: make undisclosed optional
        all_ids = user_groups.index
        valid_users = user_groups[
            ~(user_groups["predicted_class"].isin(["noise", "undisclosed"]))
        ].index

        # note that even if we discard undisclosed users, they may be present in the allow_list.
        # we check agains the full list of users
        if allow_list_ids is not None:
            valid_users = set(valid_users) | (set(allow_list_ids) & set(all_ids))

        user_ids = user_ids.loc[valid_users].sort_values("row_id")
        app.logger.info(f"Relevant users for {group} prediction: #{len(user_ids)}")

    labels = pd.DataFrame(
        0, index=user_ids.index, columns=app.group_config.keys(), dtype=int
    )

    # if there are location patterns, tream them specially:
    user_data = None

    def load_user_data():
        df = dd.read_parquet(processed_path / "user.unique.parquet")
        #print(df.head())
        #print(user_ids.index)
        df = df[df['user.id'].isin(user_ids.index)].set_index('user.id')
        return df.compute()

    for key, meta in app.group_config.items():
        group_re = None
        try:
            app.logger.info(f'location patterns for {key}, {meta["location_patterns"]}')
            group_re = re.compile("|".join(meta["location_patterns"]), re.IGNORECASE)
        except KeyError:
            app.logger.info(f"no location patterns in {key}")
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

            meta["account_ids"].extend(group_ids)
        else:
            # use them as labels
            labels[key].loc[group_ids] = 1

    # special case: age
    if group == "age":
        if user_data is None:
            user_data = load_user_data()

        min_age = 10
        max_age = 90

        def pick_age(values):
            if not values:
                return 0
            # print(values)
            for x in values[0]:
                if not x or int(x) < min_age:
                    continue
                return int(x)
            return 0

        # TODO: esto debe estar en la configuración del proyecto
        age_patterns = re.compile(
            r"(?:^|level|lvl|nivel)\s?([0-6][0-9])\.|(?:^|\W)([0-9]{2})\s?(?:años|veranos|otoños|inviernos|primaveras|years old|vueltas|lunas|soles)",
            flags=re.IGNORECASE | re.UNICODE,
        )

        found_age = (
            user_data["user.description"]
            .fillna("")
            .str.findall(age_patterns)
            .map(pick_age)
        )
        found_age = found_age[found_age.between(min_age, max_age)].copy()
        app.logger.info("found_age", found_age.shape)

        #print(found_age.sample(10))
        #print(found_age.value_counts())

        labeled_age = pd.cut(
            found_age, bins=[0, 17, 29, 39, 49, max_age + 1], labels=app.group_config.keys()
        )

        app.logger.info(labeled_age.value_counts())

        #print(labeled_age.sample(10))

        for key in app.group_config.keys():
            #print(key)
            #print((labeled_age == key).index)
            labels[key].loc[labeled_age[labeled_age == key].index] = 1

        skip_numeric_tokens = True
    else:
        skip_numeric_tokens = False

    if user_data is not None:
        user_data = None

    #print(labels.sample(10))
    #print(labels.sum())

    t = Timer()
    chronometer = []
    process_names = []
    t.start()
    clf, predictions, feature_names_all, top_terms, X = classifier_pipeline(
        processed_path,
        app.group_config,
        user_ids,
        labels,
        xgb_parameters,
        allowed_user_ids=allow_list_ids,
        allowed_users_class=allow_id_class,
        early_stopping_rounds=pipeline_config.get("early_stopping_rounds", 10),
        eval_fraction=pipeline_config.get("eval_fraction", 0.05),
        threshold_offset_factor=pipeline_config.get("threshold_offset_factor", 0.10),
        skip_numeric_tokens=skip_numeric_tokens,
        max_group_labels=max_group_labels,
    )

    current_timer = t.stop()
    chronometer.append(current_timer)
    process_names.append(f"classification")

    save_classifier(
        group, processed_path, X, clf, predictions, feature_names_all, top_terms
    )

    for loc in top_terms.columns:
        app.logger.info(loc)
        # print(top_terms.loc[relevant_features['label']].sort_values(loc, ascending=False)[loc].head(15))
        app.logger.info(
            top_terms[top_terms[loc] > 10]
            .sort_values(loc, ascending=False)[loc]
            .sample(min(25, len(top_terms[top_terms[loc] > 10])))
            .head()
        )
        app.logger.info(
            top_terms[top_terms.index.str.contains("tweet")]
            .sort_values(loc, ascending=False)[loc]
            .head()
        )
        app.logger.info(
            top_terms[top_terms.index.str.contains("#")]
            .sort_values(loc, ascending=False)[loc]
            .head()
        )
        app.logger.info(
            top_terms[top_terms.index.str.contains("@")]
            .sort_values(loc, ascending=False)[loc]
            .head()
        )
        app.logger.info(
            top_terms[top_terms.index.str.contains("domain")]
            .sort_values(loc, ascending=False)[loc]
            .head()
        )


    app.logger.info("Chronometer: " + str(chronometer))
    app.logger.info("Chronometer process names: " + str(process_names))


if __name__ == "__main__":
    main()
