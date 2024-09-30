import os
from glob import glob

import click

from tsundoku.utils.config import TsundokuApp
from tsundoku.utils.timer import Timer

from .functions import (
    aggregate_daily_stats,
    consolidate_users,
    identify_network_lcc,
    sum_word_frequencies_per_group,
)


@click.command()
@click.argument("experiment", type=str)
@click.argument("group", type=str)
@click.option("--overwrite", is_flag=True)
def main(experiment, group, overwrite):
    app = TsundokuApp("Experimental Analysis")

    source_path = app.data_path / "raw"
    experimental_settings = app.experiment_config["experiments"][experiment]

    if not source_path.exists():
        raise FileNotFoundError(source_path)

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

    group_names = glob(str(app.project_path / "groups" / "*.toml"))
    group_names = list(map(lambda x: os.path.basename(x).split(".")[0], group_names))
    app.logger.info(f"Group names: {group_names}")

    # let's go

    data_base = app.data_path / "interim"
    data_paths = [data_base / key for key in key_folders]
    processed_path = app.data_path / "processed" / experimental_settings.get("key")
    target_path = processed_path / "consolidated"

    if not target_path.exists():
        target_path.mkdir(parents=True)
        app.logger.info(f"{target_path} created")

    t = Timer()
    chronometer = []
    process_names = []

    t.start()
    user_ids = consolidate_users(
        processed_path, target_path, group, overwrite=overwrite, group_names=group_names
    )
    current_timer = t.stop()
    chronometer.append(current_timer)
    process_names.append(f"consolidate_users")

    t.start()
    aggregate_daily_stats(
        data_paths, processed_path, user_ids, target_path, group, overwrite=overwrite
    )
    current_timer = t.stop()
    chronometer.append(current_timer)
    process_names.append(f"aggregate_daily_stats")

    t.start()
    sum_word_frequencies_per_group(
        data_paths, processed_path, target_path, group, overwrite=overwrite
    )
    current_timer = t.stop()
    chronometer.append(current_timer)
    process_names.append(f"sum_word_frequencies_per_group")

    t.start()
    identify_network_lcc(
        processed_path,
        target_path,
        user_ids,
        "rt.user.id",
        min_freq=app.experiment_config["thresholds"].get("edge_weight", 1),
        network_type="retweet",
    )
    current_timer = t.stop()
    chronometer.append(current_timer)
    process_names.append(f"identify_network_retweets")

    t.start()
    identify_network_lcc(
        processed_path,
        target_path,
        user_ids,
        "quote.user.id",
        min_freq=app.experiment_config["thresholds"].get("edge_weight", 1),
        network_type="quote",
    )
    current_timer = t.stop()
    chronometer.append(current_timer)
    process_names.append(f"identify_network_quotes")

    t.start()
    identify_network_lcc(
        processed_path,
        target_path,
        user_ids,
        "in_reply_to_user_id",
        min_freq=app.experiment_config["thresholds"].get("edge_weight", 1),
        network_type="reply",
    )
    current_timer = t.stop()
    chronometer.append(current_timer)
    process_names.append(f"identify_network_replies")

    app.logger.info("Chronometer: " + str(chronometer))
    app.logger.info("Chronometer process names: " + str(process_names))


if __name__ == "__main__":
    main()
