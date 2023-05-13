# -*- coding: utf-8 -*-
import logging
import os
import click
import pandas as pd
import gzip

from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from tsundoku.data.importer import TweetImporter
from tsundoku.features.timer import Timer


@click.command()
@click.argument("date", type=str)
@click.option("--days", default=1, type=int)
@click.option("--encoding", default="utf-8", type=str)
@click.option("--pattern", default="auroracl_{}.data.parquet", type=str)
def main(date, days, encoding, pattern):
    logger = logging.getLogger(__name__)
    logger.info("Making final dataset from raw data using arrow files")

    project = TweetImporter(Path(os.environ["TSUNDOKU_PROJECT_PATH"]) / "config.toml")
    logger.info(str(project.config))

    source_path = Path(os.environ["TWEET_PATH"]) / "parquet"
    logger.info("CURRENT TWEET_PATH: " + str(source_path))

    t = Timer()
    chronometer = []
    dates = []
    for i, current_date in enumerate(pd.date_range(date, freq="1D", periods=days)):
        t.start()
        current_date = str(current_date.date())
        project.import_date_from_arrow(
            current_date, pattern=pattern, source_path=source_path)
        current_timer = t.stop()
        chronometer.append(current_timer)
        dates.append(current_date)

    logger.info("Chronometer: " + str(chronometer))
    logger.info("Chronometer dates: " + str(dates))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())

    main()