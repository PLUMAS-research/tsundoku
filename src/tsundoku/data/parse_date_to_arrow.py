# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv

from tsundoku.data.importer import TweetImporter
import gzip


@click.command()
@click.argument("date", type=str)
@click.option("--days", default=1, type=int)
@click.option("--encoding", default="utf-8", type=str)
@click.option("--pattern", default="auroracl_{}.data.gz", type=str)
def main(date, days, encoding, pattern):
    logger = logging.getLogger(__name__)
    logger.info("Transforming from .json to .parquet for arrow library usage")

    project = TweetImporter(Path(os.environ["TSUNDOKU_PROJECT_PATH"]) / "config.toml")
    logger.info(str(project.config))

    source_path = Path(os.environ["TWEET_PATH"])

    logger.info("CURRENT TWEET_PATH: " + str(source_path))

    for i, current_date in enumerate(pd.date_range(date, freq="1D", periods=days)):
        current_date = str(current_date.date())
        project.parse_date_to_arrow(
            current_date, pattern=pattern, source_path=source_path)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())
    main()
