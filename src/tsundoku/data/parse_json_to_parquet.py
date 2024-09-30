import logging
import os
import click
import pandas as pd

from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from tsundoku.data.importer import TweetImporter
from tsundoku.utils.timer import Timer


@click.command()
@click.argument("date", type=str)  # format: YYYYMMDD
@click.option("--days", default=1, type=int)
@click.option("--pattern", default="auroracl_{}.data.gz", type=str)
@click.option("--source_path", default="", type=str)
@click.option("--target_path", default="", type=str)
def main(date, days, pattern, source_path, target_path):
    logger = logging.getLogger(__name__)
    logger.info("Transforming from .json to .parquet for arrow library usage")
    print(os.environ)
    project = TweetImporter(Path(os.environ["TSUNDOKU_PROJECT_PATH"]) / "config.toml")
    logger.info(str(project.config))

    source_path = (
        source_path if (source_path != "") else Path(os.environ["JSON_TWEET_PATH"])
    )
    target_path = target_path if (target_path != "") else Path(os.environ["TWEET_PATH"])

    logger.info("CURRENT JSON_TWEET_PATH: " + str(source_path))
    logger.info("TARGET TWEET_PATH: " + str(target_path))

    t = Timer()
    chronometer = []
    dates = []
    tweets = []
    for i, current_date in enumerate(pd.date_range(date, freq="1D", periods=days)):
        t.start()
        current_date = str(current_date.date())

        read_tweets = project.parse_date_data_to_parquet(
            current_date,
            pattern=pattern,
            source_path=source_path,
            target_path=target_path,
        )

        current_timer = t.stop()
        chronometer.append(current_timer)
        dates.append(current_date)
        tweets.append(read_tweets)
        print(
            f"Succesfully parsed {current_date} data into parquet files in {current_timer} seconds!"
        )

    logger.info("Chronometer: " + str(chronometer))
    logger.info("Chronometer dates: " + str(dates))
    logger.info("Read Tweets: " + str(tweets))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv(usecwd=True), override=True)
    main()
