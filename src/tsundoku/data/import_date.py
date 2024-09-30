import os
import click
import pandas as pd

from pathlib import Path

from tsundoku.data.importer import TweetImporter
from tsundoku.utils.timer import Timer
from tsundoku.utils.config import TsundokuApp


@click.command()
@click.argument("date", type=str)  # format: YYYYMMDD
@click.option("--days", default=1, type=int)
@click.option("--pattern", default="auroracl_{}.data.parquet", type=str)
@click.option("--source_path", default="", type=str)
def main(date, days, pattern, source_path):
    app = TsundokuApp("Data Importer per Date")

    project = TweetImporter(app.project_path / "config.toml")
    app.logger.info(str(project.config))

    source_path = source_path if (source_path != "") else Path(os.environ["TWEET_PATH"])
    app.logger.info("CURRENT TWEET_PATH: " + str(source_path))

    t = Timer()
    chronometer = []
    dates = []
    tweets = []
    for current_date in pd.date_range(date, freq="1D", periods=days):
        t.start()
        current_date = str(current_date.date())

        imported_tweets = project.import_date(
            current_date, pattern=pattern, source_path=source_path
        )

        current_timer = t.stop()
        chronometer.append(current_timer)
        dates.append(current_date)
        tweets.append(imported_tweets)
        app.logger.info(f"Succesfully imported {current_date} data in {current_timer} seconds!")

    app.logger.info("Chronometer: " + str(chronometer))
    app.logger.info("Chronometer dates: " + str(dates))
    app.logger.info("Imported Tweets: " + str(tweets))


if __name__ == "__main__":
    main()
