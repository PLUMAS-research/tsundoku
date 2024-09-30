import os
import click
import dask.dataframe as dd
import pandas as pd


from glob import glob
from pathlib import Path

from tsundoku.utils.tweets import TWEET_DTYPES
from tsundoku.utils.urls import DISCARD_URLS, get_domain
from tsundoku.utils.files import write_parquet
from tsundoku.utils.timer import Timer
from tsundoku.utils.config import TsundokuApp

@click.command()
@click.argument("date", type=str)  # format: YYYYMMDD
@click.option("--days", default=1, type=int)
@click.option("--overwrite", is_flag=True)
def main(date, days, overwrite):
    app = TsundokuApp('Feature Creation')

    source_path = app.data_path / "raw"

    if not source_path.exists():
        raise FileNotFoundError(source_path)

    source_folders = sorted(glob(str(source_path / "*")))
    app.logger.info(
        f"{len(source_folders)} folders with data. {source_folders[0]} up to {source_folders[-1]}"
    )

    t = Timer()
    chronometer_total = []
    chronometer_compute_user_metrics = []
    chronometer_compute_tweet_metrics = []
    dates = []
    for i, current_date in enumerate(pd.date_range(date, freq="1D", periods=days)):
        current_date = str(current_date.date())
        tweet_path = app.data_path / "raw" / f"{current_date}"
        target = app.data_path / "interim" / f"{current_date}"

        if not target.exists():
            target.mkdir(parents=True)
            app.logger.info(f"created: {tweet_path} -> {target}")
        else:
            app.logger.info(f"{target} already exists.")
            if not overwrite:
                app.logger.info(f"skipping.")
                continue

        target_files = glob(str(Path(tweet_path) / "*.parquet"))

        non_empty_files = list(filter(lambda x: os.stat(x).st_size > 0, target_files))

        if not non_empty_files:
            app.logger.warning(f"{date} has no validfiles.")
            continue

        tweets = dd.read_parquet(non_empty_files, schema=TWEET_DTYPES)

        if tweets.npartitions <= 0:
            app.logger.warning(f"{date} has no files")
            continue

        t.start()
        app.logger.info(
            f"{current_date} ({tweets.npartitions} partitions) -> computing user metrics"
        )
        compute_user_metrics(tweets, target, overwrite)
        compute_user_metrics_time = t.stop()
        chronometer_compute_user_metrics.append(compute_user_metrics_time)

        t.start()
        app.logger.info(
            f"{current_date} ({tweets.npartitions} partitions) -> computing tweet metrics"
        )
        compute_tweet_metrics(tweets, target, overwrite)
        compute_tweet_metrics_time = t.stop()
        chronometer_compute_tweet_metrics.append(compute_tweet_metrics_time)

        app.logger.info(f"{current_date} -> done! :D")
        dates.append(current_date)
        total_time = compute_user_metrics_time + compute_tweet_metrics_time
        chronometer_total.append(total_time)
        app.logger.info(f"{current_date} -> total time: {total_time}")

    app.logger.info(f"total time: {sum(chronometer_total)}")
    app.logger.info(
        f"total time (compute_user_metrics): {sum(chronometer_compute_user_metrics)}"
    )
    app.logger.info(
        f"total time (compute_tweet_metrics): {sum(chronometer_compute_tweet_metrics)}"
    )
    app.logger.info(f"total time: {sum(chronometer_total)}")

    app.logger.info("Chronometer: " + str(chronometer_total))
    app.logger.info(
        "Chronometer (compute_user_metrics): " + str(chronometer_compute_user_metrics)
    )
    app.logger.info(
        "Chronometer (compute_tweet_metrics): " + str(chronometer_compute_tweet_metrics)
    )
    app.logger.info("Dates: " + str(dates))


def compute_user_metrics(tweets, target_path, overwrite):
    users = None
    if overwrite or not (target_path / "unique_users.parquet").exists():
        users = (
            tweets.drop_duplicates(subset="user.id")
            .compute()
            .filter(regex=r"^user\.?.*")
        )

        write_parquet(users, target_path / "unique_users.parquet")

    if overwrite or not (target_path / "user_name_vocabulary.parquet").exists():
        if users is None:
            users = dd.read_parquet(target_path / "unique_users.parquet")
        
        user_name_vocabulary = (
            users[["user.id", "user.name_tokens"]]
            .explode("user.name_tokens")
            .rename(columns={"user.name_tokens": "token"})
            .assign(token=lambda x: x["token"].str.lower())
            .groupby("token")
            .size()
            .rename("frequency")
            .sort_values()
            .to_frame()
            .reset_index()
        )

        write_parquet(user_name_vocabulary, target_path / "user_name_vocabulary.parquet")

    if overwrite or not (target_path / "user_description_vocabulary.parquet").exists():
        if users is None:
            users = dd.read_parquet(target_path / "unique_users.parquet")

        user_description_vocabulary = (
            users[["user.id", "user.description_tokens"]]
            .explode("user.description_tokens")
            .rename(columns={"user.description_tokens": "token"})
            .assign(token=lambda x: x["token"].str.lower())
            .groupby("token")
            .size()
            .rename("frequency")
            .sort_values()
            .to_frame()
            .reset_index()
        )

        write_parquet(user_description_vocabulary, target_path / "user_description_vocabulary.parquet")


def compute_tweet_metrics(tweets, target_path, overwrite):
    if overwrite or not (target_path / "tweets_per_user.parquet").exists():
        tweets_per_user = (
            tweets.drop_duplicates("id")
            .groupby("user.id")
            .size()
            .compute()
            .reset_index()
        )

        tweets_per_user.columns = tweets_per_user.columns.astype(str)
        write_parquet(tweets_per_user, target_path / "tweets_per_user.parquet")

    if overwrite or not (target_path / "tweets_list_per_user.parquet").exists():
        tweets_list_per_user = (
            tweets.drop_duplicates("id")
            .groupby("user.id")["text"]
            .agg(list)
            .reset_index()
            .compute()
        )
        tweets_list_per_user = tweets_list_per_user.rename(columns={"text": "tweets"})
        write_parquet(
            tweets_list_per_user, target_path / "tweets_list_per_user.parquet"
        )

    tweet_vocabulary = None

    if overwrite or not (target_path / "tweet_vocabulary.parquet").exists():
        tweet_vocabulary = (
            tweets.drop_duplicates("id")
            .explode("tweet.tokens")
            .groupby(["user.id", "tweet.tokens"])
            .size()
            .rename("frequency")
            .reset_index()
            .rename(columns={"tweet.tokens": "token"})
            .assign(token=lambda x: x["token"].str.lower())
            .compute()
        )

        write_parquet(tweet_vocabulary, target_path / "tweet_vocabulary.parquet")

    if overwrite or not (target_path / "tweet_token_frequency.parquet").exists():
        if tweet_vocabulary is None:
            tweet_vocabulary = dd.read_parquet(target_path / "tweet_vocabulary.parquet")

        tweet_token_frequency = (
            tweet_vocabulary.groupby("token")
            .agg(total_frequency=("frequency", "sum"), total_users=("user.id", "count"))
            .reset_index()
        )

        write_parquet(
            tweet_token_frequency, target_path / "tweet_token_frequency.parquet"
        )

        tweet_vocabulary = None

    if overwrite or not (target_path / "retweet_counts.parquet").exists():
        retweet_counts = (
            tweets[tweets["rt.id"] > 0]
            .drop_duplicates("id")
            .groupby(["rt.id", "rt.user.id"])
            .size()
            .rename("frequency")
            .compute()
            .sort_values(ascending=False)
            .reset_index()
        )

        write_parquet(retweet_counts, target_path / "retweet_counts.parquet")

    if overwrite or not (target_path / "quote_counts.parquet").exists():
        quote_counts = (
            tweets[tweets["quote.id"] > 0]
            .drop_duplicates("id")
            .groupby(["quote.id", "quote.user.id"])
            .size()
            .rename("frequency")
            .compute()
            .sort_values(ascending=False)
            .reset_index()
        )

        write_parquet(quote_counts, target_path / "quote_counts.parquet")

    if overwrite or not (target_path / "reply_counts.parquet").exists():
        reply_counts = (
            tweets[tweets["in_reply_to_user_id"] > 0]
            .drop_duplicates("id")
            .groupby(["in_reply_to_status_id", "in_reply_to_user_id"])
            .size()
            .rename("frequency")
            .compute()
            .sort_values(ascending=False)
            .reset_index()
        )

        write_parquet(reply_counts, target_path / "reply_counts.parquet")

    if overwrite or not (target_path / "retweet_edgelist.parquet").exists():
        retweet_edgelist = (
            tweets[tweets["rt.id"] > 0]
            .drop_duplicates("id")
            .groupby(["user.id", "rt.user.id"])
            .size()
            .rename("frequency")
            .compute()
            .sort_values(ascending=False)
            .reset_index()
        )

        write_parquet(retweet_edgelist, target_path / "retweet_edgelist.parquet")

    if overwrite or not (target_path / "quote_edgelist.parquet").exists():
        quote_edgelist = (
            tweets[tweets["quote.id"] > 0]
            .drop_duplicates("id")
            .groupby(["user.id", "quote.user.id"])
            .size()
            .rename("frequency")
            .compute()
            .sort_values(ascending=False)
            .reset_index()
        )

        write_parquet(quote_edgelist, target_path / "quote_edgelist.parquet")

    if overwrite or not (target_path / "reply_edgelist.parquet").exists():
        reply_edgelist = (
            tweets[tweets["in_reply_to_user_id"] > 0]
            .drop_duplicates("id")
            .groupby(["user.id", "in_reply_to_user_id"])
            .size()
            .rename("frequency")
            .compute()
            .sort_values(ascending=False)
            .reset_index()
        )

        write_parquet(reply_edgelist, target_path / "reply_edgelist.parquet")

    if overwrite or not (target_path / "user_urls.parquet").exists():
        user_urls = (
            tweets[tweets["entities.urls"].notnull()]
            .drop_duplicates("id")[["user.id", "entities.urls"]]
            .assign(**{"entities.urls": lambda x: x["entities.urls"].str.split("|")})
            .explode("entities.urls")
            .assign(domain=lambda x: x["entities.urls"].map(get_domain))
            .pipe(lambda x: x[~x["domain"].isin(DISCARD_URLS)])
            .groupby(["user.id", "domain"])
            .size()
            .rename("frequency")
            .compute()
            .sort_values(ascending=False)
            .reset_index()
        )

        write_parquet(user_urls, target_path / "user_urls.parquet")

    if overwrite or not (target_path / "daily_stats.parquet").exists():
        all_tweets = (
            tweets[
                ["id", "user.id", "rt.user.id", "quote.user.id", "in_reply_to_user_id"]
            ]
            .drop_duplicates(subset="id")
            .compute()
        )

        user_stats = (
            all_tweets.set_index("user.id")
            .astype(bool)
            .reset_index()
            .groupby("user.id")
            .sum()
        )

        plain = (
            all_tweets[
                (all_tweets["rt.user.id"] == 0)
                & (all_tweets["quote.user.id"] == 0)
                & (all_tweets["in_reply_to_user_id"] == 0)
            ]
            .groupby("user.id")
            .size()
            .rename("data.plain_count")
        )

        popularity = (
            all_tweets[all_tweets["rt.user.id"] > 0]
            .groupby("rt.user.id")
            .size()
            .rename("data.rts_received")
        )

        quotability = (
            all_tweets[all_tweets["quote.user.id"] > 0]
            .groupby("quote.user.id")
            .size()
            .rename("data.quotes_received")
        )

        conversation = (
            all_tweets[all_tweets["in_reply_to_user_id"] > 0]
            .groupby("in_reply_to_user_id")
            .size()
            .rename("data.replies_received")
        )

        user_stats = (
            user_stats.join(popularity, how="left")
            .join(plain, how="left")
            .join(quotability, how="left")
            .join(conversation, how="left")
            .fillna(0)
            .astype(int)
            .rename(
                columns={
                    "id": "data.statuses_count",
                    "rt.user.id": "data.rts_count",
                    "quote.user.id": "data.quotes_count",
                    "in_reply_to_user_id": "data.replies_count",
                }
            )
        )

        user_daily_stats = (
            dd.read_parquet(target_path / "unique_users.parquet")[
                [
                    "user.id",
                    "user.followers_count",
                    "user.friends_count",
                    "user.statuses_count",
                ]
            ]
            .set_index("user.id")
            .join(user_stats, how="inner")
            .reset_index()
        )
        user_daily_stats.to_parquet(
            target_path,
            name_function=lambda i: f"user_daily_stats{f'_{i}' if i != 0 else ''}.parquet",
            engine="pyarrow",
        )


if __name__ == "__main__":
    main()
