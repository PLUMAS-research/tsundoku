import gzip
import os
from glob import glob

import click
import gensim
import numpy as np
import pandas as pd
import rapidjson as json
from cytoolz import keymap, valmap
from scipy.sparse import load_npz

from tsundoku.utils.config import TsundokuApp


@click.command()
@click.argument("experiment", type=str)
@click.option("--n_topics", type=int, default=50)
def main(experiment, n_topics):
    app = TsundokuApp("LDA Topic Modelling")

    source_path = app.data_path / "raw"

    if not source_path.exists():
        raise FileNotFoundError(source_path)

    experimental_settings = app.experiment_config["experiments"][experiment]

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

    users = pd.read_parquet(
        processed_path / "consolidated" / "user.consolidated_groups.parquet"
    )
    #print(users.head())

    docterm_matrix = load_npz(processed_path / "user.tweets.matrix.npz")

    vocabulary = pd.read_parquet(
        processed_path / "user.tweet_vocabulary.relevant.parquet"
    )

    frequencies = pd.read_parquet(
        processed_path / "consolidated" / "tweet.word_frequencies.parquet"
    )

    frequent_vocabulary = vocabulary.join(
        frequencies.groupby("token")["n_users"].sum(), on="token", how="inner"
    ).pipe(
        lambda x: x[
            x["n_users"].between(
                experimental_settings.get("topic_modeling", {}).get("min_users", 1),
                x["n_users"].quantile(
                    experimental_settings.get("topic_modeling", {}).get(
                        "max_users_quantile", 1.0
                    )
                ),
            )
        ]
    )

    #print(frequent_vocabulary.head(15))

    filtered_users = users[
        users["user.dataset_tweets"].between(
            experimental_settings.get("topic_modeling", {}).get("min_tweets", 1),
            users["user.dataset_tweets"].quantile(
                experimental_settings.get("topic_modeling", {}).get(
                    "max_tweets_quantile", 1.0
                )
            ),
        )
    ]

    filtered_docterm_matrix = docterm_matrix[filtered_users["row_id"].values, :][
        :, frequent_vocabulary["token_id"].values
    ]

    corpus = gensim.matutils.Sparse2Corpus(
        filtered_docterm_matrix, documents_columns=False
    )

    lda = gensim.models.ldamulticore.LdaMulticore(
        corpus,
        num_topics=experimental_settings.get("topic_modeling", {}).get("n_topics", n_topics),
        id2word=frequent_vocabulary.reset_index()["token"].to_dict(),
        workers=experimental_settings.get("topic_modeling", {}).get("n_jobs", 1),
        passes=experimental_settings.get("topic_modeling", {}).get("passes", 1),
        alpha=experimental_settings.get("topic_modeling", {}).get("alpha", "symmetric"),
        iterations=experimental_settings.get("topic_modeling", {}).get(
            "iterations", 50
        ),
        random_state=experimental_settings.get("topic_modeling", {}).get(
            "random_state", 666
        ),
    )

    app.logger.info(lda.print_topics())

    lda.save(str(processed_path / "consolidated" / f"user.topic_model.gensim.gz"))

    frequent_vocabulary.rename(
        {"index": "topic_term_id", "token_id": "dtm_col_id"}, axis=1
    ).assign(row_id=np.arange(len(frequent_vocabulary))).drop(
        ["frequency", "n_users"], axis=1
    ).to_parquet(
        processed_path / "consolidated" / "user.topic_model.vocabulary.parquet"
    )

    with gzip.open(
        processed_path / "consolidated" / "user.topic_model.doc_topics.json.gz", "wt"
    ) as f:
        for i, (doc, uid) in enumerate(
            zip(
                gensim.matutils.Sparse2Corpus(
                    docterm_matrix[:, frequent_vocabulary["token_id"].values],
                    documents_columns=False,
                ),
                users["user.id"].values,
            )
        ):
            record = {"row_id": i, "user.id": int(uid)}
            record.update(
                valmap(float, keymap(str, dict(lda.get_document_topics(doc))))
            )
            json.dump(record, f)
            f.write("\n")


if __name__ == "__main__":
    main()
