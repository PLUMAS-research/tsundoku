import click
import pandas as pd

from tsundoku.models.network import Network
from tsundoku.utils.config import TsundokuApp


@click.command("infer_communities")
@click.argument("experiment", type=str)
def main(experiment):
    app = TsundokuApp("Community Inference")

    experimental_settings = app.experiment_config["experiments"][experiment]
    app.logger.info(f"Experimental settings: {experimental_settings}")

    processed_path = app.data_path / "processed" / experimental_settings.get("key")

    users = pd.read_parquet(
        processed_path / "consolidated" / "user.consolidated_groups.parquet"
    )

    rts = pd.read_parquet(processed_path / "user.retweet_edges.all.parquet").pipe(
        lambda x: x[
            x["user.id"].isin(users["user.id"]) & x["rt.user.id"].isin(users["user.id"])
        ]
    )

    high_participation = users[users["user.dataset_tweets"] >= 1]

    edge_list = rts[
        rts["user.id"].isin(high_participation["user.id"])
        & rts["rt.user.id"].isin(high_participation["user.id"])
    ]

    rt_network = Network.from_edgelist(
        edge_list[edge_list["frequency"] >= 1],
        source="user.id",
        target="rt.user.id",
        weight="frequency",
    )
    rt_network.network

    connected_rt_network = rt_network.largest_connected_component(directed=True)

    app.logger.info(
        f"{connected_rt_network.num_vertices}, {connected_rt_network.num_edges}"
    )

    connected_rt_network.save(
        processed_path / "consolidated" / "rt_connected.network.gt"
    )

    app.logger.info("saved")

    connected_rt_network.detect_communities(
        method="hierarchical", hierarchical_covariate_type="discrete-poisson"
    )

    connected_rt_network.save(
        processed_path / "consolidated" / "communities_rt_connected.network.gt"
    )


if __name__ == "__main__":
    main()
