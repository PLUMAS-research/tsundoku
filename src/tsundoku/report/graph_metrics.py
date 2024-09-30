from cytoolz import itemmap
import graph_tool
import graph_tool.topology
import graph_tool.centrality
import graph_tool.clustering
import graph_tool.correlations #.assortativity
import pandas as pd 
from multiprocessing.pool import ThreadPool
import numpy as np 
import time 
import click
from tsundoku.utils.files import read_toml
import logging
from pathlib import Path
import os
import toml
from dotenv import find_dotenv, load_dotenv

from tsundoku.models.network import Network



def add_edge_property(g, id_to_label, series, dtype="float"):
    series_dict = series.to_dict()

    prop_values = [
        series_dict[
            (
                id_to_label[int(e.source())],
                id_to_label[int(e.target())],
            )
        ]
        for e in g.edges()
    ]

    series_prop = g.new_edge_property(dtype, vals=prop_values)
    g.edge_properties[series.name] = series_prop

def compute_graph_metrics(graph, dataframe, id_to_node, source_property, target_property, task):
    start = time.time()

    # Initialize results dictionary and dataframes list
    results = {}
    dataframes = []

    # PageRank
    node_pagerank = graph_tool.centrality.pagerank(graph, weight=graph.edge_properties['edge_weight'])
    pagerank_df = pd.DataFrame(node_pagerank.a, columns=['pagerank'])
    pagerank_df['user.id'] = pagerank_df.index.map(itemmap(reversed, id_to_node))
    pagerank_df = pagerank_df.set_index('user.id')
    dataframes.append(('pagerank', pagerank_df))
    results['average_pagerank'] = pagerank_df['pagerank'].mean()

    # Node Degree
    node_degree = graph.degree_property_map('total')
    degree_df = pd.DataFrame(node_degree.a, columns=['degree'])
    degree_df['user.id'] = degree_df.index.map(itemmap(reversed, id_to_node))
    degree_df = degree_df.set_index('user.id')
    dataframes.append(('degree', degree_df))
    results['average_degree'] = degree_df['degree'].mean()

    # Density
    num_vertices = graph.num_vertices()
    num_edges = graph.num_edges()
    density = (2 * num_edges) / (num_vertices * (num_vertices - 1))
    results['density'] = density

    # Clustering Coefficient
    node_clustering = graph_tool.clustering.local_clustering(graph, weight=graph.edge_properties['edge_weight'])
    clustering_df = pd.DataFrame(node_clustering.a, columns=['clustering'])
    clustering_df['user.id'] = clustering_df.index.map(itemmap(reversed, id_to_node))
    clustering_df = clustering_df.set_index('user.id')
    dataframes.append(('clustering', clustering_df))
    results['average_clustering'] = clustering_df['clustering'].mean()

    # Transitivity
    transitivity = graph_tool.clustering.global_clustering(graph)
    results['transitivity'] = transitivity[0]

    # Assortativity (Stance)
    vertex_property = add_properties(dataframe, graph, source_property , target_property)
    assortativity = graph_tool.correlations.scalar_assortativity(graph, vertex_property)
    #property_name = source_property.split('_')[1] # extract property name 
    results[f'assortativity_{task}'] = assortativity[0]

    # HITS, Authority, and Hub Scores
    hits, node_authority, node_hub = graph_tool.centrality.hits(graph)
    authority_df = pd.DataFrame(node_authority.a, columns=['authority'])
    authority_df['user.id'] = authority_df.index.map(itemmap(reversed, id_to_node))
    authority_df = authority_df.set_index('user.id')
    dataframes.append(('authority', authority_df))
    results['average_authority'] = authority_df['authority'].mean()

    hub_df = pd.DataFrame(node_hub.a, columns=['hub'])
    hub_df['user.id'] = hub_df.index.map(itemmap(reversed, id_to_node))
    hub_df = hub_df.set_index('user.id')
    dataframes.append(('hub', hub_df))
    results['average_hub'] = hub_df['hub'].mean()

    results['hits'] = hits
    results['total_time'] = time.time() - start
    return results, dataframes

def save_metrics_and_dataframes(results, dataframes, folder_path):
    # Save results dictionary as a parquet file
    results_df = pd.DataFrame([results])
    results_df.to_parquet(f'{folder_path}/metrics_graph_tool.parquet')

    # Save each dataframe separately as a parquet file
    for name, df in dataframes:
        df.to_parquet(f'{folder_path}/{name}_graph_tool.parquet')

def add_properties(dataframe, graph, source_property , target_property):

    # Initialize property maps
    source_prop_map = graph.new_vertex_property("int")
    target_prop_map = graph.new_vertex_property("int")
    combined_prop_map = graph.new_vertex_property("double")  # average stance 

    # Assign source, target, and combined properties 
    for v in graph.vertices():
        vertex_id = int(v)
        source_value = None
        target_value = None

        # Check and assign source property
        if vertex_id in dataframe.source.values:
            source_value = dataframe.loc[dataframe['source'] == vertex_id, source_property].iloc[0]
            source_prop_map[v] = source_value

        # Check and assign target property
        if vertex_id in dataframe.target.values:
            target_value = dataframe.loc[dataframe['target'] == vertex_id, target_property].iloc[0]
            target_prop_map[v] = target_value

        # Calculate and assign combined property
        if source_value is not None and target_value is not None:
            combined_prop_map[v] = (source_value + target_value) / 2.0 # average source and target values 
        elif source_value is not None:
            combined_prop_map[v] = source_value
        elif target_value is not None:
            combined_prop_map[v] = target_value
    
    return combined_prop_map



@click.command()
@click.option("--experiment", type=str, default="full")
@click.option("--group", type=str, default="relevance")
def main(experiment, group):
    logger = logging.getLogger(__name__)
    config = read_toml(Path(os.environ["TSUNDOKU_PROJECT_PATH"]) / "config.toml")[
        "project"
    ]
    logger.info(str(config))

    experiment_file = Path(config["path"]["config"]) / "experiments.toml"

    if not experiment_file.exists():
        raise FileNotFoundError(experiment_file)

    with open(experiment_file) as f:
        experiment_config = toml.load(f)
        logging.info(f"{experiment_config}")

    experimental_settings = experiment_config["experiments"][experiment]

    dataset = experimental_settings.get("key")

    processed_path = (
        Path(config["path"]["data"]) / "processed" / experimental_settings.get("key")
    )

    files_path = str(processed_path) + '/'
    consolidated_path = str(processed_path / 'consolidated') + '/'

    task = group



    # READ USERS DATA 
    df_users = pd.read_parquet(files_path + 'user.unique.parquet', engine='pyarrow') 
    dict_user_id = {x:y for x, y in zip(df_users['user.id'], df_users['user.screen_name'])}
    dict_user_names = {}
    for uid , uname in zip(df_users['user.id'] , df_users['user.name']):
        dict_user_names[uid] = uname
    df_stance = pd.read_parquet(files_path + f'{task}.classification.predictions.parquet' , engine='pyarrow')
    user_stance_dict = { id_ : stance for id_, stance in zip(df_stance['user.id'], df_stance['predicted_class'] )} 

    # STANCE MAPPING (GENERAL)
    #stance_mapping = {'empathy': 1 , 'threat': 2, 'undisclosed': 3}
    stance_mapping = {class_: i for i, class_ in enumerate(df_stance['predicted_class'].unique(), start = 1)}
    print(stance_mapping)

    # READ RETWEETS DATA 
    rts = pd.read_parquet(files_path + 'user.retweet_edges.all.parquet')# .sample(1000)
    print(rts.shape)
    rts_filtered = rts[rts['frequency'] > 1].copy() # filter frq > 1 
    print(rts_filtered.shape)
    unique_user_ids = set(rts_filtered['user.id'].unique()) | set(rts_filtered['rt.user.id'].unique()) # unique users 
    print(len(unique_user_ids))
    id_to_node = dict(zip(unique_user_ids, range(len(unique_user_ids)))) # dict idx2node 
    rts_filtered['source'] = rts_filtered['user.id'].map(id_to_node)
    rts_filtered['target'] = rts_filtered['rt.user.id'].map(id_to_node)
    rts_filtered['user.screen_name'] = rts_filtered['user.id'].map(dict_user_id)
    rts_filtered['rt_user_name'] = rts_filtered['rt.user.id'].map(dict_user_id)
    rts_filtered['user.id.stance'] = rts_filtered['user.id'].map(user_stance_dict) # add stance 
    rts_filtered['rt.user.id.stance'] = rts_filtered['rt.user.id'].map(user_stance_dict) # add stance 
    rts_filtered['user.id.stance'].fillna('undisclosed', inplace=True)
    rts_filtered['rt.user.id.stance'].fillna('undisclosed', inplace=True)
    rts_filtered['source_stance'] = rts_filtered['user.id.stance'].map(stance_mapping) # add stance as int 
    rts_filtered['target_stance'] = rts_filtered['rt.user.id.stance'].map(stance_mapping) # add stance as int 

    # create the retweet graph 
    start = time.time()
    rt_network = Network.from_edgelist(rts_filtered, weight='frequency')
    #rt_graph, rt_node_map, rt_id_to_node = graph_from_edgelist(rts_filtered, weight='frequency')

    metrics, dfs = compute_graph_metrics(rt_network.graph, rts_filtered, rt_network.id_to_label, 'source_stance', 'target_stance', task)
    save_metrics_and_dataframes(metrics, dfs, consolidated_path)

    def display_sample_dataframes_and_metrics(metrics, dataframes, num_rows=5):
        # Display a sample of each dataframe
        for name, df in dataframes:
            print(f"Sample of DataFrame '{name}':")
            print(df.head(num_rows))
            print()  # For better spacing

        # Display the key metrics
        print("Key Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

    display_sample_dataframes_and_metrics(metrics, dfs)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv(usecwd=True), override=True)

    main()