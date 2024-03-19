print('GENERATING GRAPH METRICS...')

import pandas as pd 
import json 
import networkx as nx
import os
import time 

def calculate_graph_metrics(G, task):
    metrics = {}
    # Existing metrics
    metrics['average_degree'] = sum(dict(G.degree()).values()) / float(G.number_of_nodes())
    metrics['density'] = nx.density(G)
    metrics['clustering_coefficient'] = nx.average_clustering(G)
    metrics['transitivity'] = nx.transitivity(G)

    # HITS metrics
    hubs, authorities = nx.hits(G, max_iter=1000, normalized=True)
    
    # Calculate the average of hubs and authorities
    metrics['average_hub_score'] = sum(hubs.values()) / len(hubs)
    metrics['average_authority_score'] = sum(authorities.values()) / len(authorities)

    # Calculate stance assortativity 
    metrics[f'{task}_assortativity_coefficient'] = nx.attribute_assortativity_coefficient(G, 'combined_stance')
    
    return metrics, hubs , authorities

def generate_graph(df, source , target , source_stance , target_stance):
    # Create a directed graph from the DataFrame
    G = nx.from_pandas_edgelist(df, source=source, target=target, edge_attr='frequency', create_using=nx.DiGraph())

    # Add 'stance' attribute for origin (user.id) and target (in_reply_to_user_id) nodes
    stance_origin = pd.Series(df[source_stance].values, index=df[source]).to_dict()
    stance_target = pd.Series(df[target_stance].values, index=df[target]).to_dict()

    nx.set_node_attributes(G, stance_origin, 'stance_origin')
    nx.set_node_attributes(G, stance_target, 'stance_target')

    # combine stance
    for node, data in G.nodes(data=True):
        stance_origin = data.get('stance_origin', '')
        stance_target = data.get('stance_target', '')
        combined_stance = f"{stance_origin}_{stance_target}" if stance_origin and stance_target else stance_origin or stance_target
        G.nodes[node]['combined_stance'] = combined_stance

    return G

datasets = os.listdir('/Users/andrescarvallo/Desktop/tsundoku-fast/example_project/data/processed')

#datasets = [x for x in datasets if x not in ['covid_1', 'propuesta_constitucional', 'covid_2']] # datasets ya completados 

datasets = ['softwarex_final']

for dataset in datasets:

    if dataset == 'propuesta_constitucional':
        task = 'plebiscito'
    
    else:
        task = 'stance'


    print(f'Dataset provided: {dataset}')

    files_path = f'/Users/andrescarvallo/Desktop/tsundoku-fast/example_project/data/processed/{dataset}/'
    consolidated_path = f"/Users/andrescarvallo/Desktop/tsundoku-fast/example_project/data/processed/{dataset}/consolidated/"

    df_users = pd.read_parquet(files_path + 'user.unique.parquet', engine='pyarrow') 
    dict_user_id = {x:y for x, y in zip(df_users['user.id'], df_users['user.screen_name'])}

    dict_user_names = {}
    for uid , uname in zip(df_users['user.id'] , df_users['user.name']):
        dict_user_names[uid] = uname
    
    df_stance = pd.read_parquet(files_path + f'{task}.classification.predictions.parquet' , engine='pyarrow')
    user_stance_dict = { id_ : stance for id_, stance in zip(df_stance['user.id'], df_stance['predicted_class'] )} 

    # REPLIES 
    print('Replies graph metrics ...')
    start = time.time()
    df_replies = pd.read_parquet(files_path + 'user.reply_edges.all.parquet') #.sample(1000)
    df_replies = df_replies[df_replies['frequency'] > 1]
    df_replies['user.screen_name'] = df_replies['user.id'].map(dict_user_id)
    df_replies['in_reply_to_user_name'] = df_replies['in_reply_to_user_id'].map(dict_user_id)
    df_replies['user.id.stance'] = df_replies['user.id'].map(user_stance_dict) # add stance 
    df_replies['in_reply_to_user_id_stance'] = df_replies['in_reply_to_user_id'].map(user_stance_dict) # add stance 
    df_replies.dropna(inplace=True)
    df_replies = df_replies[df_replies['user.screen_name'] != df_replies['in_reply_to_user_name']]
    G_replies = generate_graph(df_replies , 
                               source= 'user.screen_name' , 
                               target= 'in_reply_to_user_name' , 
                               source_stance= 'user.id.stance'  , 
                               target_stance= 'in_reply_to_user_id_stance')
    
    replies_metrics, replies_hubs, replies_authorities  = calculate_graph_metrics(G_replies, task)

    df_hubs_replies = pd.DataFrame(replies_hubs.items(), columns=['user', 'hub_score']) 
    df_authorities_replies = pd.DataFrame(replies_authorities.items(), columns=['user', 'authority_score'])
    print(f'time: {time.time() - start} secs')

    # RETWEET 
    print('Retweet graph metrics ...')
    start = time.time()
    df_retweets = pd.read_parquet(files_path + 'user.retweet_edges.all.parquet')# .sample(1000)
    df_retweets = df_retweets[df_retweets['frequency'] > 1]
    df_retweets['user.screen_name'] = df_retweets['user.id'].map(dict_user_id)
    df_retweets['rt_user_name'] = df_retweets['rt.user.id'].map(dict_user_id)
    df_retweets['user.id.stance'] = df_retweets['user.id'].map(user_stance_dict) # add stance 
    df_retweets['rt.user.id.stance'] = df_retweets['rt.user.id'].map(user_stance_dict) # add stance 
    df_retweets.dropna(inplace=True)
    df_retweets = df_retweets[df_retweets['user.screen_name'] != df_retweets['rt_user_name']]
    G_retweets = generate_graph(df_retweets , 
                                source= 'user.screen_name', 
                                target= 'rt_user_name' , 
                                source_stance= 'user.id.stance' , 
                                target_stance= 'rt.user.id.stance')
    retweet_metrics, retweet_hubs, retweet_authorities = calculate_graph_metrics(G_retweets, task)
    df_hubs_retweet = pd.DataFrame(retweet_hubs.items(), columns=['user', 'hub_score']) 
    df_authorities_retweet = pd.DataFrame(retweet_authorities.items(), columns=['user', 'authority_score'])
    print(f'time: {time.time() - start} secs')

    # QUOTES 
    print('Quotes graph metrics ...')
    start = time.time()
    df_quotes = pd.read_parquet(files_path + 'user.quote_edges.all.parquet') #.sample(1000)
    df_quotes = df_quotes[df_quotes['frequency'] > 1]
    df_quotes['user.screen_name'] = df_quotes['user.id'].map(dict_user_id)
    df_quotes['quote_user_name'] = df_quotes['quote.user.id'].map(dict_user_id)
    df_quotes['user.id.stance'] = df_quotes['user.id'].map(user_stance_dict) # add stance 
    df_quotes['quote.user.id.stance'] = df_quotes['quote.user.id'].map(user_stance_dict) # add stance 
    df_quotes.dropna(inplace=True)
    df_quotes = df_quotes[df_quotes['user.screen_name'] != df_quotes['quote_user_name']]
    G_quotes = generate_graph(df_quotes , 
                              source= 'user.screen_name' , 
                              target= 'quote_user_name', 
                              source_stance= 'user.id.stance' , 
                              target_stance= 'quote.user.id.stance')
    quotes_metrics, quotes_hubs, quotes_authorities = calculate_graph_metrics(G_quotes, task)

    df_hubs_quotes = pd.DataFrame(quotes_hubs.items(), columns=['user', 'hub_score']) 
    df_authorities_quotes = pd.DataFrame(quotes_authorities.items(), columns=['user', 'authority_score'])
    print(f'time: {time.time() - start} secs')

    #SAVE METRICS 
    with open(consolidated_path + 'replies_metrics.json', 'w') as f:
        json.dump(replies_metrics, f)

    with open(consolidated_path + 'retweet_metrics.json', 'w') as f:
        json.dump(retweet_metrics, f)

    with open(consolidated_path + 'quotes_metrics.json', 'w') as f:
        json.dump(quotes_metrics, f)
    
    # SAVE HITS METRICS 
    df_hubs_replies.to_parquet(consolidated_path + 'hubs_replies.parquet')
    df_authorities_replies.to_parquet(consolidated_path + 'authorities_replies.parquet')
    
    df_hubs_retweet.to_parquet(consolidated_path + 'hubs_retweet.parquet')
    df_authorities_retweet.to_parquet(consolidated_path + 'authorities_retweet.parquet')
    
    df_hubs_quotes.to_parquet(consolidated_path + 'hubs_quotes.parquet')
    df_authorities_quotes.to_parquet(consolidated_path + 'authorities_quotes.parquet')
    
        


