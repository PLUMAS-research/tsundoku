print('GENERATING REPORTS...')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from random import choice
import numpy as np
import base64
from io import BytesIO
import os
from pathlib import Path
from tsundoku.utils.files import read_toml
import networkx as nx
import json 
import gzip 
from collections import Counter 
sns.set(style="whitegrid")
import warnings
warnings.filterwarnings('ignore')
import plotly.graph_objs as go
from datetime import datetime, timedelta
import plotly.express as px
import argparse

#task = 'plebiscito' # stance , plebiscito 

# Create the parser
parser = argparse.ArgumentParser(description="Generate report")
parser.add_argument('--dataset', help='Name of the dataset', default='social_outburst')
parser.add_argument('--graph_cut', help='Graph visualization cut', default=20)

args = parser.parse_args()

# Change task depending on the provided dataset 
if args.dataset == 'propuesta_constitucional':
    task = 'plebiscito'
else:
    task = 'stance'

# Now you can use args.dataset, which will be 'social_outburst' by default
print(f"Dataset provided: {args.dataset}")
print(f"Task: {task}")
print(f"Graph cut visualization: {args.graph_cut} most frequent")

base_path = '/Users/andrescarvallo/Desktop/tsundoku-fast/'
experiments_path = '/Users/andrescarvallo/Desktop/tsundoku-fast/example_project/experiments.toml'
consolidated_path = f"/Users/andrescarvallo/Desktop/tsundoku-fast/example_project/data/processed/{args.dataset}/consolidated/"
files_path = f'/Users/andrescarvallo/Desktop/tsundoku-fast/example_project/data/processed/{args.dataset}/'

# experiment information 
experiments = read_toml(experiments_path)
experiment_info = experiments['experiments'][args.dataset]
experiment_info['Experiment Name'] = experiment_info.pop('key')
experiment_info['Start Date'] = experiment_info.pop('folder_start')
experiment_info['End Date'] = experiment_info.pop('folder_end')
experiment_info.pop('folder_pattern', None)
df_experiment = pd.DataFrame([experiment_info]).reset_index()
df_experiment = df_experiment[['Experiment Name' , 'Start Date' , 'End Date', 'discussion_only', 'discussion_directed']]
df_experiment_html = df_experiment.to_html(classes='table table-striped center-table')

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def plot_to_html(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return f'data:image/png;base64,{image_base64}'

def plot_top_users(metric, df_users):
    top_users = df_users.nlargest(10, metric)
    fig = px.bar(top_users, x='user.name', y=metric, 
                 labels={metric: 'Count'}
                 )
    fig.update_layout(xaxis_tickangle=-45)
    plot_html = fig.to_html()
    return plot_html

def obtain_graph_html(G):
    # Setting up the color palette for nodes
    palette = sns.color_palette("husl", len(G.nodes))
    colors = [palette[i] for i in range(len(G.nodes))]

    # Choose a layout to minimize node overlaps
    pos = nx.spring_layout(G, k=0.4, iterations=50)  # Adjust k and iterations as needed

    # Increase the figure size to help spread out the nodes
    plt.figure(figsize=(20, 20))

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=700)

    # Normalize edge widths
    min_width = 0.5  # Minimum edge width
    max_width = 5    # Maximum edge width
    freq_values = [G[u][v]['frequency'] for u, v in G.edges()]
    min_freq = min(freq_values)
    max_freq = max(freq_values)
    edge_widths = [
        min_width + (G[u][v]['frequency'] - min_freq) / (max_freq - min_freq) * (max_width - min_width)
        for u, v in G.edges()
    ]

    # Draw edges with normalized widths
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=12)

    # Turn off axis and grid
    plt.axis('off')
    plt.grid(False)

    # Hide axes spines
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Convert plot to HTML
    plot_html = plot_to_html(plt.gcf())
    plt.close()
    return plot_html


def create_plotly_html_strings(top_matrix_sum, top_xgb_relevance):
    """
    This function takes pre-aggregated DataFrames and returns HTML strings for Plotly plots.
    :param top_matrix_sum: DataFrame with the top matrix_sum values.
    :param top_xgb_relevance: DataFrame with the top xgb.relevance values.
    :return: HTML strings for the Plotly plots.
    """
    # Plot 1: Top "type" with greater total "matrix_sum"
    fig1 = px.bar(top_matrix_sum, y='matrix_sum', labels={'matrix_sum': 'Total Matrix Sum', 'type': 'Type'})
    fig1.update_layout(xaxis_title="Type", yaxis_title="Total Matrix Sum", xaxis={'categoryorder': 'total descending'})
    fig1.update_xaxes(tickangle=45)
    plot_html1 = fig1.to_html()

    # Plot 2: Top "type" with higher average "xgb.relevance"
    fig2 = px.bar(top_xgb_relevance, y='xgb.relevance', labels={'xgb.relevance': 'Average XGB Relevance', 'type': 'Type'})
    fig2.update_layout(xaxis_title="Type", yaxis_title="Average XGB Relevance", xaxis={'categoryorder': 'total descending'})
    fig2.update_xaxes(tickangle=45)
    plot_html2 = fig2.to_html()
    return plot_html1, plot_html2

def plot_value_counts(value_counts):
    fig = px.bar(x=value_counts.index, y=value_counts.values, labels={'x': 'Predicted Class', 'y': 'Counts'})
    fig.update_layout()
    fig.update_xaxes(tickangle=90)
    predictions_html = fig.to_html()
    return predictions_html

def generate_graph(df, top_n, target):
    top_replies_df = df.sort_values(by='frequency', ascending=False).head(top_n)
    G = nx.from_pandas_edgelist(top_replies_df, source='user.screen_name', target= target, edge_attr='frequency', create_using=nx.DiGraph())
    return obtain_graph_html(G)

# READ DATA 
df_users = pd.read_parquet(files_path + 'user.unique.parquet', engine='pyarrow') 
df_users['user.location'].fillna('No location', inplace=True)
df_users['user.location'].replace('', 'No location', inplace=True)
dict_user_id = {x:y for x, y in zip(df_users['user.id'], df_users['user.screen_name'])}
df_users_html = df_users.head(5).to_html(classes='table table-striped center-table')

dict_user_names = {}
for uid , uname in zip(df_users['user.id'] , df_users['user.name']):
    dict_user_names[uid] = uname

# PREDICTIONS OF TYPES OF PERSONS 
df_predictions = pd.read_parquet(files_path + 'person.classification.predictions.parquet', engine='pyarrow')
df_predictions['user.name'] = df_predictions['user.id'].map(dict_user_names)

# STANCE PREDICTIONS 
df_stance = pd.read_parquet(files_path + f'{task}.classification.predictions.parquet' , engine='pyarrow')
df_stance['user.name'] = df_stance['user.id'].map(dict_user_names)

# PERSON FEATURES 
df_features = pd.read_parquet(files_path + 'person.classification.features.parquet', engine='pyarrow')

# USER NAME TOKEN COUNTS 
df_user_tokens_counts = pd.read_parquet(files_path + 'user.name_tokens.relevant.parquet', engine='pyarrow').head(10) #top 10 
df_user_tokens_counts = df_user_tokens_counts.sort_values('frequency', ascending=False) # sorting 

# USER DESCRIPTION TOKEN COUNTS 
df_user_description_tokens_counts = pd.read_parquet(files_path + 'user.description_tokens.relevant.parquet', engine='pyarrow').head(10) #top 10 
df_user_description_tokens_counts = df_user_description_tokens_counts.sort_values('frequency', ascending=False) # sorting 

# TWEETS CONTENT ANALYSIS PER STANCE CLASS 
df_tweets = pd.read_parquet(consolidated_path + 'tweet.word_frequencies.parquet' , engine='pyarrow' )

# TWEETS TIME SERIES 
df_time_series = pd.read_parquet(consolidated_path + 'user.daily_stats.parquet' , engine='pyarrow')

# PLOT USERS STATS 
user_tweets_html = plot_top_users('user.dataset_tweets', df_users)
user_followers_html = plot_top_users('user.followers_count', df_users)
user_friends_html = plot_top_users('user.friends_count', df_users)

# PLOT MOST FREQUENT LOCATIONS 
top_locations = df_users['user.location'].value_counts().head(10)
df_top_locations = pd.DataFrame({'Location': top_locations.index, 'Frequency': top_locations.values})
df_top_locations = df_top_locations.sort_values(by='Frequency', ascending=True)
fig = px.bar(df_top_locations, x='Frequency', y='Location', orientation='h',
             labels={'Frequency': 'Frequency', 'Location': 'Location'})
fig.update_layout(xaxis_title="Frequency", yaxis_title="Location")
location_plot_html = fig.to_html()

# PLOT MOST FREQUENT user.name_tokens  
fig = px.bar(df_user_tokens_counts, x='frequency', y='token',
             labels={'frequency': 'Frequency', 'token': 'Token'},
             orientation='h')  # horizontal bar chart
fig.update_layout(xaxis_title="Frequency", yaxis_title="Token")
user_tokens_plot_html = fig.to_html()

# PLOT MOST FREQUENT user.description  
fig = px.bar(df_user_description_tokens_counts, x='frequency', y='token',
             labels={'frequency': 'Frequency', 'token': 'Token'},
             orientation='h')  # horizontal bar chart
fig.update_layout(xaxis_title="Frequency", yaxis_title="Token")
user_description_tokens_plot_html = fig.to_html()

# PLOT FEATURES 
grouped_df = df_features.groupby('type').agg({'matrix_sum': 'sum', 'xgb.relevance': 'mean'})
top_matrix_sum = grouped_df.sort_values(by="matrix_sum", ascending=False).head(10)
top_xgb_relevance = grouped_df.sort_values(by="xgb.relevance", ascending=False).head(10)
plot_html1_matrix, plot_html2_xgb = create_plotly_html_strings(top_matrix_sum, top_xgb_relevance)

# PLOT PERSON TYPE PREDICTION 
value_counts = df_predictions['predicted_class'].value_counts()
prediction_html = plot_value_counts(value_counts)

# PLOT STANCE PREDICTION 
value_counts_stance = df_stance['predicted_class'].value_counts()
prediction_stance_html = plot_value_counts(value_counts_stance)

# PLOT MOST CONFIDENT MODELS PREDICTIONS USER NAMES 
classes = ['female', 'male', 'institutional']
html_plots = {}
for cls in classes:
    fig = px.bar(df_predictions[df_predictions.predicted_class == cls].sample(10, random_state = 13), y='user.name', x=cls, orientation='h', 
                 labels={'user.name': 'User Name', cls: 'Probability'})
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})  # Order by ascending probability
    html_plots[cls] = fig.to_html()

# PLOT PROBABILITY DISTRIBUTION FOR EACH CLASS
html_plots_distrib = {}

for cls in classes:
    # Create a histogram for each class to show the distribution of probabilities
    fig = px.histogram(df_predictions[df_predictions.predicted_class == cls], x=cls, nbins=20, 
                       labels={cls: 'Probability'})

    # Update layout to show x-ticks
    fig.update_layout(xaxis=dict(showticklabels=True, nticks=10),  # Adjust nticks as needed
                      yaxis=dict(title='Count'))

    # Convert the plot to HTML
    html_plots_distrib[cls] = fig.to_html()


# PLOT MOST CONFIDENT STANCE PREDICTIONS USER NAMES 
#classes_stance = ['empathy', 'threat']
classes_stance = [x for x in df_stance['predicted_class'].unique() if x != 'undisclosed']

html_plots_stance = {}
for cls in classes_stance:
    fig = px.bar(df_stance[df_stance.predicted_class == cls].sample(10, random_state = 13), y='user.name', x=cls, orientation='h', 
                 labels={'user.name': 'User Name', cls: 'Probability'})
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})  # Order by ascending probability
    html_plots_stance[cls] = fig.to_html()

# PLOT PROBABILITY DISTRIBUTION FOR EACH CLASS
html_plots_distrib_stance = {}

for cls in classes_stance:
    # Create a histogram for each class to show the distribution of probabilities
    fig = px.histogram(df_stance[df_stance.predicted_class == cls], x=cls, nbins=20, 
                       labels={cls: 'Probability'})

    # Update layout to show x-ticks
    fig.update_layout(xaxis=dict(showticklabels=True, nticks=10),  # Adjust nticks as needed
                      yaxis=dict(title='Count'))

    # Convert the plot to HTML
    html_plots_distrib_stance[cls] = fig.to_html()


# REPLIES graph report 
df_replies = pd.read_parquet(files_path + 'user.reply_edges.all.parquet')
df_replies['user.screen_name'] = df_replies['user.id'].map(dict_user_id)
df_replies['in_reply_to_user_name'] = df_replies['in_reply_to_user_id'].map(dict_user_id)
df_replies.dropna(inplace=True)
df_replies = df_replies[df_replies['user.screen_name'] != df_replies['in_reply_to_user_name']]
replies_plot_html = generate_graph(df_replies, int(args.graph_cut) , 'in_reply_to_user_name')

# RETWEETS graph report 
df_retweets = pd.read_parquet(files_path + 'user.retweet_edges.all.parquet')
df_retweets['user.screen_name'] = df_retweets['user.id'].map(dict_user_id)
df_retweets['rt_user_name'] = df_retweets['rt.user.id'].map(dict_user_id)
df_retweets.dropna(inplace=True)
df_retweets = df_retweets[df_retweets['user.screen_name'] != df_retweets['rt_user_name']]
retweet_plot_html = generate_graph(df_retweets, int(args.graph_cut) , 'rt_user_name')

# QUOTES graph report 
df_quotes = pd.read_parquet(files_path + 'user.quote_edges.all.parquet')
df_quotes['user.screen_name'] = df_quotes['user.id'].map(dict_user_id)
df_quotes['quote_user_name'] = df_quotes['quote.user.id'].map(dict_user_id)
df_quotes.dropna(inplace=True)
df_quotes = df_quotes[df_quotes['user.screen_name'] != df_quotes['quote_user_name']]
quotes_plot_html  = generate_graph(df_quotes, int(args.graph_cut) , 'quote_user_name')

# LOAD METRICS 
with open(consolidated_path + 'replies_metrics.json') as f:
    replies_metrics = json.load(f)

with open(consolidated_path + 'retweet_metrics.json') as f:
    retweet_metrics = json.load(f)

with open(consolidated_path + 'quotes_metrics.json') as f:
    quotes_metrics = json.load(f)

# VISUALIZE METRICS 
graph_metrics = [replies_metrics, retweet_metrics,quotes_metrics]
df_graphs = pd.DataFrame(graph_metrics)
df_graphs.index = ['Replies', 'Retweets', 'Quotes']
df_graphs = df_graphs.to_html(classes='table table-striped center-table')

# HITS METRICS BAR PLOTS 
df_hubs_replies = pd.read_parquet(consolidated_path + 'hubs_replies.parquet', engine = 'pyarrow').sort_values('hub_score', ascending=False).head(5)
df_authorities_replies = pd.read_parquet(consolidated_path + 'authorities_replies.parquet', engine = 'pyarrow').sort_values('authority_score', ascending=False).head(5)
df_hubs_retweet = pd.read_parquet(consolidated_path + 'hubs_retweet.parquet', engine = 'pyarrow').sort_values('hub_score', ascending=False).head(5)
df_authorities_retweet = pd.read_parquet(consolidated_path + 'authorities_retweet.parquet', engine = 'pyarrow').sort_values('authority_score', ascending=False).head(5)
df_hubs_quotes = pd.read_parquet(consolidated_path + 'hubs_quotes.parquet', engine = 'pyarrow').sort_values('hub_score', ascending=False).head(5)
df_authorities_quotes = pd.read_parquet(consolidated_path + 'authorities_quotes.parquet', engine = 'pyarrow').sort_values('authority_score', ascending=False).head(5)

# Summarizing Hubs DataFrames
df_hubs_replies = df_hubs_replies[['user']].head(5).reset_index(drop=True)
df_hubs_retweet = df_hubs_retweet[['user']].head(5).reset_index(drop=True)
df_hubs_quotes = df_hubs_quotes[['user']].head(5).reset_index(drop=True)
df_hubs_summary = pd.concat([df_hubs_replies, df_hubs_retweet, df_hubs_quotes], axis=1)
df_hubs_summary.columns = ['Hubs Replies', 'Hubs Retweet', 'Hubs Quotes']

# Summarizing Authorities DataFrames
df_authorities_replies = df_authorities_replies[['user']].head(5).reset_index(drop=True)
df_authorities_retweet = df_authorities_retweet[['user']].head(5).reset_index(drop=True)
df_authorities_quotes = df_authorities_quotes[['user']].head(5).reset_index(drop=True)
df_authorities_summary = pd.concat([df_authorities_replies, df_authorities_retweet, df_authorities_quotes], axis=1)
df_authorities_summary.columns = ['Authorities Replies', 'Authorities Retweet', 'Authorities Quotes']

# hubs and authorities summary tables 
hubs_html = df_hubs_summary.to_html(classes='table table-striped center-table')
authorities_html = df_authorities_summary.to_html(classes='table table-striped center-table')


# PLOT CONTENT ANALYSIS
token_frequencies = df_tweets.groupby([f'predicted.{task}', 'token']).agg({'frequency': 'sum'}).reset_index()
grouped = token_frequencies.groupby(f'predicted.{task}')

html_token_plots = {}
for name, group in grouped:
    # Get top 10 tokens for each stance
    top_tokens = group.nlargest(10, 'frequency')
    # Create Plotly figure
    fig = px.bar(top_tokens, x='token', y='frequency', title = f"Top 10 Most Frequent Tokens in {name.capitalize()} Class")
    # Convert figure to HTML and store in dictionary
    html_token_plots[name] = fig.to_html()

# TIME SERIES PLOT  
# Convert 'date' to datetime if it's not already
df_time_series['date'] = pd.to_datetime(df_time_series['date'])

# Group by date and sum the counts
grouped_df = df_time_series.groupby('date').agg({'data.rts_count': 'sum', 
                                     'data.quotes_count': 'sum', 
                                     'data.replies_count': 'sum'})

# Calculate the total count for each date
grouped_df['total_count'] = grouped_df[['data.rts_count', 'data.quotes_count', 'data.replies_count']].sum(axis=1)

grouped_df = grouped_df.reset_index()

# Create the plot with Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=grouped_df['date'], y=grouped_df['data.rts_count'], fill='tozeroy', name='Retweets'))
fig.add_trace(go.Scatter(x=grouped_df['date'], y=grouped_df['data.replies_count'], fill='tozeroy', name='Replies'))
fig.add_trace(go.Scatter(x=grouped_df['date'], y=grouped_df['data.quotes_count'], fill='tozeroy', name='Quotes'))

# Updating layout
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Count',
    hovermode='x'
)

# Convert to HTML string
time_plot_html = fig.to_html()

# Save HTML report
html_report = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Experiment Report {args.dataset}</title>
    <!-- Bootstrap CSS CDN -->
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
        #table-of-contents {{
            font-size: 1.2em; /* Adjust the size as needed */
        }}
        .table-container {{
            overflow-x: auto;
        }}
        .center-table {{
            margin-left: auto;
            margin-right: auto;
        }}
        th {{
            text-align: center; /* Center-align table headers */
        }}
        .small-table {{
            font-size: 0.85em; /* Smaller font size for the first table */
        }}
        .table-striped tbody tr:nth-of-type(odd) {{
            background-color: #f2f2f2; /* Light gray background for odd rows */
        }}
        .plot img {{
            max-width: 100%;
            height: auto;
        }}
        .plot {{
            text-align: center; /* Center-align plots */
        }}
        .back-to-top {{
            margin-bottom: 20px; /* Adds space below the button */
        }}
    </style>
</head>

<body>
    <div class="container mt-4">
        <h1 class="text-center mb-4">Experiment Report: {args.dataset}</h1>

        <!-- Table of Contents -->
        <div id="table-of-contents" class="mb-3">
            <h2>Table of Contents</h2>
            <ul>
                <li><a href="#section1">Users Information</a></li>
                <li><a href="#section2">Social Network Graphs and Metrics</a></li>
                <li><a href="#section3">User Type Predictions</a></li>
                <li><a href="#section4">User Classification Predictions</a></li>
                <li><a href="#section5">Social Media Engagement through time</a></li>
            </ul>
        </div>

        <div class="container mt-4">
        <h3>Experiment details</h3>
        {df_experiment_html}
        </div> 

        <!-- Section 1 Users Statistics -->
        <div id="section1" class="mb-3">
        <div class="mb-3">
            <h2>Users Information</h2>
            {df_users_html}
        </div>

        <div class="row">
            <div class="col-md-6 plot mb-3">
                <h2>Top 10 Users (#Tweets) </h2>
                {user_tweets_html}
            </div>

            <div class="col-md-6 plot mb-3">
                <h2>Top 10 Users (#Followers) </h2>
                {user_followers_html}
            </div>
        </div>

        <div class="row">
            <div class="col-md-6 plot mb-3">
                <h2>Top 10 Users (#Friends) </h2>
                {user_friends_html}
            </div>

            <div class="col-md-6 plot mb-3">
                <h2>Relevant User Name Tokens Frequency</h2>
                {user_tokens_plot_html}
            </div>

            <div class="col-md-6 plot mb-3">
                <h2>Relevant User Description Tokens Frequency</h2>
                {user_description_tokens_plot_html}
            </div>

            <div class="col-md-6 plot mb-3">
                <h2>Top 10 User Locations</h2>
                {location_plot_html}
            </div>

            <div class="col-md-6 plot mb-3">
                <h2>Top 10 Types with Greater Total 'matrix_sum'</h2>
                {plot_html1_matrix}
            </div>

            <div class="col-md-6 plot mb-3">
                <h2>Top 10 Types with Greater Total 'xgb.relevance'</h2>
                {plot_html2_xgb}
            </div>
        <a href="#" class="btn btn-primary back-to-top">Back to Top</a>
        </div>
        
        
        <!-- Section 2 Social Network Graphs and Metrics -->
        <div id="section2" class="plot mb-3">
        <div class="plot mb-3">
            <h2>Tweet Replies Graph: {args.graph_cut} most frequent </h2>
            <img src="{replies_plot_html}" alt="Replies Graph">
        </div>

        <div class="plot mb-3">
            <h2>Retweets Graph: {args.graph_cut} most frequent </h2>
            <img src="{retweet_plot_html}" alt="Replies Graph">
        </div>

        <div class="plot mb-3">
            <h2>Tweet Quotes Graph: {args.graph_cut} most frequent </h2>
            <img src="{quotes_plot_html}" alt="Replies Graph">
        </div>

        <div class="mb-3">
            <h2>Graph Metrics </h2>
            {df_graphs}
        </div>

<div class="container">
    <!-- Hubs and Authorities Section -->
    <div class="row">
        <div class="col-md-12">
            <h2>Top 5 Users Hubs for each graph</h2>
            {hubs_html}
        </div>
    </div>
    <div class="row">
        <div class="col-md-12">
            <h2>Top 5 Users Authorities for each Graph </h2>
            {authorities_html}
        </div>
    </div>
    <a href="#" class="btn btn-primary back-to-top">Back to Top</a>
 </div>

    
    <!-- Section 3 User Type Predictions -->
    <div id="section3" class="plot mb-3">
    <div class="plot mb-3">
        <h2> Most Frequent User Type Predictions </h2>
        {prediction_html}
    </div>

    <!-- Sample and Distribution Sections -->
    <div class="row">
        <div class="col-md-6 plot mb-3">
            <h2>Sample - Female Class</h2>
            {html_plots['female']}
        </div>
        <div class="col-md-6 plot mb-3">
            <h2>Distribution - Female Class</h2>
            {html_plots_distrib['female']}
        </div>
    </div>

    <div class="row">
        <div class="col-md-6 plot mb-3">
            <h2>Sample - Male Class</h2>
            {html_plots['male']}
        </div>
        <div class="col-md-6 plot mb-3">
            <h2>Distribution - Male Class</h2>
            {html_plots_distrib['male']}
        </div>
    </div>

    <div class="row">
        <div class="col-md-6 plot mb-3">
            <h2>Sample - Institutional Class</h2>
            {html_plots['institutional']}
        </div>
        <div class="col-md-6 plot mb-3">
            <h2>Distribution - Institutional Class</h2>
            {html_plots_distrib['institutional']}
        </div>
    </div>
    <a href="#" class="btn btn-primary back-to-top">Back to Top</a>
</div>
    
    <!-- Section 4 User Classification Predictions -->
    <div id="section4" class="plot mb-3">
    <div class="plot mb-3">
        <h2> Most Frequent User {task.capitalize()} Predictions </h2>
        {prediction_stance_html}
    </div>

    <div class="row">
        <div class="col-md-6 plot mb-3">
            <h2>Sample - {classes_stance[0].capitalize()} Class</h2>
            {html_plots_stance[classes_stance[0]]}
        </div>
        <div class="col-md-6 plot mb-3">
            <h2>Distribution - {classes_stance[0].capitalize()} Class</h2>
            {html_plots_distrib_stance[classes_stance[0]]}
        </div>
        <div class="col-md-6 plot mb-3">
            <h2>Sample - {classes_stance[1].capitalize()} Class</h2>
            {html_plots_stance[classes_stance[1]]}
        </div>
        <div class="col-md-6 plot mb-3">
            <h2>Distribution - {classes_stance[1].capitalize()} Class</h2>
            {html_plots_distrib_stance[classes_stance[1]]}
        </div>
    </div>

    <div class="plot mb-3">
        <h2>Token frequency for each {task} class</h2>
        <div>
            {html_token_plots[classes_stance[0]]}
        </div>
        <div>
            {html_token_plots[classes_stance[1]]}
        </div>
    </div>
    <a href="#" class="btn btn-primary back-to-top">Back to Top</a>
    </div>

    
    <!-- Section 5 Social Media Engagement through time -->
    <div id="section5" class="plot mb-3">
    <div class="plot mb-3">
        <h2>Social Media Engagement through time</h2>
        {time_plot_html}
    </div>
    <a href="#" class="btn btn-primary back-to-top">Back to Top</a>
    </div>
</div>


    <!-- Bootstrap JS and Popper.js (optional) -->
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.9.2/dist/umd/popper.min.js"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
</body>
</html>
"""

with open(f'/Users/andrescarvallo/Desktop/tsundoku-fast/reports/report_{args.dataset}.html', 'w') as file:
    file.write(html_report)

