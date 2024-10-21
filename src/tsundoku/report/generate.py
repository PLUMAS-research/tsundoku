import base64
import shutil
from glob import glob
from io import BytesIO

import click
import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
from cytoolz import keyfilter
from matplotlib.colors import rgb2hex

from tsundoku.utils.config import TsundokuApp


@click.command("generate_report")
@click.argument("experiment", type=str)
@click.argument("group", type=str)
@click.option("--graph_cut", type=int, default=500)
def main(experiment, group, graph_cut):
    app = TsundokuApp("Report Generation")
    app.read_group_config(group)

    app.logger.info("GENERATING REPORTS...")
    # Change task depending on the provided dataset
    task = group

    # Now you can use args.dataset, which will be 'social_outburst' by default
    app.logger.info(f"Dataset provided: {experiment}")
    app.logger.info(f"Task: {group}")
    app.logger.info(f"Graph cut visualization: {graph_cut} most frequent")

    js_files = glob(str(app.tsundoku_path.resolve() / "vendor" / "*.js"))
    app.logger.info(f'JS files: {", ".join(js_files)}')

    experiment_info = app.experiment_config["experiments"][experiment]

    processed_path = app.data_path / "processed" / experiment_info.get("key")

    report_path = app.project_path / "reports" / experiment

    if not report_path.exists():
        report_path.mkdir(parents=True)

    files_path = str(processed_path) + "/"
    consolidated_path = str(processed_path / "consolidated") + "/"

    experiment_info["Experiment Name"] = experiment_info.pop("key")
    experiment_info["Start Date"] = experiment_info.pop("folder_start")
    experiment_info["End Date"] = experiment_info.pop("folder_end")
    experiment_info.pop("folder_pattern", None)
    df_experiment = pd.DataFrame([experiment_info]).reset_index()
    df_experiment = df_experiment[
        [
            "Experiment Name",
            "Start Date",
            "End Date",
            "discussion_only",
            "discussion_directed",
        ]
    ]
    df_experiment_html = df_experiment.to_html(
        classes="table table-striped center-table"
    )

    def flatten_dict(d, parent_key="", sep="_"):
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
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        return f"data:image/png;base64,{image_base64}"

    def plot_top_users(metric, df_users):
        top_users = df_users.nlargest(10, metric)
        fig = px.bar(top_users, x="user.name", y=metric, labels={metric: "Count"})
        fig.update_layout(xaxis_tickangle=-45)
        plot_html = fig.to_html()
        return plot_html

    def create_plotly_html_strings(top_matrix_sum, top_xgb_relevance):
        """
        This function takes pre-aggregated DataFrames and returns HTML strings for Plotly plots.
        :param top_matrix_sum: DataFrame with the top matrix_sum values.
        :param top_xgb_relevance: DataFrame with the top xgb.relevance values.
        :return: HTML strings for the Plotly plots.
        """
        # Plot 1: Top "type" with greater total "matrix_sum"
        fig1 = px.bar(
            top_matrix_sum,
            y="matrix_sum",
            labels={"matrix_sum": "Total Matrix Sum", "type": "Type"},
        )
        fig1.update_layout(
            xaxis_title="Type",
            yaxis_title="Total Matrix Sum",
            xaxis={"categoryorder": "total descending"},
        )
        fig1.update_xaxes(tickangle=45)
        plot_html1 = fig1.to_html()

        # Plot 2: Top "type" with higher average "xgb.relevance"
        fig2 = px.bar(
            top_xgb_relevance,
            y="xgb.relevance",
            labels={"xgb.relevance": "Average XGB Relevance", "type": "Type"},
        )
        fig2.update_layout(
            xaxis_title="Type",
            yaxis_title="Average XGB Relevance",
            xaxis={"categoryorder": "total descending"},
        )
        fig2.update_xaxes(tickangle=45)
        plot_html2 = fig2.to_html()
        return plot_html1, plot_html2

    def plot_value_counts(value_counts):
        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            labels={"x": "Predicted Class", "y": "Counts"},
        )
        fig.update_layout()
        fig.update_xaxes(tickangle=90)
        predictions_html = fig.to_html(include_plotlyjs=False, full_html=False)
        return predictions_html

    def generate_graph(df, top_n, target, group_membership, group_colors, report_path):
        top_replies_df = df.sort_values(by="frequency", ascending=False).head(top_n)
        G = nx.from_pandas_edgelist(
            top_replies_df,
            source="user.screen_name",
            target=target,
            edge_attr="frequency",
            create_using=nx.DiGraph(),
        )

        # print(group_membership.keys())
        node_attr = keyfilter(lambda x: x in G.nodes, group_membership)
        node_color = {key: group_colors[value] for key, value in node_attr.items()}
        #print(node_attr)
        nx.set_node_attributes(G, node_attr, name="group")
        nx.set_node_attributes(G, node_color, name="color")
        # print(G.nodes(data=True))
        nx.write_gexf(G, report_path / f"{target}.gexf")
        return ""
        # return obtain_graph_html(G)

    # READ DATA
    df_users = pd.read_parquet(files_path + "user.unique.parquet", engine="pyarrow")
    df_users["user.location"].fillna("No location", inplace=True)
    df_users["user.location"].replace("", "No location", inplace=True)
    dict_user_id = {
        x: y for x, y in zip(df_users["user.id"], df_users["user.screen_name"])
    }
    df_users_html = df_users.head(5).to_html(classes="table table-striped center-table")

    dict_user_names = {}
    for uid, uname in zip(df_users["user.id"], df_users["user.name"]):
        dict_user_names[uid] = uname

    # PREDICTIONS OF TYPES OF PERSONS
    # df_predictions = pd.read_parquet(files_path + 'person.classification.predictions.parquet', engine='pyarrow')
    # df_predictions['user.name'] = df_predictions['user.id'].map(dict_user_names)

    # STANCE PREDICTIONS
    df_stance = pd.read_parquet(
        files_path + f"{task}.classification.predictions.parquet", engine="pyarrow"
    )
    df_stance["user.name"] = df_stance["user.id"].map(dict_user_names)
    df_stance["user.screen_name"] = df_stance["user.id"].map(
        df_users.set_index("user.id")["user.screen_name"]
    )

    group_membership = df_stance.set_index("user.screen_name")[
        "predicted_class"
    ].to_dict()

    group_colors = dict(
        zip(
            app.experiment_config[group]["order"],
            app.experiment_config[group].get(
                "colors",
                map(
                    rgb2hex,
                    sns.color_palette(
                        "husl", n_colors=len(app.experiment_config[group]["order"])
                    ),
                ),
            ),
        )
    )

    # PERSON FEATURES
    # df_features = pd.read_parquet(files_path + 'person.classification.features.parquet', engine='pyarrow')

    # USER NAME TOKEN COUNTS
    df_user_tokens_counts = pd.read_parquet(
        files_path + "user.name_tokens.relevant.parquet", engine="pyarrow"
    ).head(
        10
    )  # top 10
    df_user_tokens_counts = df_user_tokens_counts.sort_values(
        "frequency", ascending=False
    )  # sorting

    # USER DESCRIPTION TOKEN COUNTS
    df_user_description_tokens_counts = pd.read_parquet(
        files_path + "user.description_tokens.relevant.parquet", engine="pyarrow"
    ).head(
        10
    )  # top 10
    df_user_description_tokens_counts = df_user_description_tokens_counts.sort_values(
        "frequency", ascending=False
    )  # sorting

    # TWEETS CONTENT ANALYSIS PER STANCE CLASS
    df_tweets = pd.read_parquet(
        consolidated_path + "tweet.word_frequencies.parquet", engine="pyarrow"
    )

    # TWEETS TIME SERIES
    df_time_series = pd.read_parquet(
        consolidated_path + "user.daily_stats.parquet", engine="pyarrow"
    )

    # PLOT USERS STATS
    user_tweets_html = plot_top_users("user.dataset_tweets", df_users)
    user_followers_html = plot_top_users("user.followers_count", df_users)
    user_friends_html = plot_top_users("user.friends_count", df_users)

    # PLOT MOST FREQUENT LOCATIONS
    top_locations = df_users["user.location"].value_counts().head(10)
    df_top_locations = pd.DataFrame(
        {"Location": top_locations.index, "Frequency": top_locations.values}
    )
    df_top_locations = df_top_locations.sort_values(by="Frequency", ascending=True)
    fig = px.bar(
        df_top_locations,
        x="Frequency",
        y="Location",
        orientation="h",
        labels={"Frequency": "Frequency", "Location": "Location"},
    )
    fig.update_layout(xaxis_title="Frequency", yaxis_title="Location")
    location_plot_html = fig.to_html(include_plotlyjs=False, full_html=False)

    # PLOT MOST FREQUENT user.name_tokens
    fig = px.bar(
        df_user_tokens_counts,
        x="frequency",
        y="token",
        labels={"frequency": "Frequency", "token": "Token"},
        orientation="h",
    )  # horizontal bar chart
    fig.update_layout(xaxis_title="Frequency", yaxis_title="Token")
    user_tokens_plot_html = fig.to_html(include_plotlyjs=False, full_html=False)

    # PLOT MOST FREQUENT user.description
    fig = px.bar(
        df_user_description_tokens_counts,
        x="frequency",
        y="token",
        labels={"frequency": "Frequency", "token": "Token"},
        orientation="h",
    )  # horizontal bar chart
    fig.update_layout(xaxis_title="Frequency", yaxis_title="Token")
    user_description_tokens_plot_html = fig.to_html(
        include_plotlyjs=False, full_html=False
    )

    # PLOT STANCE PREDICTION
    value_counts_stance = df_stance["predicted_class"].value_counts()
    prediction_stance_html = plot_value_counts(value_counts_stance)

    classes_stance = [
        x for x in df_stance["predicted_class"].unique() if x != "undisclosed"
    ]

    html_plots_stance = {}
    for cls in classes_stance:
        fig = px.bar(
            df_stance[df_stance.predicted_class == cls].sample(10, random_state=13),
            y="user.name",
            x=cls,
            orientation="h",
            labels={"user.name": "User Name", cls: "Probability"},
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"}
        )  # Order by ascending probability
        html_plots_stance[cls] = fig.to_html(include_plotlyjs=False, full_html=False)

    # PLOT PROBABILITY DISTRIBUTION FOR EACH CLASS
    html_plots_distrib_stance = {}

    for cls in classes_stance:
        # Create a histogram for each class to show the distribution of probabilities
        fig = px.histogram(
            df_stance[df_stance.predicted_class == cls],
            x=cls,
            nbins=20,
            labels={cls: "Probability"},
        )

        # Update layout to show x-ticks
        fig.update_layout(
            xaxis=dict(showticklabels=True, nticks=10),  # Adjust nticks as needed
            yaxis=dict(title="Count"),
        )

        # Convert the plot to HTML
        html_plots_distrib_stance[cls] = fig.to_html(
            include_plotlyjs=False, full_html=False
        )

    # REPLIES graph report
    df_replies = pd.read_parquet(files_path + "user.reply_edges.all.parquet")
    df_replies["user.screen_name"] = df_replies["user.id"].map(dict_user_id)
    df_replies["in_reply_to_user_name"] = df_replies["in_reply_to_user_id"].map(
        dict_user_id
    )
    df_replies.dropna(inplace=True)
    df_replies = df_replies[
        df_replies["user.screen_name"] != df_replies["in_reply_to_user_name"]
    ]
    replies_plot_html = generate_graph(
        df_replies,
        graph_cut,
        "in_reply_to_user_name",
        group_membership,
        group_colors,
        report_path,
    )

    # RETWEETS graph report
    df_retweets = pd.read_parquet(files_path + "user.retweet_edges.all.parquet")
    df_retweets["user.screen_name"] = df_retweets["user.id"].map(dict_user_id)
    df_retweets["rt_user_name"] = df_retweets["rt.user.id"].map(dict_user_id)
    df_retweets.dropna(inplace=True)
    df_retweets = df_retweets[
        df_retweets["user.screen_name"] != df_retweets["rt_user_name"]
    ]
    retweet_plot_html = generate_graph(
        df_retweets,
        graph_cut,
        "rt_user_name",
        group_membership,
        group_colors,
        report_path,
    )

    # QUOTES graph report
    df_quotes = pd.read_parquet(files_path + "user.quote_edges.all.parquet")
    df_quotes["user.screen_name"] = df_quotes["user.id"].map(dict_user_id)
    df_quotes["quote_user_name"] = df_quotes["quote.user.id"].map(dict_user_id)
    df_quotes.dropna(inplace=True)
    df_quotes = df_quotes[df_quotes["user.screen_name"] != df_quotes["quote_user_name"]]
    quotes_plot_html = generate_graph(
        df_quotes,
        graph_cut,
        "quote_user_name",
        group_membership,
        group_colors,
        report_path,
    )

    # PLOT CONTENT ANALYSIS
    token_frequencies = (
        df_tweets.groupby([f"predicted.{task}", "token"])
        .agg({"frequency": "sum"})
        .reset_index()
    )
    grouped = token_frequencies.groupby(f"predicted.{task}")

    html_token_plots = {}
    for name, group in grouped:
        # Get top 10 tokens for each stance
        top_tokens = group.nlargest(10, "frequency")
        # Create Plotly figure
        fig = px.bar(
            top_tokens,
            x="token",
            y="frequency",
            title=f"Top 10 Most Frequent Tokens in {name.capitalize()} Class",
        )
        # Convert figure to HTML and store in dictionary
        html_token_plots[name] = fig.to_html(include_plotlyjs=False, full_html=False)

    # TIME SERIES PLOT
    # Convert 'date' to datetime if it's not already
    df_time_series["date"] = pd.to_datetime(df_time_series["date"])

    # Group by date and sum the counts
    grouped_df = df_time_series.groupby("date").agg(
        {
            "data.rts_count": "sum",
            "data.quotes_count": "sum",
            "data.replies_count": "sum",
        }
    )

    # Calculate the total count for each date
    grouped_df["total_count"] = grouped_df[
        ["data.rts_count", "data.quotes_count", "data.replies_count"]
    ].sum(axis=1)

    grouped_df = grouped_df.reset_index()

    # Create the plot with Plotly
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=grouped_df["date"],
            y=grouped_df["data.rts_count"],
            fill="tozeroy",
            name="Retweets",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grouped_df["date"],
            y=grouped_df["data.replies_count"],
            fill="tozeroy",
            name="Replies",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grouped_df["date"],
            y=grouped_df["data.quotes_count"],
            fill="tozeroy",
            name="Quotes",
        )
    )

    # Updating layout
    fig.update_layout(xaxis_title="Date", yaxis_title="Count", hovermode="x")

    # Convert to HTML string
    time_plot_html = fig.to_html(include_plotlyjs=False, full_html=False)

    # Save HTML report
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Experiment Report {experiment}</title>
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

            .scrollable {{
                overflow-x: scroll;
            }}

            .network-graph {{
                margin: auto;
                height: 1200px;
                text-align: left;
            }}

            .network-graph canvas {{
                margin: 0;
                padding: 0;
                width: 1200px;
                height: 1200px;
            }}
        </style>
    </head>

    <body>
        <div class="container mt-4">
            <h1 class="text-center mb-4">Experiment Report: {experiment}</h1>

            <!-- Table of Contents -->
            <div id="table-of-contents" class="mb-3">
                <h2>Table of Contents</h2>
                <ul>
                    <li><a href="#section1">Users Information</a></li>
                    <li><a href="#section2">Social Network Graphs and Metrics</a></li>
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

                

            
            <a href="#" class="btn btn-primary back-to-top">Back to Top</a>
            </div>
            
            
            <!-- Section 2 Social Network Graphs and Metrics -->
            <div id="section2" class="plot mb-3">
            <div class="plot mb-3">
                <h2>Tweet Replies Graph: {graph_cut} most frequent </h2>
                <div id="reply-graph" class="network-graph"></div>
            </div>

            <div class="plot mb-3">
                <h2>Retweets Graph: {graph_cut} most frequent </h2>
                <div id="retweet-graph" class="network-graph"></div>
            </div>

            <div class="plot mb-3">
                <h2>Tweet Quotes Graph: {graph_cut} most frequent </h2>
                <div id="quote-graph" class="network-graph"></div>
            </div>

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
        <script src="plotly-2.35.2.min.js"></script>
        <script src="graphology.min.js"></script>
    <script src="graphology-library.min.js"></script>
    <script src="sigma.min.js"></script>
    <script src="plotly-2.35.2.min.js"></script>
    <script>
            fetch("./rt_user_name.gexf")
            .then((res) => res.text())
            .then((res) => {{
                let graph = graphologyLibrary.gexf.parse(graphology.Graph, res);
                const container = document.getElementById("retweet-graph");

                const degrees = graph.nodes().map((node) => graph.degree(node));
                const minDegree = Math.min(...degrees);
                const maxDegree = Math.max(...degrees);
                const minSize = 3,
                    maxSize = 30;
                graph.forEachNode((node) => {{
                    const degree = graph.degree(node);
                    graph.setNodeAttribute(
                        node,
                        "size",
                        minSize + ((degree - minDegree) / (maxDegree - minDegree)) * (maxSize - minSize),
                    );
                }});

                graphologyLibrary.layout.circular.assign(graph);

                graphologyLibrary.layoutForceAtlas2.assign(graph, {{
                    iterations: 150, getEdgeWeight: 'frequency',
                    settings: graphologyLibrary.layoutForceAtlas2.inferSettings(graph)
                }});

                console.log(graph)

                const renderer = new Sigma(graph, container);
            }});

            fetch("./quote_user_name.gexf")
            .then((res) => res.text())
            .then((res) => {{
                let graph = graphologyLibrary.gexf.parse(graphology.Graph, res);
                const container = document.getElementById("quote-graph");

                const degrees = graph.nodes().map((node) => graph.degree(node));
                const minDegree = Math.min(...degrees);
                const maxDegree = Math.max(...degrees);
                const minSize = 3,
                    maxSize = 30;
                graph.forEachNode((node) => {{
                    const degree = graph.degree(node);
                    graph.setNodeAttribute(
                        node,
                        "size",
                        minSize + ((degree - minDegree) / (maxDegree - minDegree)) * (maxSize - minSize),
                    );
                }});

                graphologyLibrary.layout.circular.assign(graph);

                graphologyLibrary.layoutForceAtlas2.assign(graph, {{
                    iterations: 150, getEdgeWeight: 'frequency',
                    settings: graphologyLibrary.layoutForceAtlas2.inferSettings(graph)
                }});

                console.log(graph)

                const renderer = new Sigma(graph, container);
            }});

            fetch("./in_reply_to_user_name.gexf")
            .then((res) => res.text())
            .then((res) => {{
                let graph = graphologyLibrary.gexf.parse(graphology.Graph, res);
                const container = document.getElementById("reply-graph");

                const degrees = graph.nodes().map((node) => graph.degree(node));
                const minDegree = Math.min(...degrees);
                const maxDegree = Math.max(...degrees);
                const minSize = 3,
                    maxSize = 30;
                graph.forEachNode((node) => {{
                    const degree = graph.degree(node);
                    graph.setNodeAttribute(
                        node,
                        "size",
                        minSize + ((degree - minDegree) / (maxDegree - minDegree)) * (maxSize - minSize),
                    );
                }});

                graphologyLibrary.layout.circular.assign(graph);

                graphologyLibrary.layoutForceAtlas2.assign(graph, {{
                    iterations: 150, getEdgeWeight: 'frequency',
                    settings: graphologyLibrary.layoutForceAtlas2.inferSettings(graph)
                }});

                console.log(graph)

                const renderer = new Sigma(graph, container);
            }});
    </script>
    </body>
    </html>
    """

    with open(report_path / f"index.html", "w") as file:
        file.write(html_report)

    for filename in js_files:
        shutil.copy2(filename, str(report_path.resolve()))
        app.logger.info(f"Copied: {filename}")


if __name__ == "__main__":
    main()
